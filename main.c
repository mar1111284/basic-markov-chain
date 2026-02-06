#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <sys/stat.h>
#include <stddef.h>

// --- Threading ---
#define NUM_THREADS 8

// --- Text / Generation Limits ---
#define MAX_LINE_LEN 4096
#define WORD_MAX_LEN 128
#define MAX_WORDS_PER_GENERATION 1000
#define MAX_CANDIDATES 600

// --- Tables / Hashing ---
#define HASH_TABLE_SIZE (1 << 21)        // hash table for words
#define STARTER_TABLE_SIZE (1 << 17)     // starter words table
#define INTERN_TABLE_SIZE (1 << 20)      // string interning table

// --- Memory / Arena ---
#define ARENA_BLOCK_SIZE (1ULL << 22)    // memory arena block size
#define CONTEXT_BUF_SIZE 1024             // buffer for context

// --- Keywords / Synonyms ---
#define MAX_KEYWORDS 8
#define MAX_SYNONYMS_PER_WORD 32
#define MAX_SYN_ENTRIES 8192

// --- Generation Defaults ---
#define DEFAULT_MIN_TEXT_LENGTH 50        // safety if argument missing
#define DEFAULT_MAX_TEXT_LENGTH 60        // safety if argument missing

typedef struct App App;

// ---------------------------
// Candidate generated word info
// ---------------------------
typedef struct {
    const char **words;   // Array of token strings
    int         length;   // Number of tokens
    double      ppl;      // Perplexity or probability measure
    int         rep_count;// Repetition count for scoring
    double      score;    // Overall score
} Candidate;

// ---------------------------
// N-gram context key (up to 4 words)
// ---------------------------
typedef struct {
    const char *w[4];
    uint8_t     order;    // Actual number of words in context
} ContextKey;

// ---------------------------
// Arena memory block for fast allocation
// ---------------------------
typedef struct ArenaBlock {
    struct ArenaBlock *next;
    size_t used;
    char data[ARENA_BLOCK_SIZE];
} ArenaBlock;

// ---------------------------
// Interned string table entry (linked list)
// ---------------------------
typedef struct InternEntry {
    const char *str;
    struct InternEntry *next;
} InternEntry;

// ---------------------------
// Linked list of possible next words
// ---------------------------
typedef struct Next {
    const char *word;
    int count;            // Frequency count
    struct Next *next;
} Next;

// ---------------------------
// Hashtable entry for n-gram model
// ---------------------------
typedef struct Entry {
    ContextKey  key;
    Next       *nexts;       // Linked list of next words
    uint32_t    total_count; // Sum of next word counts
    struct Entry *next;      // Collision chain
} Entry;

// ---------------------------
// Starter words for generation
// ---------------------------
typedef struct Starter {
    const char *word;
    int count;              // Frequency count
    struct Starter *next;   // Linked list chaining
} Starter;

// ---------------------------
// Synonym entry for thematic generation
// ---------------------------
typedef struct {
    const char *word;                         // Main word
    const char *syns[MAX_SYNONYMS_PER_WORD];  // Synonyms array
    int num_syns;                             // Number of synonyms
} SynEntry;

// ---------------------------
// Thread-local data for multi-threaded generation
// ---------------------------
typedef struct {
    const char *filename;          
    long start_byte;               
    long end_byte;       
    Entry **local_hashtable;       // Per-thread hash table
    Starter **local_starters;      // Per-thread starter list
    long local_total_starters;     
    long local_vocab;              
    ArenaBlock *local_arena;       // Per-thread memory arena
} ThreadData;

typedef struct {
    Entry **local;
    Entry **global;
    size_t start_bucket;
    size_t end_bucket;
} MergeRangeArgs;

// ---------------------------
// Thematic starter word with frequency
// ---------------------------
typedef struct {
    const char *word;
    long count;                     // Frequency for weighted selection
} ThematicStarter;


typedef struct App {
    // ------------------------------------------------------------
    //  1. Configuration & Command-line / runtime parameters
    // ------------------------------------------------------------
    char          *input_file;          // main training corpus
    char          *output_file;         // (currently unused?)
    char          *syn_file;            // synonyms file path (default: synonyms.txt)

    int            min_gen_len;
    int            max_gen_len;
    int            max_candidates;

    int            num_threads;
    unsigned int   seed;
    int            verbose;             // 0 = silent, 1 = info, 2 = debug, ...

    // ------------------------------------------------------------
    //  2. Keyword / Theme filtering
    // ------------------------------------------------------------
    const char   **keywords;            // array of original keywords (interned)
    int            keyword_count;

    const char    *keyword_syn_pool[2048];  // flattened pool of keywords + synonyms
    int            keyword_syn_pool_count;

    ThematicStarter *thematic_starters;     // precomputed weighted thematic starters
    int             thematic_starter_count;
    long            total_thematic_weight;

    // ------------------------------------------------------------
    //  3. Repetition & generation quality control
    // ------------------------------------------------------------
    int            rep_window;          // how far back to check for repetition
    double         rep_penalty;         // multiplier when repetition is detected (e.g. 0.3)
    double         temperature;         // 0.0 = deterministic, 1.0 = original distribution, >1.0 = more random
    double         average_theme_density;   // average over all accepted candidates
    int            top_k;

    // ------------------------------------------------------------
    //  4. Model tables & sizes
    // ------------------------------------------------------------
    Entry        **hashtable;           // main n-gram → successors table
    size_t         hashtable_size;

    Starter      **starter_table;       // first-word frequency table
    size_t         starter_table_size;

    InternEntry  **intern_table;        // string interning hash table
    size_t         intern_table_size;

    long           vocab_size;          // approximate number of unique tokens
    long           total_starters;      // total count of first-word occurrences

    atomic_long    total_lines_processed;  // progress tracking across threads
    long      	   total_lines;

    // ------------------------------------------------------------
    //  5. Candidate generation state
    // ------------------------------------------------------------
    Candidate      candidates[MAX_CANDIDATES];
    int            candidate_count;
    int            best_candidate_index;
    double         best_score;
    int            generation_attempts;
    double 		   theme_force_probability;
    double         min_theme_density;

    // ------------------------------------------------------------
    //  6. Memory management – arena allocator
    // ------------------------------------------------------------
    ArenaBlock    *arena_head;          // first block in chain
    ArenaBlock    *arena_current;       // block we're currently allocating from

    // Optional / debug statistics
    size_t         arena_total_bytes_used;
    size_t         arena_block_count;

    // ------------------------------------------------------------
    //  7. Custom object pools (freelists) & allocation stats
    // ------------------------------------------------------------
    Entry         *entry_freelist;
    Next          *next_freelist;

    // Optional counters (useful for debugging leaks / performance)
    long           entry_alloc_count;
    long           next_alloc_count;
    long           entry_from_freelist;
    long           next_from_freelist;

} App;

static App g_app;
static pthread_mutex_t intern_mutex = PTHREAD_MUTEX_INITIALIZER;

// ---------------------------
// Debug / Display
// ---------------------------
void display_global_debug(void);
void display_candidates(void);
static void print_words_properly(FILE *out, const char **words, int count);

// ---------------------------
// Application initialization
// ---------------------------
static int app_init(void);
static int parse_arguments(int argc, char **argv);

// ---------------------------
// Model lifecycle
// ---------------------------
static void free_model(void);
int build_model_mt(void);

// ---------------------------
// Word / Text utilities
// ---------------------------
static inline bool is_punct(const char *tok);
int split_into_words(char *line, const char **tokens_out, int max_tokens);
long get_lines_nb(const char *filename);

// ---------------------------
// Synonym / Thematic utilities
// ---------------------------
int load_synonyms(void);
static inline bool is_keyword_related(const char *w);
static inline bool is_in_syn_pool(const char *interned_str);
static const char *select_thematic_starter(void);
void precompute_thematic_starters(void);

// ---------------------------
// Candidate generation / scoring
// ---------------------------
static int generate_sequence(const char **out_words);
static void generate_multiple_candidates(void);
static double score_candidate(const char **words, int nw);
static double compute_perplexity(const char **words, int nw);
static void select_and_print_best(void); // naive for now
// ---------------------------
// Context / Hash utilities
// ---------------------------
static inline uint64_t xxhash64(const char *str);           // xxHash - very fast
static inline unsigned long fnv64_hash(const char *str);
static inline uint64_t hash_context(const ContextKey *c);   // fnv1a style
static inline bool context_equal(const ContextKey *a, const ContextKey *b);
static inline const char *intern(const char *s);

// ---------------------------
// N-gram / Transition table management
// ---------------------------
void add_transition_local_thread(Entry **table,const char *window[4],const char *next_word,ArenaBlock **arena);
void merge_local_into_global(Entry **local, Entry **global);
void merge_starters(Starter **local, Starter **global);
void add_starter_local(Starter **table, long *total, const char *word);

// ---------------------------
// Thread / chunk processing
// ---------------------------
void *process_chunk(void *arg);

// ---------------------------
// Arena memory management
// ---------------------------
void arena_init(void);
char *arena_alloc(size_t size);
char *arena_alloc_thread(ArenaBlock **arena, size_t size);
Next *alloc_next_thread_arena(ArenaBlock **arena);
Entry *alloc_entry_thread_arena(ArenaBlock **arena);

// ---------------------------
// Alloc / Free N-gram structures
// ---------------------------
Next *alloc_next(void);
void free_next(Next *n);
Entry *alloc_entry(void);
void free_entry(Entry *e);

static int app_init(void)
{
    // ------------------------------------------------------------
    //  0. Zero out the entire struct first (safest starting point)
    // ------------------------------------------------------------
    memset(&g_app, 0, sizeof(g_app));

    // ------------------------------------------------------------
    //  1. Config / Files
    // ------------------------------------------------------------
    g_app.input_file  = NULL;
    g_app.output_file = NULL;
    g_app.syn_file    = "synonyms.txt";           // default value

    // ------------------------------------------------------------
    //  2. Generation parameters
    // ------------------------------------------------------------
    g_app.min_gen_len     = DEFAULT_MIN_TEXT_LENGTH;
    g_app.max_gen_len     = DEFAULT_MAX_TEXT_LENGTH;
    g_app.max_candidates  = MAX_CANDIDATES;
	g_app.temperature = 1.2;
	g_app.top_k = 8;
	g_app.rep_penalty = 0.15;
	g_app.rep_window              = 10;       // 8–16 is fine
	g_app.theme_force_probability  = 0.4;     // lower = less forced jumps
	g_app.average_theme_density = 0.0;
	g_app.min_theme_density = 0.05;


    // ------------------------------------------------------------
    //  3. Threading & performance
    // ------------------------------------------------------------
    g_app.num_threads = NUM_THREADS;

    // ------------------------------------------------------------
    //  4. Randomness & logging
    // ------------------------------------------------------------
    g_app.seed    = (unsigned)time(NULL);
    g_app.verbose = 1;                    // default: info level
    srand(g_app.seed);

    // ------------------------------------------------------------
    //  5. Keyword / Theme filtering
    // ------------------------------------------------------------
    g_app.keyword_count           = 0;
    g_app.keywords                = NULL;                    // allocated later if needed
    g_app.keyword_syn_pool_count  = 0;
    memset(g_app.keyword_syn_pool, 0, sizeof(g_app.keyword_syn_pool));

    g_app.thematic_starters       = NULL;
    g_app.thematic_starter_count  = 0;
    g_app.total_thematic_weight   = 0;

    // ------------------------------------------------------------
    //  6. Model table sizes (power-of-2 preferred for fast modulo)
    // ------------------------------------------------------------
    g_app.hashtable_size     = HASH_TABLE_SIZE;       // 1<<21 = ~2M buckets
    g_app.starter_table_size = STARTER_TABLE_SIZE;    // 1<<17 = 131072
    g_app.intern_table_size  = INTERN_TABLE_SIZE;     // 1<<20 = ~1M buckets

    // ------------------------------------------------------------
    //  7. Model state & statistics
    // ------------------------------------------------------------
    g_app.hashtable       = NULL;                     // allocated in build_model_mt
    g_app.starter_table   = NULL;                     // allocated in build_model_mt
    g_app.intern_table    = NULL;                     // allocated below
    g_app.vocab_size      = 0;
    g_app.total_starters  = 0;
    atomic_init(&g_app.total_lines_processed, 0);
    g_app.total_lines     = 0;

    // ------------------------------------------------------------
    //  8. Candidate generation state
    // ------------------------------------------------------------
    g_app.candidate_count       = 0;
    g_app.best_candidate_index  = -1;
    g_app.best_score            = 0.0;
    g_app.generation_attempts   = 0;
    memset(g_app.candidates, 0, sizeof(g_app.candidates));

    // ------------------------------------------------------------
    //  9. Memory management (arena)
    // ------------------------------------------------------------
    g_app.arena_head    = NULL;
    g_app.arena_current = NULL;
    arena_init();           // creates the first block
    g_app.arena_total_bytes_used = 0;
    g_app.arena_block_count      = 0;

    // ------------------------------------------------------------
    // 10. Custom freelists for Entry and Next nodes
    // ------------------------------------------------------------
    g_app.entry_freelist = NULL;
    g_app.next_freelist  = NULL;

    g_app.entry_alloc_count       = 0;
    g_app.next_alloc_count        = 0;
    g_app.entry_from_freelist     = 0;
    g_app.next_from_freelist      = 0;

    // ------------------------------------------------------------
    // 11. Allocate the interning hash table
    // ------------------------------------------------------------
    g_app.intern_table = calloc(g_app.intern_table_size, sizeof(InternEntry *));
    if (!g_app.intern_table) {
        fprintf(stderr, "app_init: failed to allocate intern_table (%zu entries)\n",
                g_app.intern_table_size);
        return -1;
    }

    return 0;
}

void display_global_debug(void) {
    printf("\n\n=== App Global Debug Dump ===\n\n");

    // ------------------------------------------------------------------------
    // Configuration / Runtime Parameters
    // ------------------------------------------------------------------------
    printf("--- Configuration ---\n");
    printf("  input_file     : %s\n", g_app.input_file     ? g_app.input_file     : "(null)");
    printf("  output_file    : %s\n", g_app.output_file    ? g_app.output_file    : "(null)");
    printf("  syn_file       : %s\n", g_app.syn_file       ? g_app.syn_file       : "(null)");
    printf("  min_gen_len    : %d\n", g_app.min_gen_len);
    printf("  max_gen_len    : %d\n", g_app.max_gen_len);
    printf("  max_candidates : %d\n", g_app.max_candidates);
    printf("  rep_window     : %d\n", g_app.rep_window);
    printf("  rep_penalty    : %.2f\n", g_app.rep_penalty);
    printf("  num_threads    : %d\n", g_app.num_threads);
    printf("  seed           : %u\n", g_app.seed);
    printf("  verbose        : %d\n", g_app.verbose);
    printf("\n");

    // ------------------------------------------------------------------------
    // Keyword / Theme Filters
    // ------------------------------------------------------------------------
    printf("--- Keyword Filters ---\n");
    printf("  keyword_count  : %d\n", g_app.keyword_count);
    if (g_app.keyword_count > 0 && g_app.keywords) {
        printf("  keywords       : ");
        for (int i = 0; i < g_app.keyword_count; i++) {
            printf("%s%s", g_app.keywords[i], (i < g_app.keyword_count - 1) ? ", " : "\n");
        }
    } else {
        printf("  keywords       : (none)\n");
    }

    printf("--- Synonym / Theme Pool ---\n");
    printf("  keyword_syn_pool_count : %d\n", g_app.keyword_syn_pool_count);
    if (g_app.keyword_syn_pool_count > 0) {
        int counted_keywords = 0;
        for (int i = 0; i < g_app.keyword_syn_pool_count; i++) {
            for (int k = 0; k < g_app.keyword_count; k++) {
                if (g_app.keyword_syn_pool[i] == g_app.keywords[k]) {
                    counted_keywords++;
                    break;
                }
            }
        }
        int counted_synonyms = g_app.keyword_syn_pool_count - counted_keywords;
        printf("    └─ original keywords : %d\n", counted_keywords);
        printf("    └─ added from groups : %d\n", counted_synonyms);

        const int MAX_SHOWN = 10;
        printf("  Pool content (%d total):\n", g_app.keyword_syn_pool_count);
        for (int i = 0; i < g_app.keyword_syn_pool_count && i < MAX_SHOWN; i++)
            printf("    %s\n", g_app.keyword_syn_pool[i]);
        if (g_app.keyword_syn_pool_count > MAX_SHOWN)
            printf("    ... (and %d more)\n", g_app.keyword_syn_pool_count - MAX_SHOWN);
    } else {
        printf("    (empty - no keywords or synonym groups loaded)\n");
    }
    printf("\n");

    // ------------------------------------------------------------------------
    // Thematic Starters
    // ------------------------------------------------------------------------
    printf("--- Thematic Starters ---\n");
    printf("  thematic_starter_count : %d\n", g_app.thematic_starter_count);
    printf("  total_thematic_weight  : %ld\n", g_app.total_thematic_weight);

    if (g_app.thematic_starter_count > 0 && g_app.thematic_starters) {
        int counted_keywords = 0;
        for (int i = 0; i < g_app.thematic_starter_count; i++) {
            for (int k = 0; k < g_app.keyword_count; k++) {
                if (g_app.thematic_starters[i].word == g_app.keywords[k]) {
                    counted_keywords++;
                    break;
                }
            }
        }
        int counted_extra = g_app.thematic_starter_count - counted_keywords;
        printf("    └─ from original keywords : %d\n", counted_keywords);
        printf("    └─ from synonym groups    : %d\n", counted_extra);

        // Optional: show top N by count
        const int MAX_SHOWN = 10;
        printf("  Thematic starters (%d total):\n", g_app.thematic_starter_count);
        for (int i = 0; i < g_app.thematic_starter_count && i < MAX_SHOWN; i++)
            printf("    %s (count: %ld)\n",
                   g_app.thematic_starters[i].word,
                   g_app.thematic_starters[i].count);
        if (g_app.thematic_starter_count > MAX_SHOWN)
            printf("    ... (and %d more)\n", g_app.thematic_starter_count - MAX_SHOWN);
    } else {
        printf("    (none - no matching starters in synonym pool)\n");
    }
    printf("\n");

    // ------------------------------------------------------------------------
    // Model Tables / Stats
    // ------------------------------------------------------------------------
    printf("--- Model Stats ---\n");
    printf("  total_lines        : %ld\n", g_app.total_lines);
    printf("  hashtable         : %p\n", (void *)g_app.hashtable);
    printf("  hashtable_size    : %zu\n", g_app.hashtable_size);
    printf("  intern_table      : %p\n", (void *)g_app.intern_table);
    printf("  intern_table_size : %zu\n", g_app.intern_table_size);
    printf("  starter_table     : %p\n", (void *)g_app.starter_table);
    printf("  starter_table_size: %zu\n", g_app.starter_table_size);
    printf("  vocab_size        : %ld\n", g_app.vocab_size);
    printf("  total_starters    : %ld\n", g_app.total_starters);
    printf("  total_lines_processed : %ld\n", atomic_load(&g_app.total_lines_processed));
    printf("\n");

    // ------------------------------------------------------------------------
    // Memory Arena
    // ------------------------------------------------------------------------
    printf("--- Memory Arena ---\n");
    printf("  head                  : %p\n", (void *)g_app.arena_head);
    printf("  current               : %p\n", (void *)g_app.arena_current);
    printf("  block_size            : %zu\n", (size_t)ARENA_BLOCK_SIZE);
    printf("  arena_total_bytes_used : %zu\n", g_app.arena_total_bytes_used);
    printf("  arena_block_count      : %zu\n", g_app.arena_block_count);
    printf("\n");
    
	// --- Generation filters & controls ---
	printf("--- Generation Filters & Controls ---\n");
	printf("  rep_window            : %d\n", g_app.rep_window);
	printf("  rep_penalty           : %.2f    (multiplier <1 = penalize repeats)\n", g_app.rep_penalty);
	printf("  temperature           : %.2f    (0.0 = greedy, 1.0 = neutral, >1.0 = more random)\n", g_app.temperature);
	printf("  top_k                 : %d      (0 = disabled, higher = more diversity)\n", g_app.top_k);
	printf("  theme_force_probability : %.2f    (probability to strongly prefer thematic/synonym word)\n", g_app.theme_force_probability);
	printf("  Average theme density : %.3f\n",g_app.average_theme_density);
	printf("\n");

    // ------------------------------------------------------------------------
    // Candidate Generation
    // ------------------------------------------------------------------------
    printf("--- Candidates ---\n");
    printf("  Attempts made         : %d\n", g_app.generation_attempts);
    printf("  Valid candidates kept : %d\n", g_app.candidate_count);
    if (g_app.candidate_count > 0)
        printf("  Best score found      : %.4f\n", g_app.best_score);
    else
        printf("  (none generated - check corpus, keywords or parameters)\n");
    printf("\n");

    // --- Freelist / Allocation Stats ---
    printf("--- Freelist / Allocation Stats ---\n");
    printf("  entry_freelist        : %p\n", (void *)g_app.entry_freelist);
    printf("  next_freelist         : %p\n", (void *)g_app.next_freelist);
    printf("  entry_alloc_count      : %ld\n", g_app.entry_alloc_count);
    printf("  next_alloc_count       : %ld\n", g_app.next_alloc_count);
    printf("  entry_from_freelist    : %ld\n", g_app.entry_from_freelist);
    printf("  next_from_freelist     : %ld\n", g_app.next_from_freelist);
    printf("\n");

    printf("=== End of debug dump ===\n\n");
}

long get_lines_nb(const char *filename) {
    if (!filename || !*filename) {
        fprintf(stderr, "get_lines_nb: invalid filename\n");
        return 0;
    }

    FILE *f = fopen(filename, "r");
    if (!f) {
        perror("get_lines_nb: cannot open file");
        return 0;
    }

    long lines = 0;
    char buffer[8192]; // large buffer for efficient reading

    while (fgets(buffer, sizeof(buffer), f)) {
        lines++;
    }

    if (ferror(f)) {
        perror("get_lines_nb: error reading file");
        fclose(f);
        return 0;
    }

    fclose(f);
    return lines;
}

// ------------------------------------------------------------------------
// Freelist / Allocation Stats - Next nodes
// ------------------------------------------------------------------------
Next *alloc_next(void) {
    if (g_app.next_freelist) {
        Next *n = g_app.next_freelist;
        g_app.next_freelist = n->next;
        memset(n, 0, sizeof(Next));   // reset before reuse
        return n;
    }
    Next *n = calloc(1, sizeof(Next));
    if (!n) {
        fprintf(stderr, "alloc_next: calloc failed\n");
        exit(1);
    }
    return n;
}

void free_next(Next *n) {
    if (!n) return;
    n->next = g_app.next_freelist;
    g_app.next_freelist = n;
}

// ------------------------------------------------------------------------
// Freelist / Allocation Stats - Entry nodes
// ------------------------------------------------------------------------
Entry *alloc_entry(void) {
    if (g_app.entry_freelist) {
        Entry *e = g_app.entry_freelist;
        g_app.entry_freelist = e->next;
        memset(e, 0, sizeof(Entry));  // reset before reuse
        return e;
    }
    Entry *e = calloc(1, sizeof(Entry));
    if (!e) {
        fprintf(stderr, "alloc_entry: calloc failed\n");
        exit(1);
    }
    return e;
}

void free_entry(Entry *e) {
    if (!e) return;
    e->next = g_app.entry_freelist;
    g_app.entry_freelist = e;
}

// ------------------------------------------------------------------------
// Arena Allocator
// ------------------------------------------------------------------------
void arena_init(void) {
    if (g_app.arena_head) {
        // Already initialized → nothing to do
        return;
    }

    ArenaBlock *block = calloc(1, sizeof(ArenaBlock));
    if (!block) {
        fprintf(stderr, "arena_init: calloc failed\n");
        exit(1);
    }

    g_app.arena_head    = block;
    g_app.arena_current = block;
    block->used         = 0;
}

// ------------------------------------------------------------------------
// Arena Allocation - returns pointer to contiguous memory
// ------------------------------------------------------------------------
char *arena_alloc(size_t size) {
    if (!g_app.arena_current) {
        arena_init();  // ensure arena is ready
    }

    // Check if current block has enough space
    if (g_app.arena_current->used + size + 1 > ARENA_BLOCK_SIZE) {
        ArenaBlock *new_block = calloc(1, sizeof(ArenaBlock));
        if (!new_block) {
            fprintf(stderr, "arena_alloc: calloc failed for new block\n");
            exit(1);
        }
        g_app.arena_current->next = new_block;
        g_app.arena_current = new_block;
    }

    char *p = g_app.arena_current->data + g_app.arena_current->used;
    g_app.arena_current->used += size + 1;  // +1 for safety / null terminator if needed
    return p;
}

// ------------------------------------------------------------------------
// Select and Print the Best Candidate
// ------------------------------------------------------------------------
static void select_and_print_best(void) {
    if (g_app.candidate_count == 0) {
        fprintf(stderr, "\n[CRITICAL] No valid candidates generated.\n");
        return;
    }

    // --------------------------------------------------------------------
    // Archive all candidates
    // --------------------------------------------------------------------
    FILE *archive = fopen("archive.txt", "a");  // append mode
    if (archive) {
        for (int i = 0; i < g_app.candidate_count; i++) {
            Candidate *c = &g_app.candidates[i];
            print_words_properly(archive, c->words, c->length);
            fputs("\n---\n", archive); // separator between candidates
        }
        fclose(archive);
    } else {
        fprintf(stderr, "\n[WARNING] Could not open archive.txt for writing.\n");
    }

    int best_idx = -1;
    double best_score = 1e15;  // initialize with a large number

    // --------------------------------------------------------------------
    // Evaluate each candidate and select the one with the lowest score.
    // In case of tie, prefer the longer candidate (more words).
    // --------------------------------------------------------------------
    for (int i = 0; i < g_app.candidate_count; i++) {
        Candidate *c = &g_app.candidates[i];
        double score = score_candidate(c->words, c->length);

        if (score < best_score || 
            (score == best_score && c->length > (best_idx >= 0 ? g_app.candidates[best_idx].length : 0))) {
            best_score = score;
            best_idx = i;
        }
    }

    if (best_idx < 0) {
        fprintf(stderr, "\n[CRITICAL] No best candidate found after scoring.\n");
        return;
    }

    // --------------------------------------------------------------------
    // Output the winning candidate
    // --------------------------------------------------------------------
    Candidate *winner = &g_app.candidates[best_idx];
    g_app.best_score = best_score;
	
	putchar('\n');
    print_words_properly(stdout, winner->words, winner->length);
    putchar('\n');
}


// ------------------------------------------------------------------------
// Merge Local Hashtable Range into Global Hashtable (Threaded)
// ------------------------------------------------------------------------
void *merge_local_into_global_range(void *arg) {
    MergeRangeArgs *mra = (MergeRangeArgs *)arg;

    Entry **local  = mra->local;
    Entry **global = mra->global;

    // --- Temporary per-bucket hints for faster matching ---
    Entry **hints = calloc(mra->end_bucket - mra->start_bucket, sizeof(Entry *));
    if (!hints) hints = NULL;  // fallback: full scan will still work

    // --- Temporary mapping to track local -> global entry matches ---
    struct TempMapping {
        Entry *local_entry;
        Entry *global_match;
    };

    struct TempMapping *mappings = NULL;
    size_t map_count = 0;
    size_t map_cap   = 0;

    // --- First pass: insert new entries or locate existing ones ---
    for (size_t i = mra->start_bucket; i < mra->end_bucket; i++) {
        Entry *e = local[i];
        while (e) {
            Entry *next_e = e->next;

            uint64_t h       = hash_context(&e->key);
            size_t   bucket  = h % g_app.hashtable_size;
            Entry   *match   = NULL;

            // --- Fast hint check ---
            if (hints && hints[i - mra->start_bucket] &&
                context_equal(&hints[i - mra->start_bucket]->key, &e->key)) {
                match = hints[i - mra->start_bucket];
            }

            // --- Full scan fallback ---
            if (!match) {
                Entry *g = global[bucket];
                while (g) {
                    if (context_equal(&g->key, &e->key)) {
                        match = g;
                        if (hints) hints[i - mra->start_bucket] = g;
                        break;
                    }
                    g = g->next;
                }
            }

            // --- Track mapping for second pass ---
            if (map_count >= map_cap) {
                map_cap = map_cap ? map_cap * 2 : 65536;
                struct TempMapping *new_map = realloc(mappings, map_cap * sizeof(*mappings));
                if (!new_map) {
                    free(mappings);
                    free(hints);
                    fprintf(stderr, "[merge] realloc failed\n");
                    return NULL;
                }
                mappings = new_map;
            }
            mappings[map_count++] = (struct TempMapping){e, match};

            // --- Insert into global if not already present ---
            if (!match) {
                e->next = global[bucket];
                global[bucket] = e;
                if (hints) hints[i - mra->start_bucket] = e;
            }

            e = next_e;
        }
    }

    // --- Second pass: merge Next* lists for matching entries ---
    for (size_t i = 0; i < map_count; i++) {
        Entry *local_e  = mappings[i].local_entry;
        Entry *global_e = mappings[i].global_match;

        if (!global_e) continue;

        Next *ln = local_e->nexts;
        while (ln) {
            Next *next_ln = ln->next;
            Next **insert_ptr = &global_e->nexts;
            Next *gn = global_e->nexts;
            bool found = false;

            while (gn) {
                if (gn->word == ln->word) { // pointer equality for speed
                    gn->count += ln->count;
                    global_e->total_count += ln->count;
                    free_next(ln);
                    found = true;
                    break;
                }
                insert_ptr = &gn->next;
                gn = gn->next;
            }

            if (!found) {
                ln->next = *insert_ptr;
                *insert_ptr = ln;
                global_e->total_count += ln->count;
            }

            ln = next_ln;
        }

        // --- Free the local Entry struct if it was merged ---
        free_entry(local_e);
    }

    free(mappings);
    free(hints);
    return NULL;
}

// ------------------------------------------------------------------------
// Merge Local Starter Table into Global Starter Table
// ------------------------------------------------------------------------
void merge_starters(Starter **local, Starter **global) {
    // Counter for optional progress reporting
    long merged_buckets = 0;

    for (size_t i = 0; i < g_app.starter_table_size; i++) {
        Starter *s = local[i];
        while (s) {
            Starter *next_s = s->next;

            Starter *g = global[i];
            bool found = false;

            // --- Search for existing starter in the same bucket ---
            while (g) {
                if (strcmp(g->word, s->word) == 0) {
                    g->count += s->count;  // merge counts
                    found = true;
                    break;
                }
                g = g->next;
            }

            // --- Insert new starter if not found ---
            if (!found) {
                s->next = global[i];
                global[i] = s;
            }

            s = next_s;
        }

        merged_buckets++;

        // --- debug/progress reporting every 1000 buckets --- NO NEED
        // if (merged_buckets % 1000 == 0) {
        //     if (g_app.verbose >= 2) {
        //         printf("[merge_starters] Processed %ld buckets\n", merged_buckets);
        //     }
        // }
    }
}

// ------------------------------------------------------------------------
// Add a starter word to the local starter table
// ------------------------------------------------------------------------
void add_starter_local(Starter **table, long *total, const char *word) {
    // --- Safety checks ---
    if (!word || !*word) return;

    // Ignore single-character punctuation as starters
    if (strlen(word) == 1 && ispunct((unsigned char)word[0])) return;

    // --- Compute hash for bucket ---
    unsigned long h = fnv64_hash(word) % g_app.starter_table_size;

    // --- Scan bucket for existing starter ---
    Starter *s = table[h];
    while (s) {
        // Use pointer identity (interned strings) for speed
        if (s->word == word) {
            s->count++;
            (*total)++;
            return;
        }
        s = s->next;
    }

    // --- Insert new starter if not found ---
    Starter *new_s = calloc(1, sizeof(Starter));
    if (!new_s) {
        fprintf(stderr, "[add_starter_local] Memory allocation failed for Starter\n");
        exit(1);
    }

    new_s->word  = word;  // assume string is interned elsewhere
    new_s->count = 1;
    new_s->next  = table[h];
    table[h]     = new_s;
    (*total)++;
}


// ------------------------------------------------------------------------
// Thread-local Arena Allocation
// ------------------------------------------------------------------------

// --- Allocate memory from a thread-local arena ---
char *arena_alloc_thread(ArenaBlock **arena, size_t size) {
    if (!arena || !*arena) {
        fprintf(stderr, "[arena_alloc_thread] Invalid arena pointer\n");
        exit(1);
    }

    // --- Check if current block has enough space ---
    if ((*arena)->used + size + 1 > ARENA_BLOCK_SIZE) {
        ArenaBlock *new_block = calloc(1, sizeof(ArenaBlock));
        if (!new_block) {
            fprintf(stderr, "[arena_alloc_thread] Failed to allocate new ArenaBlock\n");
            exit(1);
        }
        (*arena)->next = new_block;
        *arena = new_block;
    }

    // --- Allocate memory from current block ---
    char *p = (*arena)->data + (*arena)->used;
    (*arena)->used += size + 1;
    return p;
}

// --- Allocate a Next node from a thread-local arena ---
Next *alloc_next_thread_arena(ArenaBlock **arena) {
    Next *n = (Next*)arena_alloc_thread(arena, sizeof(Next));
    memset(n, 0, sizeof(Next));
    return n;
}

// --- Allocate an Entry node from a thread-local arena ---
Entry *alloc_entry_thread_arena(ArenaBlock **arena) {
    Entry *e = (Entry*)arena_alloc_thread(arena, sizeof(Entry));
    memset(e, 0, sizeof(Entry));
    return e;
}


// ------------------------------------------------------------------------
// Thread-local: Add a word transition using only the thread-local arena
// ------------------------------------------------------------------------
void add_transition_local_thread(
    Entry **table,            // local hashtable
    const char *window[4],    // sliding context window (last 4 words)
    const char *next_word,    // word to add
    ArenaBlock **arena        // thread-local arena
) {
    if (!next_word || !table || !arena || !*arena) return;

    // --- Build 1–4 length contexts from the window ---
    ContextKey ctxs[4];
    int ctx_count = 0;
    for (int len = 1; len <= 4; len++) {
        if (!window[len - 1]) break;
        ContextKey *c = &ctxs[ctx_count++];
        c->order = len;
        for (int i = 0; i < len; i++) c->w[i] = window[i];
    }

    // --- Process each context key ---
    for (int i = 0; i < ctx_count; i++) {
        ContextKey *key = &ctxs[i];
        uint64_t h = hash_context(key);
        size_t idx = h & (g_app.hashtable_size - 1);

        Entry *e = table[idx];
        while (e) {
            if (context_equal(&e->key, key)) {
                // --- Update existing Next list ---
                Next *n = e->nexts;
                while (n) {
                    if (n->word == next_word) {
                        n->count++;
                        e->total_count++;
                        goto next_context;
                    }
                    n = n->next;
                }

                // --- Allocate new Next node from thread-local arena ---
                Next *new_n = alloc_next_thread_arena(arena);
                new_n->word  = next_word;
                new_n->count = 1;
                new_n->next  = e->nexts;
                e->nexts     = new_n;
                e->total_count++;
                goto next_context;
            }
            e = e->next;
        }

        // --- Entry not found: allocate new Entry from thread-local arena ---
        Entry *new_e = alloc_entry_thread_arena(arena);
        new_e->key         = *key;
        new_e->total_count = 1;

        // --- Allocate first Next node ---
        Next *new_n = alloc_next_thread_arena(arena);
        new_n->word  = next_word;
        new_n->count = 1;
        new_n->next  = NULL;

        new_e->nexts = new_n;
        new_e->next  = table[idx];
        table[idx]   = new_e;

    next_context:;
    }
}

// ------------------------------------------------------------------------
// Local: Add a word transition using the global allocator (freelist / malloc)
// ------------------------------------------------------------------------
void add_transition_local(
    Entry **table,           // global/local hashtable
    const char *window[4],   // sliding context window (last 4 words)
    const char *next_word    // word to add
) {
    if (!table || !next_word) return;

    // --- Build 1–4 length contexts from the sliding window ---
    ContextKey ctxs[4];
    int ctx_count = 0;
    for (int len = 1; len <= 4; len++) {
        if (!window[len - 1]) break;
        ContextKey *c = &ctxs[ctx_count++];
        c->order = len;
        for (int i = 0; i < len; i++) c->w[i] = window[i];
    }

    // --- Process each context key ---
    for (int i = 0; i < ctx_count; i++) {
        ContextKey *key = &ctxs[i];
        uint64_t h = hash_context(key);
        size_t idx = h & (g_app.hashtable_size - 1);

        Entry *e = table[idx];
        while (e) {
            if (context_equal(&e->key, key)) {
                // --- Update existing Next list ---
                Next *n = e->nexts;
                while (n) {
                    if (n->word == next_word) {
                        n->count++;
                        e->total_count++;
                        goto next_context;
                    }
                    n = n->next;
                }

                // --- Allocate a new Next node ---
                Next *new_n = alloc_next();
                new_n->word  = next_word;
                new_n->count = 1;
                new_n->next  = e->nexts;
                e->nexts     = new_n;
                e->total_count++;
                goto next_context;
            }
            e = e->next;
        }

        // --- Entry not found: allocate a new Entry ---
        Entry *new_e = alloc_entry();
        new_e->key         = *key;
        new_e->total_count = 1;

        // --- Allocate first Next node for this new Entry ---
        Next *new_n = alloc_next();
        new_n->word  = next_word;
        new_n->count = 1;
        new_n->next  = NULL;

        new_e->nexts = new_n;
        new_e->next  = table[idx];
        table[idx]   = new_e;

    next_context:;
    }
}

// ------------------------------------------------------------------------
// Thread: Process a file chunk and build local Markov model
// ------------------------------------------------------------------------
void *process_chunk(void *arg) {
    ThreadData *td = (ThreadData*)arg;

    FILE *f = fopen(td->filename, "r");
    if (!f) {
        fprintf(stderr, "[Thread %ld] Cannot open %s from byte %ld\n",
                (long)pthread_self(), td->filename, td->start_byte);
        return NULL;
    }

    if (fseek(f, td->start_byte, SEEK_SET) != 0) {
        fprintf(stderr, "[Thread %ld] Seek failed to %ld in %s\n",
                (long)pthread_self(), td->start_byte, td->filename);
        fclose(f);
        return NULL;
    }

    // --- Local buffers and sliding window ---
    char line[MAX_LINE_LEN];
    const char *words[256];
    const char *window[4] = {NULL};

    // --- Local seen array for vocabulary deduplication ---
    char *seen = calloc(g_app.intern_table_size, 1);
    if (!seen) {
        fprintf(stderr, "[Thread %ld] Failed to allocate seen array (%ld bytes)\n",
                (long)pthread_self(), g_app.intern_table_size);
        fclose(f);
        return NULL;
    }

    long current_pos = td->start_byte;
    long local_lines = 0;
    long last_reported = 0;  // per-thread last report

    // --- Read lines and build local structures ---
    while (fgets(line, sizeof(line), f)) {
        local_lines++;
        atomic_fetch_add(&g_app.total_lines_processed, 1);

        size_t line_len = strlen(line);
        current_pos += line_len + 1;

        // --- Safety: do not exceed chunk boundary ---
        if (current_pos > td->end_byte) break;

        // Remove trailing newline
        line[strcspn(line, "\n")] = '\0';

        // Split line into words
        int nw = split_into_words(line, words, 256);

        for (int j = 0; j < nw; j++) {
            const char *curr = words[j];

            // --- Track unique vocabulary words ---
            unsigned long vh = fnv64_hash(curr) % g_app.intern_table_size;
            if (!seen[vh]) {
                seen[vh] = 1;
                td->local_vocab++;
            }

            // --- Determine sentence start ---
            bool sent_start = (j == 0) ||
                              (window[0] && strchr(".!?", window[0][strlen(window[0])-1]));

            if (sent_start) {
                add_starter_local(td->local_starters, &td->local_total_starters, curr);
            }

            // --- Add Markov transition using thread-local arena ---
            add_transition_local_thread(td->local_hashtable, window, curr, &td->local_arena);

            // --- Slide the window ---
            window[3] = window[2];
            window[2] = window[1];
            window[1] = window[0];
            window[0] = curr;
        }

        // --- Progress report ---
        if (g_app.verbose >= 1) {
            long global_processed = atomic_load(&g_app.total_lines_processed);
            if (global_processed - last_reported >= 10000) {
                last_reported = global_processed;
                double percent = (double)global_processed / g_app.total_lines * 100.0;
                if (percent > 100.0) percent = 100.0;
                printf("\r\033[K%.1f%%", percent);
                fflush(stdout);
            }
        }
    }

    // --- Cleanup ---
    fclose(f);
    free(seen);

    return NULL;
}

// ------------------------------------------------------------------------
// Build Markov model (multi-threaded)
// ------------------------------------------------------------------------
int build_model_mt(void) {
    const char *filename = g_app.input_file;
    if (!filename || !*filename) {
        fprintf(stderr, "Error: No input file specified in app config\n");
        return -1;
    }

    // --- Count total lines ---
    g_app.total_lines = get_lines_nb(filename);
    if (g_app.total_lines == 0) {
        fprintf(stderr, "Input file is empty or unreadable\n");
        return -1;
    }

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    // --- Determine file size ---
    FILE *f = fopen(filename, "r");
    if (!f) {
        perror("Cannot open input file");
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    if (file_size <= 0) {
        fprintf(stderr, "File is empty or cannot get size\n");
        fclose(f);
        return -1;
    }
    fclose(f);

    // --- Allocate global hashtable ---
    if (!g_app.hashtable) {
        g_app.hashtable = calloc(g_app.hashtable_size, sizeof(Entry*));
        if (!g_app.hashtable) { perror("calloc failed for global hashtable"); return -1; }
    } else {
        memset(g_app.hashtable, 0, g_app.hashtable_size * sizeof(Entry*));
    }

    // --- Allocate starter table ---
    if (!g_app.starter_table) {
        g_app.starter_table = calloc(g_app.starter_table_size, sizeof(Starter*));
        if (!g_app.starter_table) { 
            free(g_app.hashtable); 
            g_app.hashtable = NULL; 
            perror("calloc failed for starter_table"); 
            return -1; 
        }
    } else {
        memset(g_app.starter_table, 0, g_app.starter_table_size * sizeof(Starter*));
    }

    g_app.vocab_size     = 0;
    g_app.total_starters = 0;

    // --- Initialize global arena ---
    g_app.arena_head    = NULL;
    g_app.arena_current = NULL;
    arena_init();

    // --------------------------------------------------------------------
    // Spawn parsing threads
    // --------------------------------------------------------------------
    pthread_t threads[NUM_THREADS];
    ThreadData td[NUM_THREADS];
    long bytes_per_thread = file_size / NUM_THREADS;

    for (int t = 0; t < NUM_THREADS; t++) {
        td[t].filename            = filename;
        td[t].start_byte          = t * bytes_per_thread;
        td[t].end_byte            = (t == NUM_THREADS - 1) ? file_size : (t + 1) * bytes_per_thread;
        td[t].local_hashtable     = calloc(g_app.hashtable_size, sizeof(Entry*));
        td[t].local_starters      = calloc(g_app.starter_table_size, sizeof(Starter*));
        td[t].local_total_starters = 0;
        td[t].local_vocab         = 0;
        td[t].local_arena         = calloc(1, sizeof(ArenaBlock));

        if (!td[t].local_hashtable || !td[t].local_starters || !td[t].local_arena) {
            fprintf(stderr, "Thread %d allocation failed\n", t);
            for (int i = 0; i <= t; i++) {
                free(td[i].local_hashtable);
                free(td[i].local_starters);
                free(td[i].local_arena);
            }
            free(g_app.hashtable);
            free(g_app.starter_table);
            g_app.hashtable = NULL;
            g_app.starter_table = NULL;
            return -1;
        }

        pthread_create(&threads[t], NULL, process_chunk, &td[t]);
    }

    for (int t = 0; t < NUM_THREADS; t++) pthread_join(threads[t], NULL);

    long final_processed = atomic_load(&g_app.total_lines_processed);
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double read_parse_time = (ts_end.tv_sec - ts_start.tv_sec) + 
                             (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;

    if (g_app.verbose >= 1) {
        printf("\n\nReading & parsing finished:\n");
        printf("  - Total lines processed: %ld\n", final_processed);
        printf("  - Time: %.2f seconds\n\n", read_parse_time);
        fflush(stdout);
    }

    // --------------------------------------------------------------------
    // Multi-threaded merge of hashtables
    // --------------------------------------------------------------------
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    int num_merge_threads = NUM_THREADS;
    pthread_t mthreads[num_merge_threads];

    typedef struct {
        Entry **local;
        Entry **global;
        size_t start_bucket;
        size_t end_bucket;
    } MergeRangeArgs;

    MergeRangeArgs args[num_merge_threads];
    size_t buckets_per_thread = g_app.hashtable_size / num_merge_threads;

    for (int i = 0; i < num_merge_threads; i++) {
        args[i].local        = td[i].local_hashtable;
        args[i].global       = g_app.hashtable;
        args[i].start_bucket = i * buckets_per_thread;
        args[i].end_bucket   = (i == num_merge_threads - 1) ? g_app.hashtable_size : (i + 1) * buckets_per_thread;

        pthread_create(&mthreads[i], NULL, merge_local_into_global_range, &args[i]);
    }

    for (int i = 0; i < num_merge_threads; i++) pthread_join(mthreads[i], NULL);

    // --------------------------------------------------------------------
    // Merge starters and thread arenas sequentially
    // --------------------------------------------------------------------
    for (int t = 0; t < NUM_THREADS; t++) {
        merge_starters(td[t].local_starters, g_app.starter_table);
        g_app.total_starters += td[t].local_total_starters;
        g_app.vocab_size     += td[t].local_vocab;

        // Merge arenas into global arena chain
        ArenaBlock *b = td[t].local_arena;
        while (b) {
            ArenaBlock *next_b = b->next;
            b->next = NULL;
            if (!g_app.arena_head) {
                g_app.arena_head = b;
                g_app.arena_current = b;
            } else {
                g_app.arena_current->next = b;
                g_app.arena_current = b;
            }
            b = next_b;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double merge_time = (ts_end.tv_sec - ts_start.tv_sec) + 
                        (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;

    double total_time = read_parse_time + merge_time;

    if (g_app.verbose >= 1) {
        printf("\nMerging finished:\n");
        printf("  - Merging time: %.2f seconds\n", merge_time);
        printf("\nOverall build complete:\n");
        printf("  - Total time (read+parse+merge): %.2f seconds\n", total_time);
        printf("  - Vocab size: %ld\n", g_app.vocab_size);
        printf("  - Total starters: %ld\n", g_app.total_starters);
        fflush(stdout);
    }

    return 0;
}

// ------------------------------------------------------------------------
// Hashing Utilities - Production-Grade
// ------------------------------------------------------------------------

// --- xxHash64 (very fast, non-cryptographic) --- TODO USE THIS ONE FASTER NEXT REFACT
static inline uint64_t xxhash64(const char *str) {
    // Simple, lightweight 64-bit hash for short strings
    uint64_t h = 0x517cc1b727220a95ULL;

    while (*str) {
        h ^= (uint64_t)(unsigned char)*str++;
        h *= 0x5bd1e995ULL;
        h ^= h >> 24;
    }

    return h;
}

// --- FNV-1a + finalizer hash (64-bit) ---
static inline uint64_t fnv64_hash(const char *str) {
    if (!str || !*str) return 0;

    uint64_t h = 0xcbf29ce484222325ULL; // FNV-1a offset basis (64-bit)

    while (*str) {
        h ^= (unsigned char)*str++;
        h *= 0x100000001b3ULL;            // FNV-1a prime
    }

    // Optional finalization (improves avalanche / distribution)
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;

    return h;
}

// --- Context key hash (pointer-based, fast for Markov chains) ---
static inline uint64_t hash_context(const ContextKey *ctx) {
    if (!ctx) return 0;

    uint64_t h = 0xcbf29ce484222325ULL;  // FNV-1a offset basis
    for (int i = 0; i < ctx->order; i++) {
        uintptr_t p = (uintptr_t) ctx->w[i]; // hash pointer value
        h ^= p;
        h *= 0x100000001b3ULL;
    }

    return h;
}


// ------------------------------------------------------------------------
// Context / String Utilities
// ------------------------------------------------------------------------

// --- Compare two ContextKey structs for equality ---
static inline bool context_equal(const ContextKey *a, const ContextKey *b) {
    if (!a || !b) return false;        // safety check
    if (a->order != b->order) return false;

    for (int i = 0; i < a->order; i++) {
        if (a->w[i] != b->w[i]) return false;
    }

    return true;
}

// --- Intern a string (returns persistent pointer) ---
static inline const char *intern(const char *s) {
    if (!s || !*s) return "";

    size_t len = strlen(s);
    uint64_t h = fnv64_hash(s) % g_app.intern_table_size;

    pthread_mutex_lock(&intern_mutex);

    InternEntry *e = g_app.intern_table[h];
    while (e) {
        if (strcmp(e->str, s) == 0) {
            pthread_mutex_unlock(&intern_mutex);
            return e->str;
        }
        e = e->next;
    }

    char *copy = arena_alloc(len + 1);
    memcpy(copy, s, len);
    copy[len] = '\0';

    InternEntry *new_e = malloc(sizeof(InternEntry));
    if (!new_e) { /* handle error */ }
    new_e->str  = copy;
    new_e->next = g_app.intern_table[h];
    g_app.intern_table[h] = new_e;

    pthread_mutex_unlock(&intern_mutex);
    return copy;
}

// ------------------------------------------------------------------------
// Load synonyms file into keyword pool
// ------------------------------------------------------------------------
int load_synonyms(void) {
    const char *filename = g_app.syn_file ? g_app.syn_file : "synonyms.txt";

    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Cannot open synonyms file: %s\n", filename);
        return -1;
    }

    // --- Initialize keyword pool ---
    g_app.keyword_syn_pool_count = 0;

    // Add original keywords to pool
    for (int i = 0; i < g_app.keyword_count; i++) {
        const char *kw = g_app.keywords[i];
        if (!kw) continue;

        bool dup = false;
        for (int j = 0; j < g_app.keyword_syn_pool_count; j++) {
            if (g_app.keyword_syn_pool[j] == kw) {
                dup = true;
                break;
            }
        }

        if (!dup && g_app.keyword_syn_pool_count < 2048) {
            g_app.keyword_syn_pool[g_app.keyword_syn_pool_count++] = kw;
        }
    }

    // --- Parse synonym file ---
    char line[8192];
    int groups_used = 0;
    int synonyms_added = 0;

    while (fgets(line, sizeof(line), f)) {
        line[strcspn(line, "\n")] = '\0';
        if (strlen(line) < 3) continue;

        char *words[128];
        int word_count = 0;

        char *token = strtok(line, ",");
        while (token && word_count < 128) {
            // Trim leading spaces
            while (isspace((unsigned char)*token)) token++;
            // Trim trailing spaces
            char *end = token + strlen(token) - 1;
            while (end >= token && isspace((unsigned char)*end)) *end-- = '\0';

            if (*token) words[word_count++] = token;
            token = strtok(NULL, ",");
        }

        if (word_count == 0) continue;

        // --- Check if group contains any relevant keywords ---
        bool relevant = false;
        for (int w = 0; w < word_count && !relevant; w++) {
            const char *candidate = intern(words[w]);
            for (int k = 0; k < g_app.keyword_count; k++) {
                if (g_app.keywords[k] == candidate) {
                    relevant = true;
                    break;
                }
            }
        }
        if (!relevant) continue;

        groups_used++;

        // --- Add words to global pool ---
        for (int w = 0; w < word_count; w++) {
            const char *syn_interned = intern(words[w]);

            bool dup = false;
            for (int j = 0; j < g_app.keyword_syn_pool_count; j++) {
                if (g_app.keyword_syn_pool[j] == syn_interned) {
                    dup = true;
                    break;
                }
            }

            if (!dup && g_app.keyword_syn_pool_count < 2048) {
                g_app.keyword_syn_pool[g_app.keyword_syn_pool_count++] = syn_interned;
                synonyms_added++;
            }
        }
    }

    fclose(f);

    return (g_app.keyword_syn_pool_count > 0) ? 0 : -2;
}

// ------------------------------------------------------------------------
// Check if a token is a single punctuation mark
// ------------------------------------------------------------------------
static inline bool is_punct(const char *tok) {
    return tok && strlen(tok) == 1 && strchr(".,!?;:", tok[0]);
}

// ------------------------------------------------------------------------
// Split a line into words and punctuation tokens
// Returns the number of tokens placed in tokens_out (up to max_tokens)
// All tokens are interned and lowercased
// ------------------------------------------------------------------------
int split_into_words(char *line, const char **tokens_out, int max_tokens) {
    if (!line || !tokens_out || max_tokens <= 0) return 0;

    int count = 0;
    char *p = line;

    while (*p && count < max_tokens) {
        // Skip leading whitespace
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;

        // Single punctuation as token
        if (strchr(".,!?;:", *p)) {
            char tmp[2] = {*p, '\0'};
            tokens_out[count++] = intern(tmp);
            p++;
            continue;
        }

        // Regular word
        char *start = p;
        int len = 0;

        while (*p && !isspace((unsigned char)*p) && !strchr(".,!?;:", *p)) {
            len++;
            p++;
        }

        // Trim punctuation from start and end, but keep apostrophes
        while (len > 0 && ispunct((unsigned char)*start) && *start != '\'') { start++; len--; }
        while (len > 0 && ispunct((unsigned char)*(start + len - 1)) && *(start + len - 1) != '\'') len--;

        if (len <= 0) continue;
        if (len >= WORD_MAX_LEN) len = WORD_MAX_LEN - 1;

        char tmp[WORD_MAX_LEN];
        strncpy(tmp, start, len);
        tmp[len] = '\0';

        // Lowercase
        for (int i = 0; i < len; i++)
            tmp[i] = tolower((unsigned char)tmp[i]);

        tokens_out[count++] = intern(tmp);
    }

    return count;
}

// ------------------------------------------------------------------------
// Compute perplexity of a sequence using interpolated n-grams (log-domain)
// Efficient Lidstone smoothing and backoff up to 4-grams
// ------------------------------------------------------------------------
static double compute_perplexity(const char **words, int nw) {
    if (!words || nw < 2) return 1e12;
    if (g_app.vocab_size <= 0) return 1e12;

    double log_sum = 0.0;
    int valid_count = 0;

    // --------------------------------------------------------------------
    // Interpolation weights: order 0 → 4
    // Sum should ≈1; can be tuned for performance / accuracy
    // --------------------------------------------------------------------
    static const double lambdas[5] = {0.0, 0.15, 0.25, 0.35, 0.25};

    // --------------------------------------------------------------------
    // Lidstone smoothing constant
    // --------------------------------------------------------------------
    const double alpha = 0.01;

    // --------------------------------------------------------------------
    // Iterate over all positions in sequence
    // --------------------------------------------------------------------
    for (int pos = 1; pos < nw; pos++) {
        int ctx_len = (pos < 4) ? pos : 4;  // max 4-gram
        double prob = 0.0;
        double remaining = 1.0;

        // ----------------------------------------------------------------
        // Backoff loop: high-order → low-order
        // ----------------------------------------------------------------
        for (int ord = ctx_len; ord >= 1; ord--) {
            if (remaining < 1e-9) break;

            // Build context key
            ContextKey ctx = {0};
            ctx.order = (uint8_t)ord;
            for (int k = 0; k < ord; k++) ctx.w[k] = words[pos - 1 - k];

            // Lookup hash bucket
            uint64_t h = hash_context(&ctx);
            size_t bucket = h & (g_app.hashtable_size - 1);

            long observed = 0;
            long match_count = 0;

            // Search entries
            for (Entry *e = g_app.hashtable[bucket]; e; e = e->next) {
                if (context_equal(&e->key, &ctx)) {
                    observed = e->total_count;
                    for (Next *n = e->nexts; n; n = n->next) {
                        if (n->word == words[pos]) {
                            match_count = n->count;
                            break;
                        }
                    }
                    break;
                }
            }

            // Lidstone smoothing probability
            double denom = observed + alpha * g_app.vocab_size;
            if (denom <= 0) denom = 1.0;
            double p = (observed > 0) ? (match_count + alpha) / denom
                                      : alpha / denom;


            // Weighted interpolation
            double w = lambdas[ord] * remaining;
            prob += w * p;
            remaining -= w;
        }

        // ----------------------------------------------------------------
        // Remaining mass: fallback uniform probability
        // Slightly boosted to avoid zero-prob sequences
        // ----------------------------------------------------------------
        double uniform_p = 1.0 / (g_app.vocab_size + 10000.0);
        prob += remaining * uniform_p;

        // Accumulate log-probability
        if (prob > 0.0) {
            log_sum += log(prob);
            valid_count++;
        }
    }

	if (valid_count == 0) return 1e12;

    double avg_log_p = log_sum / (double)valid_count;
    double perplexity = exp(-avg_log_p);

    // Prevent inf / NaN leaking into scoring
    if (!isfinite(perplexity) || perplexity < 1e-9) return 1e12;

    return perplexity;
}

// ------------------------------------------------------------------------
// Check if a word is related to keywords (via pointer equality in keyword pool)
// ------------------------------------------------------------------------
static inline bool is_keyword_related(const char *w) {
    if (!w) return false;

    // Linear search over pointer-interned keyword pool
    for (int i = 0; i < g_app.keyword_syn_pool_count; i++) {
        if (g_app.keyword_syn_pool[i] == w) return true;
    }

    return false;
}

// ------------------------------------------------------------------------
// Select a starter word, preferring thematic starters if available
// Weighted random selection
// ------------------------------------------------------------------------
static const char *select_thematic_starter(void) {
    // --- Prefer thematic starters ---
    if (g_app.thematic_starter_count > 0 && g_app.total_thematic_weight > 0) {
        long r = rand() % g_app.total_thematic_weight;
        long sum = 0;

        for (int i = 0; i < g_app.thematic_starter_count; i++) {
            sum += g_app.thematic_starters[i].count;
            if (r < sum) {
                return g_app.thematic_starters[i].word;
            }
        }
    }

    // --- Fallback: full starter table weighted selection ---
    if (g_app.total_starters > 0) {
        long r = rand() % g_app.total_starters;
        long sum = 0;

        for (size_t i = 0; i < g_app.starter_table_size; i++) {
            for (Starter *s = g_app.starter_table[i]; s; s = s->next) {
                sum += s->count;
                if (r < sum) return s->word;
            }
        }
    }

    // --- Ultimate fallback ---
    return intern("hello");
}

int generate_sequence(const char **out_words)
{
    int max_tokens = MAX_WORDS_PER_GENERATION;
    int generated = 0;

    const char *prev[4] = {NULL};

    // ----------------------------
    // Starter selection
    // ----------------------------
    prev[0] = select_thematic_starter();

    // Push initial context (up to 4)
    for (int k = 3; k >= 0; k--) {
        if (prev[k] && generated < max_tokens) {
            out_words[generated++] = prev[k];
        }
    }

    // ----------------------------
    // Main generation loop
    // ----------------------------
    while (generated < max_tokens) {
        const char *candidates[4] = {NULL}; // order 4 -> slot 0
        int best_syn_slot = -1;
        bool has_syn_candidate = false;

        // --- Collect candidates from 4→1 order ---
        for (int order = 4; order >= 1; order--) {
            if (generated < order) continue;

            ContextKey ctx = {0};
            ctx.order = (uint8_t)order;
            for (int k = 0; k < order; k++)
                ctx.w[k] = out_words[generated - 1 - k];

            uint64_t h = hash_context(&ctx);
            size_t idx = h & (g_app.hashtable_size - 1);

            for (Entry *e = g_app.hashtable[idx]; e; e = e->next) {
                if (!context_equal(&e->key, &ctx)) continue;

                const char *best_next = NULL;
                long max_count = -1;
                for (Next *n = e->nexts; n; n = n->next) {
                    if (n->count > max_count) {
                        max_count = n->count;
                        best_next = n->word;
                    }
                }

                if (best_next) {
                    int slot = 4 - order;
                    candidates[slot] = best_next;

                    if (is_keyword_related(best_next)) {
                        has_syn_candidate = true;
                        if (best_syn_slot == -1 || slot < best_syn_slot)
                            best_syn_slot = slot;
                    }
                }
                break; // stop after first matching context
            }
        }

        // ----------------------------
        // Decide next word
        // ----------------------------
        const char *next = NULL;

        // --- Priority 1: synonym bias (70% chance) ---
        if (has_syn_candidate && best_syn_slot != -1 && (rand() % 100 < 70)) {
            next = candidates[best_syn_slot];
        }

        // --- Priority 2: temperature + top-k sampling (replaces old weighted random) ---
		if (!next) {
			// Collect valid candidates with their base weights
			typedef struct {
				const char *word;
				double     weight;
				int        slot;
			} ScoredCand;

			ScoredCand scored[4];
			int num_valid = 0;

			static const double base_weights[4] = {0.9, 0.7, 0.4, 0.2};

			for (int slot = 0; slot < 4; slot++) {
				if (!candidates[slot]) continue;

				double w = base_weights[slot];

				// --- repetition penalty ---
				for (int k = 1; k <= g_app.rep_window && generated - k >= 0; k++) {
				    if (out_words[generated - k] == candidates[slot]) {
				        w *= g_app.rep_penalty;
				        break;
				    }
				}
				
				// --- extra penalty on common connector bigrams ---
				if (generated >= 1) {
					const char *prev = out_words[generated - 1];
					const char *common_connectors[] = {"the", "and", "of", "to", "in", "a", NULL};
					
					for (int c = 0; common_connectors[c]; c++) {
						if (strcmp(prev, common_connectors[c]) == 0) {
							// If previous word is "the/and/of/...", penalize repeating the pattern
							w *= (g_app.rep_penalty * 0.75);  // 25% extra reduction
							break;
						}
					}
				}

				scored[num_valid].word   = candidates[slot];
				scored[num_valid].weight = w;
				scored[num_valid].slot   = slot;
				num_valid++;
			}

			if (num_valid == 0) {
				// nothing to do — will fall through to fallback
			} else {
				// 1. Apply temperature (soften or sharpen distribution)
				double temp_sum = 0.0;
				double temp_weights[4];

				for (int i = 0; i < num_valid; i++) {
				    double p = scored[i].weight;
				    // Avoid log(0) or pow(0, ...)
				    if (p <= 0) p = 1e-10;
				    temp_weights[i] = pow(p, 1.0 / g_app.temperature);
				    temp_sum += temp_weights[i];
				}

				if (temp_sum > 0) {
				    for (int i = 0; i < num_valid; i++) {
				        temp_weights[i] /= temp_sum;
				    }
				} else {
				    // fallback: uniform
				    for (int i = 0; i < num_valid; i++) {
				        temp_weights[i] = 1.0 / num_valid;
				    }
				}

				// 2. Apply top-k (if enabled and makes sense)
				int sample_from = num_valid;
				if (g_app.top_k > 0 && g_app.top_k < num_valid) {
				    sample_from = g_app.top_k;

				    // Simple sort by temp_weight descending (4 elements → bubble sort is fine)
				    for (int i = 0; i < sample_from; i++) {
				        for (int j = i + 1; j < num_valid; j++) {
				            if (temp_weights[j] > temp_weights[i]) {
				                // swap
				                double tw = temp_weights[i]; temp_weights[i] = temp_weights[j]; temp_weights[j] = tw;
				                ScoredCand tmp = scored[i]; scored[i] = scored[j]; scored[j] = tmp;
				            }
				        }
				    }
				}

				// 3. Sample from (possibly top-k) distribution
				double r = (double)rand() / RAND_MAX;
				double cum = 0.0;
				next = scored[0].word;  // fallback = first one

				for (int i = 0; i < sample_from; i++) {
				    cum += temp_weights[i];
				    if (r <= cum) {
				        next = scored[i].word;
				        break;
				    }
				}
			}
		}

        // --- Priority 3: fallback to starter table ---
        if (!next) {
            if (g_app.total_starters > 0) {
                long r = rand() % g_app.total_starters;
                long sum = 0;
                for (size_t i = 0; i < g_app.starter_table_size; i++) {
                    for (Starter *s = g_app.starter_table[i]; s; s = s->next) {
                        sum += s->count;
                        if (r < sum) {
                            next = s->word;
                            goto starter_chosen;
                        }
                    }
                }
            }
            next = intern("hello");
        }

    starter_chosen:
        if (!next) next = intern(".");
        out_words[generated++] = next;

        // --- Early stopping ---
        if (generated >= g_app.max_gen_len) break;
        if (generated >= g_app.min_gen_len &&
            (strcmp(next, ".") == 0 || strcmp(next, "!") == 0 || strcmp(next, "?") == 0))
        {
            break;
        }
    }

    // ----------------------------
    // Post-processing
    // ----------------------------
    if (generated == 0) return 0;

    // --- remove leading punctuation except "." ---
    int start = 0;
    while (start < generated &&
           ispunct((unsigned char)out_words[start][0]) &&
           strcmp(out_words[start], ".") != 0)
    {
        start++;
    }
    int len = generated - start;

    if (len > g_app.max_gen_len) len = g_app.max_gen_len;
    if (len < g_app.min_gen_len) len = g_app.min_gen_len;

    // --- trim to last "." ---
    for (int i = len - 1; i >= 0; i--) {
        if (strcmp(out_words[start + i], ".") == 0) {
            len = i + 1;
            break;
        }
    }

    // --- shift output ---
    for (int i = 0; i < len; i++)
        out_words[i] = out_words[start + i];
    generated = len;

    // --- capitalize after '.' ---
    bool cap = true;
    for (int i = 0; i < generated; i++) {
        const char *w = out_words[i];
        if (!w) continue;
        if (cap && isalpha((unsigned char)w[0])) {
            char buf[WORD_MAX_LEN];
            strncpy(buf, w, sizeof(buf)-1);
            buf[sizeof(buf)-1] = '\0';
            buf[0] = toupper((unsigned char)buf[0]);
            out_words[i] = intern(buf);
            cap = false;
        }
        if (strcmp(w, ".") == 0) cap = true;
    }

    // --- force final dot ---
    if (generated == 0 || strcmp(out_words[generated-1], ".") != 0)
        out_words[generated++] = intern(".");

    return generated;
}

static void print_words_properly(FILE *out, const char **words, int count) {
    for (int i = 0; i < count; i++) {
        const char *w = words[i];
        if (!w) continue;

        // --- First word: always capitalize if alphabetical ---
        if (i == 0) {
            if (isalpha((unsigned char)w[0])) {
                fputc(toupper((unsigned char)w[0]), out);
                fputs(w + 1, out);
            } else {
                fputs(w, out);
            }
            continue;
        }

        const char *prev = words[i - 1];

        // --- Determine if space is needed ---
        bool space_before = true;

        // --- No space after opening punctuation ---
        if (prev && strchr("([{‘“\"'", prev[0])) {
            space_before = false;
        }

        // --- No space before closing or sentence-ending punctuation ---
        if (strchr(".,!?;:)]’”\"'", w[0])) {
            space_before = false;
        }

        if (space_before) fputc(' ', out);
        fputs(w, out);
    }
}

void free_model(void)
{
    printf("[free_model] Starting cleanup...\n");

    // 1. Candidates (safest first – small arrays)
    printf("[free_model] Freeing candidates (%d)...\n", g_app.candidate_count);
    for (int i = 0; i < g_app.candidate_count; i++) {
        if (g_app.candidates[i].words) {
            free((void *)g_app.candidates[i].words);
            g_app.candidates[i].words = NULL;
        }
    }
    g_app.candidate_count = 0;
    printf("[free_model] Candidates freed.\n");

    // 2. Keyword arrays
    printf("[free_model] Freeing keywords & thematic starters...\n");
    if (g_app.keywords) {
        free(g_app.keywords);
        g_app.keywords = NULL;
    }
    if (g_app.thematic_starters) {
        free(g_app.thematic_starters);
        g_app.thematic_starters = NULL;
    }
    printf("[free_model] Keywords & thematic freed.\n");

    // 3. Starter table – these were malloc/calloc'ed
    if (g_app.starter_table) {
        printf("[free_model] Freeing starter_table (%zu buckets)...\n", g_app.starter_table_size);
        for (size_t i = 0; i < g_app.starter_table_size; i++) {
            Starter *s = g_app.starter_table[i];
            while (s) {
                Starter *next_s = s->next;
                free(s);
                s = next_s;
            }
        }
        free(g_app.starter_table);
        g_app.starter_table = NULL;
        printf("[free_model] Starter table freed.\n");
    }

    // 4. Intern table – InternEntry nodes were malloc'ed
    if (g_app.intern_table) {
        printf("[free_model] Freeing intern_table (%zu buckets)...\n", g_app.intern_table_size);
        for (size_t i = 0; i < g_app.intern_table_size; i++) {
            InternEntry *e = g_app.intern_table[i];
            while (e) {
                InternEntry *next_e = e->next;
                free(e);
                e = next_e;
            }
        }
        free(g_app.intern_table);
        g_app.intern_table = NULL;
        printf("[free_model] Intern table freed.\n");
    }

    // 5. IMPORTANT: Arena blocks contain strings + many Entry/Next nodes
    //    We free whole blocks → do NOT free individual Entry/Next anymore!
    if (g_app.arena_head) {
        printf("[free_model] Freeing arena (%zu bytes used, %zu blocks)...\n",
               g_app.arena_total_bytes_used, g_app.arena_block_count);

        ArenaBlock *block = g_app.arena_head;
        size_t block_count = 0;
        while (block) {
            ArenaBlock *next = block->next;
            free(block);
            block = next;
            block_count++;
            if (block_count % 10 == 0) {
                printf("[free_model] Freed %zu arena blocks...\n", block_count);
            }
        }
        g_app.arena_head = NULL;
        g_app.arena_current = NULL;
        printf("[free_model] Arena fully freed (%zu blocks).\n", block_count);
    }

    // 6. hashtable array itself (but NOT the Entries – they live in arena!)
    if (g_app.hashtable) {
        printf("[free_model] Freeing hashtable array pointer only...\n");
        free(g_app.hashtable);
        g_app.hashtable = NULL;
        printf("[free_model] Hashtable array freed.\n");
    }

    // 7. Reset stats & freelists
    printf("[free_model] Resetting stats & freelists...\n");
    g_app.vocab_size     = 0;
    g_app.total_starters = 0;
    g_app.entry_freelist = NULL;
    g_app.next_freelist  = NULL;
    g_app.best_candidate_index = -1;
    g_app.best_score = 0.0;
    g_app.generation_attempts = 0;

    printf("[free_model] Cleanup finished successfully.\n");
}


static int parse_arguments(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <input_file> [keywords=kw1,kw2,...] [min_length=N] [max_length=N]\n",
                argv[0]);
        return 1;
    }

    g_app.input_file = argv[1];

    // Allocate keyword array if not already
    if (!g_app.keywords) {
        g_app.keywords = calloc(MAX_KEYWORDS, sizeof(char *));
        if (!g_app.keywords) {
            perror("malloc for keywords failed");
            return 1;
        }
    }
    g_app.keyword_count = 0;

    // Parse optional arguments
    for (int i = 2; i < argc; i++) {
        if (strncmp(argv[i], "keywords=", 9) == 0) {
            char tmp[MAX_LINE_LEN];
            snprintf(tmp, sizeof(tmp), "%s", argv[i] + 9);

            char *tok = strtok(tmp, ",");
            while (tok) {
                while (isspace((unsigned char)*tok)) tok++;
                char *end = tok + strlen(tok) - 1;
                while (end >= tok && isspace((unsigned char)*end)) *end-- = '\0';

                if (*tok) {
                    if (g_app.keyword_count >= MAX_KEYWORDS) {
                        fprintf(stderr, "Too many keywords (max %d)\n", MAX_KEYWORDS);
                        return 1;
                    }
                    g_app.keywords[g_app.keyword_count++] = intern(tok);
                }
                tok = strtok(NULL, ",");
            }

        } else if (strncmp(argv[i], "min_length=", 11) == 0) {
            char *end;
            long v = strtol(argv[i] + 11, &end, 10);
            if (*end || v <= 0) {
                fprintf(stderr, "Invalid min_length: %s\n", argv[i]);
                return 1;
            }
            g_app.min_gen_len = (int)v;

        } else if (strncmp(argv[i], "max_length=", 11) == 0) {
            char *end;
            long v = strtol(argv[i] + 11, &end, 10);
            if (*end || v <= 0) {
                fprintf(stderr, "Invalid max_length: %s\n", argv[i]);
                return 1;
            }
            g_app.max_gen_len = (int)v;

        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return 1;
        }
    }

    // Validate length constraints
    if (g_app.min_gen_len > g_app.max_gen_len) {
        fprintf(stderr, "Error: min_length > max_length\n");
        return 1;
    }

    return 0;
}


// ----------------------------
// main
// ----------------------------
int main(int argc, char **argv)
{
    if (app_init() != 0) {
        fprintf(stderr, "Fatal: initialization failed\n");
        return 1;
    }

    // ----------------------------
    // Parse arguments
    // ----------------------------
    if (parse_arguments(argc, argv) != 0)
        goto shutdown;

    // ----------------------------
    // Load synonyms (optional)
    // ----------------------------
    if (load_synonyms() < 0)
        fprintf(stderr, "Warning: synonyms disabled\n");

    // ----------------------------
    // Build model (multi-threaded)
    // ----------------------------
    if (build_model_mt() != 0) {
        fprintf(stderr, "Fatal: model build failed\n");
        goto shutdown;
    }

    if (g_app.vocab_size == 0) {
        fprintf(stderr,
                "CRITICAL: vocab_size is 0 → will cause div-by-zero in get_smoothed_prob\n");
    }

    if (g_app.total_starters == 0) {
        fprintf(stderr,
                "WARNING: no starter words → generation will fallback to 'hello'\n");
    }

    // ----------------------------
    // Prepare thematic starters
    // ----------------------------
    precompute_thematic_starters();

    // ----------------------------
    // Generate candidates
    // ----------------------------
    generate_multiple_candidates();

    // ----------------------------
    // Select and print the best candidate
    // ----------------------------
    select_and_print_best();

    // ----------------------------
    // Optional debug output
    // ----------------------------
    if (g_app.verbose >= 1) {
        display_global_debug();
        //display_candidates();  // Not needed currently
    }

shutdown:
    // --- Cleanup ---
    free_model();
    return 0;
}

static double score_candidate(const char **words, int nw)
{
    if (nw < g_app.min_gen_len) return 1e15;

    // 1. Theme density
    int theme_count = 0, exact_keyword_count = 0;
    for (int i = 0; i < nw; i++) {
        if (is_keyword_related(words[i])) {
            theme_count++;
            for (int k = 0; k < g_app.keyword_count; k++) {
                if (words[i] == g_app.keywords[k]) {
                    exact_keyword_count++;
                    break;
                }
            }
        }
    }
    double theme_ratio = (double)theme_count / nw;
    double theme_score = 8.0 * (1.0 - theme_ratio);  // reduced from 30

    double exact_bonus = exact_keyword_count * 0.5;

    // 2. Repetition
    int local_rep = 0;
    for (int j = 2; j < nw; j++) {
        if (words[j] == words[j-2] && words[j-1] == words[j+1]) local_rep++;
    }
    int global_rep = 0;
    for (int i = 0; i < nw - 2; i++) {
        for (int j = i + 3; j < nw - 2; j++) {
            if (words[i] == words[j] && words[i+1] == words[j+1] && words[i+2] == words[j+2])
                global_rep++;
        }
    }
    double rep_penalty = 1.0 + 1.0 * local_rep + 2.0 * global_rep;

    // 3. Perplexity (normalized)
    double ppl = compute_perplexity(words, nw);
    double norm_ppl = ppl / 100.0;  // adjust divisor to taste (50–200)

    // 4. Length reward
    double ideal_len = (g_app.min_gen_len + g_app.max_gen_len) / 2.0;
    double len_diff = fabs(nw - ideal_len);
    double len_reward = 1.0 + 0.8 * exp(-len_diff / 5.0);

    double end_bonus = 0.0;
    if (nw > 0 && (strcmp(words[nw-1], ".") == 0 || strcmp(words[nw-1], "!") == 0 || strcmp(words[nw-1], "?") == 0))
        end_bonus = 0.7;

    // Final score
    double final_score = (norm_ppl * rep_penalty * theme_score) /
                         (len_reward + end_bonus + exact_bonus + 0.5);

    return final_score;
}

static void generate_multiple_candidates(void) {
    g_app.candidate_count = 0;
    g_app.generation_attempts = 0;
    g_app.average_theme_density = 0.0;
    double theme_density_sum = 0.0;

    const int max_attempts = g_app.max_candidates * 10;
    const double MIN_THEME_DENSITY = g_app.min_theme_density > 0 ? g_app.min_theme_density : 0;

    while (g_app.candidate_count < g_app.max_candidates &&
           g_app.generation_attempts < max_attempts)
    {
        g_app.generation_attempts++;
        const char *words[MAX_WORDS_PER_GENERATION + 16] = {0};
        int nw = generate_sequence(words);

        if (nw < g_app.min_gen_len || nw > g_app.max_gen_len) {
            continue;
        }

        // Compute density (even if filter is off)
        int theme_count = 0;
        for (int i = 0; i < nw; i++) {
            if (is_keyword_related(words[i])) theme_count++;
        }
        double density = nw > 0 ? (double)theme_count / nw : 0.0;

        // Density filter (disabled)
        if (density < MIN_THEME_DENSITY) {
            continue;
        }

        const char **copy = malloc(nw * sizeof(const char *));
        if (!copy) continue;
        memcpy(copy, words, nw * sizeof(const char *));

        Candidate *c = &g_app.candidates[g_app.candidate_count];
        c->words = copy;
        c->length = nw;
        g_app.candidate_count++;

        theme_density_sum += density;
    }

    if (g_app.candidate_count > 0) {
        g_app.average_theme_density = theme_density_sum / g_app.candidate_count;
    }

    if (g_app.candidate_count == 0) {
        fprintf(stderr, "Warning: Could not generate any valid candidates "
                        "in %d attempts (min=%d, max=%d)\n",
                        max_attempts, g_app.min_gen_len, g_app.max_gen_len);
    }
}

void display_candidates(void) {
    printf("\n=== Candidate Sequences (%d total) ===\n\n", g_app.candidate_count);

    for (int i = 0; i < g_app.candidate_count; i++) {
        Candidate *c = &g_app.candidates[i];
        const char *temp_words[MAX_WORDS_PER_GENERATION + 16];
        int temp_len = c->length;

        for (int j = 0; j < temp_len; j++) {
            temp_words[j] = c->words[j];
        }
        printf("Candidate #%d  (cleaned length: %d words)\n", i + 1, temp_len);
        print_words_properly(stdout, temp_words, temp_len);
        printf("\n\n");
    }

    printf("====================================\n\n");
}

static inline bool is_in_syn_pool(const char *interned_str) {
    if (!interned_str) return false;
    for (int i = 0; i < g_app.keyword_syn_pool_count; i++) {
        if (g_app.keyword_syn_pool[i] == interned_str) return true;
    }
    return false;
}

void precompute_thematic_starters(void)
{
    // --- Free previous allocation if any ---
    if (g_app.thematic_starters) {
        free(g_app.thematic_starters);
        g_app.thematic_starters = NULL;
    }
    g_app.thematic_starter_count = 0;
    g_app.total_thematic_weight = 0;

    if (g_app.keyword_syn_pool_count == 0 || g_app.total_starters == 0) return;

    // --- First pass: count thematic starters and sum weights ---
    int count = 0;
    long total_weight = 0;

    for (size_t i = 0; i < g_app.starter_table_size; i++) {
        for (Starter *s = g_app.starter_table[i]; s; s = s->next) {
            if (is_in_syn_pool(s->word)) {
                count++;
                if (s->count > 0) total_weight += s->count;
            }
        }
    }

    if (count == 0) return; // Nothing to do

    // --- Allocate array ---
    g_app.thematic_starters = malloc(count * sizeof(ThematicStarter));
    if (!g_app.thematic_starters) return;

    g_app.thematic_starter_count = count;
    g_app.total_thematic_weight = total_weight;

    // --- Second pass: fill thematic starters ---
    int idx = 0;
    for (size_t i = 0; i < g_app.starter_table_size && idx < count; i++) {
        for (Starter *s = g_app.starter_table[i]; s; s = s->next) {
            if (is_in_syn_pool(s->word)) {
                g_app.thematic_starters[idx].word = s->word;
                g_app.thematic_starters[idx].count = s->count;
                idx++;
            }
        }
    }
}

