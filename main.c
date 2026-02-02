#include <stdio.h>
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

// TODO change names if needed, clean & organize 
#define NUM_THREADS 8
#define MAX_LINE_LEN 4096
#define WORD_MAX_LEN 128
#define MAX_WORDS_PER_GENERATION 500
#define MAX_CANDIDATES 400
#define HASH_TABLE_SIZE (1 << 21)
#define STARTER_TABLE_SIZE (1 << 17)
#define INTERN_TABLE_SIZE (1 << 20)
#define ARENA_BLOCK_SIZE (1ULL << 22)
#define CONTEXT_BUF_SIZE 1024
#define MAX_KEYWORDS 8
#define MAX_SYNONYMS_PER_WORD 32
#define MAX_SYN_ENTRIES 8192

#define DEFAULT_MIN_TEXT_LENGTH 50 			// Safety if argument missing
#define DEFAULT_MAX_TEXT_LENGTH 60			// Safety if argument missing
#define HASH_TABLE_SIZE (1 << 21)

typedef struct App App;

typedef struct {
    const char **words;
    int         length;     // token based length
    double      ppl; 
    int         rep_count;  
    double      score;
} Candidate;

typedef struct {
    const char *w[4];
    uint8_t order;
} ContextKey;

typedef struct ArenaBlock {
    struct ArenaBlock *next;
    size_t used;
    char data[ARENA_BLOCK_SIZE];
} ArenaBlock;

typedef struct InternEntry {
    const char *str;
    struct InternEntry *next;
} InternEntry;

typedef struct Next {
    const char *word;
    int count;
    struct Next *next;
} Next;

typedef struct Entry {
    ContextKey  key; 
    Next       *nexts;    
    uint32_t    total_count;   
    struct Entry *next;
} Entry;

typedef struct Starter {
    const char *word;
    int count;
    struct Starter *next;
} Starter;

typedef struct {
    const char *word;
    const char *syns[MAX_SYNONYMS_PER_WORD];
    int num_syns;
} SynEntry;

typedef struct {
    const char *filename;          
    long start_byte;               
    long end_byte;       
    Entry **local_hashtable;
    Starter **local_starters;
    long local_total_starters;
    long local_vocab;
    ArenaBlock *local_arena;
} ThreadData;

typedef struct {
    const char *word;
    long count;          // original frequency (for weighted random selection)
} ThematicStarter;

typedef struct App {
    // ─────────────────────────────────────────────────────────────
    //  1. Configuration & Command-line / runtime parameters
    // ─────────────────────────────────────────────────────────────
    char          *input_file;          // main training corpus
    char          *output_file;         // (currently unused?)
    char          *syn_file;            // synonyms file path (default: synonyms.txt)

    int            min_gen_len;
    int            max_gen_len;
    int            max_candidates;

    int            num_threads;
    unsigned int   seed;
    int            verbose;             // 0 = silent, 1 = info, 2 = debug, ...

    // ─────────────────────────────────────────────────────────────
    //  2. Keyword / Theme filtering
    // ─────────────────────────────────────────────────────────────
    const char   **keywords;            // array of original keywords (interned)
    int            keyword_count;

    const char    *keyword_syn_pool[2048];  // flattened pool of keywords + synonyms
    int            keyword_syn_pool_count;

    ThematicStarter *thematic_starters;     // precomputed weighted thematic starters
    int             thematic_starter_count;
    long            total_thematic_weight;

    // ─────────────────────────────────────────────────────────────
    //  3. Repetition & generation quality control
    // ─────────────────────────────────────────────────────────────
    int            rep_window;          // how far back to check for repetition
    double         rep_penalty;         // multiplier when repetition is detected (e.g. 0.3)

    // ─────────────────────────────────────────────────────────────
    //  4. Model tables & sizes
    // ─────────────────────────────────────────────────────────────
    Entry        **hashtable;           // main n-gram → successors table
    size_t         hashtable_size;

    Starter      **starter_table;       // first-word frequency table
    size_t         starter_table_size;

    InternEntry  **intern_table;        // string interning hash table
    size_t         intern_table_size;

    long           vocab_size;          // approximate number of unique tokens
    long           total_starters;      // total count of first-word occurrences

    atomic_long    total_lines_processed;  // progress tracking across threads

    // ─────────────────────────────────────────────────────────────
    //  5. Candidate generation state
    // ─────────────────────────────────────────────────────────────
    Candidate      candidates[MAX_CANDIDATES];
    int            candidate_count;
    int            best_candidate_index;
    double         best_score;
    int            generation_attempts;

    // ─────────────────────────────────────────────────────────────
    //  6. Memory management – arena allocator
    // ─────────────────────────────────────────────────────────────
    ArenaBlock    *arena_head;          // first block in chain
    ArenaBlock    *arena_current;       // block we're currently allocating from

    // Optional / debug statistics (not yet implemented)
    size_t         arena_total_bytes_used;
    size_t         arena_block_count;

    // ─────────────────────────────────────────────────────────────
    //  7. Custom object pools (freelists) & allocation stats
    // ─────────────────────────────────────────────────────────────
    Entry         *entry_freelist;
    Next          *next_freelist;

    // Optional counters (very useful when debugging leaks / performance)
    long           entry_alloc_count;
    long           next_alloc_count;
    long           entry_from_freelist;
    long           next_from_freelist;

} App;

static App g_app;

void display_global_debug(void);
static int  app_init(void);
static void print_words_properly(FILE *out, const char **words, int count);
static int generate_sequence(const char **out_words);
static inline bool context_equal(const ContextKey *a, const ContextKey *b);
static inline uint64_t hash_context(const ContextKey *c);
static void generate_multiple_candidates(void);
void display_candidates(void);
static double score_candidate(const char **words, int nw);
void precompute_thematic_starters(void);
static inline unsigned long hash_str(const char *str);
const char *intern(const char *s);
int split_into_words(char *line, const char **tokens_out, int max_tokens);
static void select_and_print_best(void);
bool is_in_syn_pool(const char *interned_str);
void arena_init(void);
char *arena_alloc(size_t size);
Next *alloc_next(void);
void free_next(Next *n);
Entry *alloc_entry(void);
void free_entry(Entry *e);

Next *alloc_next(void) {
    if (g_app.next_freelist) {
        Next *n = g_app.next_freelist;
        g_app.next_freelist = n->next;
        memset(n, 0, sizeof(Next));
        return n;
    }
    return calloc(1, sizeof(Next));
}

void free_next(Next *n) {
    if (!n) return;
    n->next = g_app.next_freelist;
    g_app.next_freelist = n;
}

Entry *alloc_entry(void) {
    if (g_app.entry_freelist) {
        Entry *e = g_app.entry_freelist;
        g_app.entry_freelist = e->next;
        memset(e, 0, sizeof(Entry));
        return e;
    }
    return calloc(1, sizeof(Entry));
}

void free_entry(Entry *e) {
    if (!e) return;
    e->next = g_app.entry_freelist;
    g_app.entry_freelist = e;
}

void arena_init(void) {
    if (g_app.arena_head) {
        // already initialized → optional: you can warn or just return
        return;
    }

    ArenaBlock *block = calloc(1, sizeof(ArenaBlock));
    if (!block) {
        fprintf(stderr, "arena_init: calloc failed\n");
        exit(1);
    }

    g_app.arena_head    = block;
    g_app.arena_current = block;
    // arena_total_bytes_used = 0;   // Not tracked anymore
    // arena_block_count = 1;
}

char *arena_alloc(size_t size) {
    if (g_app.arena_current->used + size + 1 > ARENA_BLOCK_SIZE) {
        ArenaBlock *new_block = calloc(1, sizeof(ArenaBlock));
        g_app.arena_current->next = new_block;
        g_app.arena_current = new_block;
    }
    char *p = g_app.arena_current->data + g_app.arena_current->used;
    g_app.arena_current->used += size + 1;
    return p;
}

static void select_and_print_best(void) { // very naive for now
    if (g_app.candidate_count == 0) {
        printf("[CRITICAL] No valid candidates generated.\n");
        return;
    }

    int best_idx = -1;
    double best_score = 1e15;

    for (int i = 0; i < g_app.candidate_count; i++) {
        Candidate *c = &g_app.candidates[i];
        double score = score_candidate(c->words, c->length);

        // Lower score is better; tie-break on length
        if (score < best_score || (score == best_score && c->length > (best_idx >= 0 ? g_app.candidates[best_idx].length : 0))) {
            best_score = score;
            best_idx = i;
        }
    }

    if (best_idx < 0) {
        printf("[CRITICAL] No best candidate found.\n");
        return;
    }

    Candidate *winner = &g_app.candidates[best_idx];
    g_app.best_score = best_score;

    print_words_properly(stdout, winner->words, winner->length);
    putchar('\n');
}

void merge_local_into_global(Entry **local, Entry **global) {
    // Temporary per-bucket hint: points to most recently matched/inserted entry in this bucket
    Entry **hints = calloc(g_app.hashtable_size, sizeof(Entry *));
    if (!hints) {
        fprintf(stderr, "calloc failed for merge hints\n");
        // You could fall back to original behavior or exit — for now just continue without hints
    }

    struct TempMapping {
        Entry *local_entry;
        Entry *global_match;
    };

    struct TempMapping *mappings = NULL;
    size_t map_count = 0;
    size_t map_cap = 0;

    // First pass: find matches or insert new entries
    for (size_t i = 0; i < g_app.hashtable_size; i++) {
        Entry *e = local[i];
        while (e) {
            Entry *next_e = e->next;

            uint64_t h = hash_context(&e->key);
            size_t bucket = h % g_app.hashtable_size;

            Entry *match = NULL;

            // 1. Fast path: check the most recent match/insert in this bucket
            if (hints && hints[bucket]) {
                if (context_equal(&hints[bucket]->key, &e->key)) {
                    match = hints[bucket];
                }
            }

            // 2. Full scan only if hint missed or no hint yet
            if (!match) {
                Entry *g = global[bucket];
                while (g) {
                    if (context_equal(&g->key, &e->key)) {
                        match = g;
                        if (hints) hints[bucket] = g;  // update hint
                        break;
                    }
                    g = g->next;
                }
            }

            // Remember this local → global mapping (for second pass)
            if (map_count >= map_cap) {
                map_cap = map_cap ? map_cap * 2 : 65536;
                struct TempMapping *new_map = realloc(mappings, map_cap * sizeof(*mappings));
                if (!new_map) {
                    fprintf(stderr, "realloc failed in merge\n");
                    // TODO: proper cleanup and fallback
                    free(mappings);
                    free(hints);
                    return;
                }
                mappings = new_map;
            }
            mappings[map_count++] = (struct TempMapping){e, match};

            // 3. If no match → insert this local entry into global chain
            if (!match) {
                e->next = global[bucket];
                global[bucket] = e;
                if (hints) hints[bucket] = e;  // new entry becomes the best hint
            }

            e = next_e;
        }
    }

    // Second pass: merge the Next* lists only for entries that had a match
    for (size_t i = 0; i < map_count; i++) {
        Entry *local_e = mappings[i].local_entry;
        Entry *global_e = mappings[i].global_match;

        if (!global_e) continue;

        Next *ln = local_e->nexts;
        while (ln) {
            Next *next_ln = ln->next;

            Next **insert_ptr = &global_e->nexts;
            Next *gn = global_e->nexts;
            bool found = false;

            while (gn) {
                if (gn->word == ln->word) {           // pointer identity — fast
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

        // We don't need the local Entry struct anymore
        free_entry(local_e);
    }

    free(mappings);
    free(hints);
}

void merge_starters(Starter **local, Starter **global) {
    long merged_buckets = 0;
    for (size_t i = 0; i < g_app.starter_table_size; i++) {
        Starter *s = local[i];
        while (s) {
            Starter *next_s = s->next;

            Starter *g = global[i];
            bool found = false;
            while (g) {
                if (strcmp(g->word, s->word) == 0) {
                    g->count += s->count;
                    found = true;
                    break;
                }
                g = g->next;
            }

            if (!found) {
                s->next = global[i];
                global[i] = s;
            }

            s = next_s;
        }

        merged_buckets++;
        if (merged_buckets % 1000 == 0) {
        }
    }
}

void add_starter_local(Starter **table, long *total, const char *word) {
    if (!word || !*word) return;
    
    if (strlen(word) == 1 && ispunct((unsigned char)word[0])) {
        return;
    }
    
    unsigned long h = hash_str(word) % g_app.starter_table_size;
    Starter *s = table[h];
    while (s) {
        if (s->word == word) {
            s->count++;
            (*total)++;
            return;
        }
        s = s->next;
    }
    Starter *new_s = malloc(sizeof(Starter));
    new_s->word = word;
    new_s->count = 1;
    new_s->next = table[h];
    table[h] = new_s;
    (*total)++;
}

void add_transition_local(Entry **table,const char *window[4],const char *next_word) {
    if (!next_word) return;

    ContextKey ctxs[4];
    int ctx_count = 0;

    for (int len = 1; len <= 4; len++) {
        if (!window[len-1]) break;   
        ContextKey *c = &ctxs[ctx_count++];
        c->order = len;
        for (int i = 0; i < len; i++) {
            c->w[i] = window[i];
        }
    }

    for (int i = 0; i < ctx_count; i++) {
        ContextKey *key = &ctxs[i];
        uint64_t h = hash_context(key);
        size_t idx = h & (g_app.hashtable_size - 1);

        Entry *e = table[idx];
        while (e) {
            if (context_equal(&e->key, key)) {
                Next *n = e->nexts;
                while (n) {
                    if (n->word == next_word) {
                        n->count++;
                        e->total_count++;
                        goto next_suffix;
                    }
                    n = n->next;
                }
                Next *new_n = alloc_next();
                new_n->word  = next_word;
                new_n->count = 1;
                new_n->next  = e->nexts;
                e->nexts     = new_n;
                e->total_count++;
                goto next_suffix;
            }
            e = e->next;
        }
        Entry *new_e       = alloc_entry();
        new_e->key         = *key;            
        new_e->total_count = 1;
        Next *new_n        = alloc_next();
        new_n->word        = next_word;
        new_n->count       = 1;
        new_e->nexts       = new_n;
        new_e->next        = table[idx];
        table[idx]         = new_e;
    next_suffix:;
    }
}

char *arena_alloc_thread(ArenaBlock **arena, size_t size) {
    if ((*arena)->used + size + 1 > ARENA_BLOCK_SIZE) {
        ArenaBlock *new_block = calloc(1, sizeof(ArenaBlock));
        (*arena)->next = new_block;
        *arena = new_block;
    }
    char *p = (*arena)->data + (*arena)->used;
    (*arena)->used += size + 1;
    return p;
}

Next *alloc_next_thread(Next **freelist) {
    if (*freelist) {
        Next *n = *freelist;
        *freelist = n->next;
        memset(n, 0, sizeof(Next));
        return n;
    }
    return calloc(1, sizeof(Next));
}

Entry *alloc_entry_thread(Entry **freelist) {
    if (*freelist) {
        Entry *e = *freelist;
        *freelist = e->next;
        memset(e, 0, sizeof(Entry));
        return e;
    }
    return calloc(1, sizeof(Entry));
}

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

    char line[MAX_LINE_LEN];
    const char *words[256];
    const char *window[4] = {NULL};

    // Local seen array for vocab deduplication
    char *seen = calloc(g_app.intern_table_size, 1);
    if (!seen) {
        fprintf(stderr, "[Thread %ld] Failed to allocate seen array (%ld bytes)\n",
                (long)pthread_self(), g_app.intern_table_size);
        fclose(f);
        return NULL;
    }

    long current_pos = td->start_byte;
    long local_lines = 0;

    while (fgets(line, sizeof(line), f)) {
        // Increment global line counter immediately
        atomic_fetch_add(&g_app.total_lines_processed, 1);
        local_lines++;

        size_t line_len = strlen(line);
        current_pos += line_len + 1;

        // Safety: stop if we've gone past our chunk
        if (current_pos > td->end_byte) {
            break;
        }

        line[strcspn(line, "\n")] = '\0';
        int nw = split_into_words(line, words, 256);

        for (int j = 0; j < nw; j++) {
            const char *curr = words[j];
            unsigned long vh = hash_str(curr) % g_app.intern_table_size;

            if (!seen[vh]) {
                seen[vh] = 1;
                td->local_vocab++;
            }

            bool sent_start = (j == 0) ||
                              (window[0] && strchr(".!?", window[0][strlen(window[0])-1]));

            if (sent_start) {
                add_starter_local(td->local_starters, &td->local_total_starters, curr);
            }

            add_transition_local(td->local_hashtable, window, curr);

            // Slide the window
            window[3] = window[2];
            window[2] = window[1];
            window[1] = window[0];
            window[0] = curr;
        }

		if (g_app.verbose >= 1) {
			long global_processed = atomic_load(&g_app.total_lines_processed);
			static long last_reported = 0;
			
			if (global_processed - last_reported >= 10000) {
				last_reported = global_processed;
				double percent = (double)global_processed / 318786 * 100.0;
				if (percent > 100.0) percent = 100.0;
				printf("\r\033[K%.1f%%", percent);
				fflush(stdout);
			}
		}
    }

    fclose(f);
    free(seen);

    return NULL;
}

int build_model_mt(void){

	memset(g_app.intern_table, 0,g_app.intern_table_size * sizeof(InternEntry*));

    const char *filename = g_app.input_file;
    if (!filename || !*filename) {
        fprintf(stderr, "Error: No input file specified in app config\n");
        return -1;
    }

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

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
    fseek(f, 0, SEEK_SET);
    fclose(f);

    // Allocate global model structures (once)
    if (g_app.hashtable == NULL) {
        g_app.hashtable = calloc(g_app.hashtable_size, sizeof(Entry*));
        if (!g_app.hashtable) {
            perror("calloc failed for global hashtable");
            return -1;
        }
    } else {
        memset(g_app.hashtable, 0, g_app.hashtable_size * sizeof(Entry*));
    }

    if (g_app.starter_table == NULL) {
        g_app.starter_table = calloc(g_app.starter_table_size, sizeof(Starter*));
        if (!g_app.starter_table) {
            perror("calloc failed for starter_table");
            free(g_app.hashtable);
            g_app.hashtable = NULL;
            return -1;
        }
    } else {
        memset(g_app.starter_table, 0, g_app.starter_table_size * sizeof(Starter*));
    }

    g_app.vocab_size     = 0;
    g_app.total_starters = 0;

    // Reset arena head/current if needed (or we could keep chaining, but for simplicity reset)
    g_app.arena_head    = NULL;
    g_app.arena_current = NULL;
    arena_init();  // Re-init global arena (creates first block)

    pthread_t threads[NUM_THREADS];
    ThreadData td[NUM_THREADS];
    long bytes_per_thread = file_size / NUM_THREADS;

    for (int t = 0; t < NUM_THREADS; t++) {
        td[t].filename           = filename;
        td[t].start_byte         = t * bytes_per_thread;
        td[t].end_byte           = (t == NUM_THREADS - 1) ? file_size : (t + 1) * bytes_per_thread;
        td[t].local_hashtable    = calloc(g_app.hashtable_size, sizeof(Entry*));
        td[t].local_starters     = calloc(g_app.starter_table_size, sizeof(Starter*));
        td[t].local_total_starters = 0;
        td[t].local_vocab        = 0;
        td[t].local_arena        = calloc(1, sizeof(ArenaBlock));  // each thread has own arena start

        if (!td[t].local_hashtable || !td[t].local_starters || !td[t].local_arena) {
            fprintf(stderr, "Thread %d allocation failed\n", t);

            for (int i = 0; i < t; i++) {
                free(td[i].local_hashtable);
                free(td[i].local_starters);
            }
            free(g_app.hashtable);
            free(g_app.starter_table);
            g_app.hashtable = NULL;
            g_app.starter_table = NULL;
            return -1;
        }

        pthread_create(&threads[t], NULL, process_chunk, &td[t]);
    }

    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }
    
	long final_processed = atomic_load(&g_app.total_lines_processed);
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double read_parse_time = (ts_end.tv_sec - ts_start.tv_sec) + (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;

    if (g_app.verbose >= 1) {
        printf("\n\nReading & parsing finished:\n");
        printf("  - Total lines processed: %ld\n", final_processed);
        printf("  - Time: %.2f seconds\n", read_parse_time);
        printf("\n");
        fflush(stdout);
    }

    clock_gettime(CLOCK_MONOTONIC, &ts_start); // reset time for mergin phase

    for (int t = 0; t < NUM_THREADS; t++) {
        if (g_app.verbose >= 1) {
            printf("Merging thread %d local table...\n", t);
            fflush(stdout);
        }

        merge_local_into_global(td[t].local_hashtable, g_app.hashtable);
        merge_starters(td[t].local_starters, g_app.starter_table);

        g_app.total_starters += td[t].local_total_starters;
        g_app.vocab_size     += td[t].local_vocab;

        ArenaBlock *b = td[t].local_arena;
        while (b) {
            ArenaBlock *next_b = b->next;
            if (!g_app.arena_head) {
                g_app.arena_head = b;
                g_app.arena_current = b;
                g_app.arena_current->next = NULL;
            } else {
                g_app.arena_current->next = b;
                g_app.arena_current = b;
                g_app.arena_current->next = NULL;
            }
            b = next_b;
        }
    }

	clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double merge_time = (ts_end.tv_sec - ts_start.tv_sec) + (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;

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

static inline uint64_t xxhash64(const char *str) { // xxHash - very fast non-cryptographic hash (public domain)
    uint64_t h = 0x517cc1b727220a95ULL;
    while (*str) {
        h ^= (uint64_t)(unsigned char)*str++;
        h *= 0x5bd1e995ULL;
        h ^= h >> 24;
    }
    return h;
}

static inline unsigned long hash_str(const char *str) {
    if (!str || !*str) return 0;

    unsigned long h = 0xcbf29ce484222325ULL;  // FNV-1a offset basis (64-bit)

    while (*str) {
        h ^= (unsigned char)*str++;
        h *= 0x100000001b3ULL;                // FNV prime
    }

    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;

    return h;
}

static inline uint64_t hash_context(const ContextKey *c) {  // fnv1a style 
    uint64_t h = 0xcbf29ce484222325ULL;  
    for (int i = 0; i < c->order; i++) {
        uintptr_t p = (uintptr_t) c->w[i];         // hash pointer value — very fast and collision-resistant enough
        h ^= p;
        h *= 0x100000001b3ULL;
    }
    return h;
}

static inline bool context_equal(const ContextKey *a, const ContextKey *b) {
    if (a->order != b->order) return false;
    for (int i = 0; i < a->order; i++) {
        if (a->w[i] != b->w[i]) return false;
    }
    return true;
}

const char *intern(const char *s) {
    if (!s || !*s) return "";
    size_t len = strlen(s);
    unsigned long h = hash_str(s) % g_app.intern_table_size;
    InternEntry *e = g_app.intern_table[h];
    while (e) {
        if (strcmp(e->str, s) == 0) return e->str;
        e = e->next;
    }
    char *copy = arena_alloc(len + 1);
    memcpy(copy, s, len);
    copy[len] = '\0';
    InternEntry *new_e = malloc(sizeof(InternEntry));
    new_e->str = copy;
    new_e->next = g_app.intern_table[h];
    g_app.intern_table[h] = new_e;
    return copy;
}

int load_synonyms(void)
{
    const char *filename = g_app.syn_file ? g_app.syn_file : "synonyms.txt";

    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Cannot open synonyms file: %s\n", filename);
        return -1;
    }

    g_app.keyword_syn_pool_count = 0;

    int keywords_added = 0;
    for (int i = 0; i < g_app.keyword_count; i++)
    {
        const char *kw = g_app.keywords[i];
        if (!kw) continue;

        bool dup = false;
        for (int j = 0; j < g_app.keyword_syn_pool_count; j++)
        {
            if (g_app.keyword_syn_pool[j] == kw) { dup = true; break; }
        }
        if (!dup && g_app.keyword_syn_pool_count < 2048)
        {
            g_app.keyword_syn_pool[g_app.keyword_syn_pool_count++] = kw;
            keywords_added++;
        }
    }

    int groups_used = 0;
    int synonyms_added = 0;

    char line[8192];
    while (fgets(line, sizeof(line), f))
    {
        line[strcspn(line, "\n")] = '\0';
        if (strlen(line) < 3) continue;
        char *words[128];
        int word_count = 0;
        char *token = strtok(line, ",");
        while (token && word_count < 128)
        {
            while (isspace(*token)) token++;
            char *end = token + strlen(token) - 1;
            while (end >= token && isspace(*end)) *end-- = '\0';

            if (*token)
            {
                words[word_count++] = token;
            }
            token = strtok(NULL, ",");
        }

        if (word_count == 0) continue;

        bool relevant = false;
        for (int w = 0; w < word_count && !relevant; w++)
        {
            const char *candidate = intern(words[w]);  // ensure interned

            for (int k = 0; k < g_app.keyword_count; k++)
            {
                if (g_app.keywords[k] == candidate)
                {
                    relevant = true;
                    break;
                }
            }
        }

        if (!relevant) continue;
        groups_used++;
        int local_added = 0;

        for (int w = 0; w < word_count; w++)
        {
            const char *syn_interned = intern(words[w]);

            bool dup = false;
            for (int j = 0; j < g_app.keyword_syn_pool_count; j++)
            {
                if (g_app.keyword_syn_pool[j] == syn_interned)
                {
                    dup = true;
                    break;
                }
            }
            if (!dup && g_app.keyword_syn_pool_count < 2048)
            {
                g_app.keyword_syn_pool[g_app.keyword_syn_pool_count++] = syn_interned;
                local_added++;
                synonyms_added++;
            }
        }
    }

    fclose(f);
    return (g_app.keyword_syn_pool_count > 0) ? 0 : -2;
}


bool is_punct(const char *tok) {
    return strlen(tok) == 1 && strchr(".,!?;:", tok[0]);
}

int split_into_words(char *line, const char **tokens_out, int max_tokens) {
    int count = 0;
    char *p = line;
    while (*p && count < max_tokens) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (*p == '\0') break;
        if (strchr(".,!?;:", *p)) {
            char tmp[2] = {*p, '\0'};
            tokens_out[count++] = intern(tmp);
            p++;
            continue;
        }
        char *start = p;
        int len = 0;
        while (*p && !isspace((unsigned char)*p) && !strchr(".,!?;:", *p)) {
            len++; p++;
        }
        while (len > 0 && ispunct((unsigned char)*start) && *start != '\'') { start++; len--; }
        while (len > 0 && ispunct((unsigned char)*(start + len - 1)) && *(start + len - 1) != '\'') len--;
        if (len <= 0) continue;
        if (len >= WORD_MAX_LEN) len = WORD_MAX_LEN - 1;
        char tmp[WORD_MAX_LEN];
        strncpy(tmp, start, len);
        tmp[len] = '\0';
        for (int i = 0; i < len; i++)
            tmp[i] = tolower((unsigned char)tmp[i]);
        tokens_out[count++] = intern(tmp);
    }
    return count;
}

// Computes perplexity of a sequence (single function, log-domain, efficient smoothing)
static double compute_perplexity(const char **words, int nw)
{
    if (nw < 2) return 1e12;

    double log_sum = 0.0;
    int valid_count = 0;

    // Fixed interpolation weights (sum should be ≈1, tune if needed)
    static const double lambdas[5] = {0.0, 0.15, 0.25, 0.35, 0.25}; // order 0→4

    // Small constant for Lidstone smoothing
    const double alpha = 0.01;

    for (int pos = 1; pos < nw; pos++) {
        int ctx_len = pos < 4 ? pos : 4;  // use up to 4-gram

        double prob = 0.0;
        double remaining = 1.0;

        // Backoff from highest to lowest order
        for (int ord = ctx_len; ord >= 1; ord--) {
            if (remaining < 1e-9) break;

            // Build context
            ContextKey ctx = {0};
            ctx.order = (uint8_t)ord;
            for (int k = 0; k < ord; k++) {
                ctx.w[k] = words[pos - 1 - k];
            }

            uint64_t h = hash_context(&ctx);
            size_t bucket = h & (g_app.hashtable_size - 1);

            long observed = 0;
            long match_count = 0;

            // Lookup
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

            // Lidstone smoothing
            double denom = observed + alpha * g_app.vocab_size;
            double p = (observed > 0) ? (match_count + alpha) / denom : alpha / denom;

            // Interpolate
            double w = lambdas[ord] * remaining;
            prob += w * p;
            remaining -= w;
        }

        // Remaining mass → uniform
        double uniform_p = 1.0 / (g_app.vocab_size + 10000.0); // slightly boosted
        prob += remaining * uniform_p;

        // Accumulate log-prob
        if (prob > 0) {
            log_sum += log(prob);
            valid_count++;
        }
    }

    if (valid_count == 0) return 1e12;

    double avg_log_p = log_sum / valid_count;
    double ppl = exp(-avg_log_p);

    return ppl;
}

bool is_keyword_related(const char *w) // Linear search could be slow 
{
    if (!w) {
        return false;
    }

    for (int i = 0; i < g_app.keyword_syn_pool_count; i++) {
        if (g_app.keyword_syn_pool[i] == w) {   // pointer equality
            return true;
        }
    }

    return false;
}

// Selects a starter word, preferring one from keyword_syn_pool if possible
static const char *select_thematic_starter(void)
{
    // Prefer thematic if we have any
    if (g_app.thematic_starter_count > 0 && g_app.total_thematic_weight > 0) {
        long r = rand() % g_app.total_thematic_weight;
        long sum = 0;

        for (int i = 0; i < g_app.thematic_starter_count; i++) {
            sum += g_app.thematic_starters[i].count;
            if (r < sum) {
                const char *chosen = g_app.thematic_starters[i].word;
                return chosen;
            }
        }
    }

    // Fallback to original full weighted random
    if (g_app.total_starters > 0) {
        long r = rand() % g_app.total_starters;
        long sum = 0;
        for (size_t i = 0; i < g_app.starter_table_size; i++) {
            for (Starter *s = g_app.starter_table[i]; s; s = s->next) {
                sum += s->count;
                if (r < sum) {
                    return s->word;
                }
            }
        }
    }

    // Ultimate fallback
    return intern("hello");
}

static int generate_sequence(const char **out_words)
{
    int max_tokens = MAX_WORDS_PER_GENERATION; // safety cap
    const char *prev[4] = {NULL};
    int generated = 0;

    // Select starter
    prev[0] = select_thematic_starter();

    // Push initial context (up to 4 words) kept for future multi-word starters
    for (int k = 3; k >= 0; k--) {
        if (prev[k]) {
            if (generated >= max_tokens) break;
            out_words[generated++] = prev[k];
        }
    }

    // Main generation loop
    while (generated < max_tokens) {
        const char *candidates[4] = {NULL};           // slot 0 = order 4, slot 1 = order 3, ...
        double cand_weights[4] = {0.0};               // only used in fallback case

        bool has_syn_candidate = false;
        int best_syn_slot = -1;                       // lowest slot = highest order

        for (int order = 4; order >= 1; order--) {
            if (generated < order) continue;

            ContextKey ctx = {0};
            ctx.order = (uint8_t)order;
            for (int k = 0; k < order; k++) {
                ctx.w[k] = out_words[generated - 1 - k];
            }

            uint64_t h = hash_context(&ctx);
            size_t idx = h & (g_app.hashtable_size - 1);

            for (Entry *e = g_app.hashtable[idx]; e; e = e->next) {
                if (context_equal(&e->key, &ctx)) {

                    // Pick the most frequent next word
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

                        // Check if this candidate is in the synonym pool
                        if (is_keyword_related(best_next)) {
                            has_syn_candidate = true;
                            if (best_syn_slot == -1 || slot < best_syn_slot) {
                                best_syn_slot = slot;
                            }
                        }
                    }
                    break;
                }
            }
        }

		// ────────────────────────────────────────────────
		// Decide next word
		// ────────────────────────────────────────────────
		const char *next = NULL;

		// Priority 1: probabilistic synonym preference
		bool use_syn_priority = false;
		if (has_syn_candidate && best_syn_slot != -1) {
			use_syn_priority = (rand() % 100 < 70);  // adjust 70 as needed
		}

		if (use_syn_priority) {
			next = candidates[best_syn_slot];
		}

		// Priority 2: Jelinek-Mercer weighted random + repetition penalty
		if (!next) {
			double base_weights[4] = {0.9, 0.7, 0.4, 0.2};
			double total_w = 0.0;

			for (int slot = 0; slot < 4; slot++) {
				if (candidates[slot]) {
				    double wgt = base_weights[slot];

				    // REPETITION PENALTY: reduce weight if candidate was used recently
				    bool is_recent = false;
				    for (int k = 1; k <= g_app.rep_window && generated - k >= 0; k++) {
				        if (out_words[generated - k] == candidates[slot]) {
				            is_recent = true;
				            break;
				        }
				    }
				    if (is_recent) {
				        wgt *= g_app.rep_penalty;  // e.g. 0.3 = strong penalty
				    }

				    total_w += wgt;
				}
			}

			if (total_w > 0.0) {
				double r = ((double)rand() / RAND_MAX) * total_w;
				double acc = 0.0;
				for (int slot = 0; slot < 4; slot++) {
				    if (candidates[slot]) {
				        double wgt = base_weights[slot];

				        // Re-apply penalty for consistency in accumulation
				        bool is_recent = false;
				        for (int k = 1; k <= g_app.rep_window && generated - k >= 0; k++) {
				            if (out_words[generated - k] == candidates[slot]) {
				                is_recent = true;
				                break;
				            }
				        }
				        if (is_recent) wgt *= g_app.rep_penalty;

				        acc += wgt;
				        if (r <= acc) {
				            next = candidates[slot];
				            break;
				        }
				    }
				}
			}
		}

		// Ultimate fallback (unchanged)
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
        if (next == NULL) {
            next = intern(".");
        }

        out_words[generated++] = next;
        
        // Early stopping: stop if we reached max length
		if (generated >= g_app.max_gen_len) {
			break;
		}

        // Early stopping if we reach end of phrase after min_gen_len
        if (generated >= g_app.min_gen_len) {
            if (strcmp(next, ".") == 0 ||
                strcmp(next, "!") == 0 ||
                strcmp(next, "?") == 0) {
                break;
            }
        }
    }

    // ────────────────────────────────────────────────
    // Post-processing (unchanged)
    // ────────────────────────────────────────────────
    if (generated == 0) {
        return 0;
    }

    // remove leading punctuation (except standalone ".")
    int start = 0;
    while (start < generated &&
           ispunct((unsigned char)out_words[start][0]) &&
           strcmp(out_words[start], ".") != 0) {
        start++;
    }
    int len = generated - start;

    // Clamp length
    if (len > g_app.max_gen_len) len = g_app.max_gen_len;
    if (len < g_app.min_gen_len) len = g_app.min_gen_len;

    // Trim to last period if possible
    for (int i = len - 1; i >= 0; i--) {
        if (strcmp(out_words[start + i], ".") == 0) {
            len = i + 1;
            break;
        }
    }

    // Shift back
    for (int i = 0; i < len; i++)
        out_words[i] = out_words[start + i];
    generated = len;

    // Capitalization after '.'
    bool cap = true;
    for (int i = 0; i < generated; i++) {
        const char *w = out_words[i];
        if (!w) continue;
        if (cap && isalpha((unsigned char)w[0])) {
            char buf[WORD_MAX_LEN];
            strncpy(buf, w, sizeof(buf) - 1);
            buf[sizeof(buf) - 1] = '\0';
            buf[0] = toupper((unsigned char)buf[0]);
            out_words[i] = intern(buf);
            cap = false;
        }
        if (strcmp(w, ".") == 0)
            cap = true;
    }

    // Force final dot
    if (generated > 0 && strcmp(out_words[generated - 1], ".") != 0)
        out_words[generated++] = intern(".");
    else if (generated == 0)
        out_words[generated++] = intern(".");

    return generated;
}

static void print_words_properly(FILE *out, const char **words, int count) {
    for (int i = 0; i < count; i++) {
        const char *w = words[i];
        if (!w) continue;

        if (i == 0) {
            // First word: capitalize if letter
            if (isalpha((unsigned char)w[0])) {
                fputc(toupper((unsigned char)w[0]), out);
                fputs(w + 1, out);
            } else {
                fputs(w, out);
            }
        } else {
            // Add space before word unless previous was opening punctuation or this is closing punctuation
            if (strchr("([{'\"", words[i-1][0]) == NULL &&  // no space after opening
                strchr(".,!?;:", w[0]) == NULL) {           // no space before punctuation
                fputc(' ', out);
            }
            fputs(w, out);
        }
    }
}

void free_model(void)
{
    // ─────────────────────────────────────────────────────────────
    //  1. Free model data structures (hashtable + transitions)
    // ─────────────────────────────────────────────────────────────
    if (g_app.hashtable) {
        for (size_t i = 0; i < g_app.hashtable_size; i++) {
            Entry *e = g_app.hashtable[i];
            while (e) {
                Entry *next_entry = e->next;

                // Free the Next chain (successor words)
                Next *n = e->nexts;
                while (n) {
                    Next *next_n = n->next;
                    free_next(n);           // uses g_app.next_freelist
                    n = next_n;
                }

                free_entry(e);              // uses g_app.entry_freelist
                e = next_entry;
            }
        }
        free(g_app.hashtable);
        g_app.hashtable = NULL;
    }

    // ─────────────────────────────────────────────────────────────
    //  2. Free starter table
    // ─────────────────────────────────────────────────────────────
    if (g_app.starter_table) {
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
    }

    // ─────────────────────────────────────────────────────────────
    //  3. Free string interning table
    // ─────────────────────────────────────────────────────────────
    if (g_app.intern_table) {
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
    }

    // ─────────────────────────────────────────────────────────────
    //  4. Free the arena (all string copies live here)
    // ─────────────────────────────────────────────────────────────
    ArenaBlock *block = g_app.arena_head;
    while (block) {
        ArenaBlock *next_block = block->next;
        free(block);
        block = next_block;
    }
    g_app.arena_head    = NULL;
    g_app.arena_current = NULL;

    // ─────────────────────────────────────────────────────────────
    //  5. Free candidate storage
    // ─────────────────────────────────────────────────────────────
    for (int i = 0; i < g_app.candidate_count; i++) {
        free(g_app.candidates[i].words);
        g_app.candidates[i].words = NULL;
    }
    g_app.candidate_count = 0;
    g_app.best_candidate_index = -1;
    g_app.best_score = 0.0;
    g_app.generation_attempts = 0;

    // ─────────────────────────────────────────────────────────────
    //  6. Free keyword-related dynamic arrays
    // ─────────────────────────────────────────────────────────────
    if (g_app.keywords) {
        free(g_app.keywords);
        g_app.keywords = NULL;
    }
    g_app.keyword_count = 0;

    if (g_app.thematic_starters) {
        free(g_app.thematic_starters);
        g_app.thematic_starters = NULL;
    }
    g_app.thematic_starter_count = 0;
    g_app.total_thematic_weight = 0;

    // ─────────────────────────────────────────────────────────────
    //  7. Clean up freelists (return nodes to system)
    // ─────────────────────────────────────────────────────────────
    while (g_app.entry_freelist) {
        Entry *e = g_app.entry_freelist;
        g_app.entry_freelist = e->next;
        free(e);
    }
    g_app.entry_freelist = NULL;

    while (g_app.next_freelist) {
        Next *n = g_app.next_freelist;
        g_app.next_freelist = n->next;
        free(n);
    }
    g_app.next_freelist = NULL;

    // ─────────────────────────────────────────────────────────────
    //  8. Reset model statistics
    // ─────────────────────────────────────────────────────────────
    g_app.vocab_size     = 0;
    g_app.total_starters = 0;

    // Optional: if you want to be very explicit
    // memset(&g_app.candidates, 0, sizeof(g_app.candidates));
}

static int parse_arguments(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_file> [keywords=kw1,kw2,...] [min_length=N] [max_length=N]\n", argv[0]);
        return 1;
    }

    g_app.input_file = argv[1];

    if (g_app.keywords == NULL) {
        g_app.keywords = malloc(MAX_KEYWORDS * sizeof(char *));
        if (!g_app.keywords) {
            perror("malloc for keywords failed");
            return 1;
        }
        memset(g_app.keywords, 0, MAX_KEYWORDS * sizeof(char *));
    }
    g_app.keyword_count = 0;

    for (int i = 2; i < argc; i++) {
        if (strncmp(argv[i], "keywords=", 9) == 0) {
            char tmp[MAX_LINE_LEN];
            snprintf(tmp, sizeof(tmp), "%s", argv[i] + 9);
            char *tok = strtok(tmp, ",");
            while (tok) {
                while (isspace(*tok)) tok++;
                char *end = tok + strlen(tok) - 1;
                while (end >= tok && isspace(*end))
                    *end-- = '\0';
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
    if (g_app.min_gen_len > g_app.max_gen_len) {
        fprintf(stderr, "min_length > max_length\n");
        return 1;
    }
    return 0;
}

/* -------------------------------------------------- */
/* main                                               */
/* -------------------------------------------------- */

int main(int argc, char **argv) {

    if (app_init() != 0) {
        fprintf(stderr, "Fatal: init failed\n");
        return 1;
    }
    
	if (parse_arguments(argc, argv) != 0)
		goto shutdown;
		
	if (load_synonyms() < 0)
		fprintf(stderr, "Warning: synonyms disabled\n");
	
	if (build_model_mt() != 0) {
		fprintf(stderr, "Model build failed\n");
		goto shutdown;
	}
	
	if (g_app.vocab_size == 0) {
		fprintf(stderr, "CRITICAL: vocab_size is 0 → will cause div-by-zero in get_smoothed_prob\n");
	}
	if (g_app.total_starters == 0) {
		fprintf(stderr, "WARNING: no starter words → generation will fallback to 'hello'\n");
	}
	
	precompute_thematic_starters();
    
    generate_multiple_candidates();

	if (g_app.verbose >= 1) {
		display_global_debug();
		//display_candidates(); No need for now
	}
		
	select_and_print_best();
	

shutdown:
	//free_model();
	return 0;
}

static double score_candidate(const char **words, int nw)
{
    if (nw < g_app.min_gen_len) return 1e15;  // too short = very bad

    // ────────────────────────────────────────────────
    // 1. Theme / synonym density (biggest factor)
    // ────────────────────────────────────────────────
    int theme_count = 0;
    int exact_keyword_count = 0;

    for (int i = 0; i < nw; i++) {
        if (is_keyword_related(words[i])) {
            theme_count++;
            // Extra bonus for exact keyword match
            for (int k = 0; k < g_app.keyword_count; k++) {
                if (words[i] == g_app.keywords[k]) {
                    exact_keyword_count++;
                    break;
                }
            }
        }
    }

    double theme_ratio = (double)theme_count / nw;
	double theme_score = 30.0 * (1.0 - theme_ratio);  // was 10 or 20 — now 30 or even 50
    double exact_bonus = exact_keyword_count * 0.5;           // reward exact keywords

    // ────────────────────────────────────────────────
    // 2. Repetition penalty (local + global)
    // ────────────────────────────────────────────────
    int local_rep = 0;
    for (int j = 2; j < nw; j++) {
        // Simple bigram alternation penalty
        if (words[j] == words[j-2] && words[j-1] == words[j+1]) {
            local_rep++;
        }
    }

    // Global repetition: count how many times the same trigram appears
    int global_rep = 0;
    for (int i = 0; i < nw - 2; i++) {
        for (int j = i + 3; j < nw - 2; j++) {
            if (words[i] == words[j] && words[i+1] == words[j+1] && words[i+2] == words[j+2]) {
                global_rep++;
            }
        }
    }

    double rep_penalty = 1.0 + 0.5 * local_rep + 1.0 * global_rep;

    // ────────────────────────────────────────────────
    // 3. Perplexity (fluency under Markov model)
    // ────────────────────────────────────────────────
    double ppl = compute_perplexity(words, nw);

    // ────────────────────────────────────────────────
    // 4. Length reward (encourage good length)
    // ────────────────────────────────────────────────
    double ideal_len = (g_app.min_gen_len + g_app.max_gen_len) / 2.0;
    double len_diff = fabs(nw - ideal_len);
    double len_reward = 1.0 + 0.8 * exp(-len_diff / 5.0);  // Gaussian-like around ideal

    // Bonus if ends with sentence punctuation
    double end_bonus = 0.0;
    if (nw > 0) {
        const char *last = words[nw-1];
        if (strcmp(last, ".") == 0 || strcmp(last, "!") == 0 || strcmp(last, "?") == 0) {
            end_bonus = 0.7;
        }
    }

    // ────────────────────────────────────────────────
    // Final composite score (lower = better)
    // ────────────────────────────────────────────────
    double final_score = 
        (ppl * rep_penalty * theme_score) /          // bad fluency + rep + low theme → high score
        (len_reward + end_bonus + exact_bonus + 0.1); // good length + ending + exact keywords → low score

    return final_score;
}

static void generate_multiple_candidates(void) {
    g_app.candidate_count = 0;
    g_app.generation_attempts = 0;
    
    const int max_attempts = g_app.max_candidates * 10; // while boundary 

    while (g_app.candidate_count < g_app.max_candidates && g_app.generation_attempts < max_attempts) {
        g_app.generation_attempts++;

        const char *words[MAX_WORDS_PER_GENERATION + 16] = {0};
        int nw = generate_sequence(words);

        if (nw >= g_app.min_gen_len && nw <= g_app.max_gen_len) {
            const char **copy = malloc(nw * sizeof(const char *));
            if (!copy) continue;

            memcpy(copy, words, nw * sizeof(const char *));

            g_app.candidates[g_app.candidate_count].words  = copy;
            g_app.candidates[g_app.candidate_count].length = nw;
            g_app.candidate_count++;

            if (g_app.verbose >= 2) {
                printf("Valid candidate #%d added (length %d after %d attempts)\n",
                       g_app.candidate_count, nw, g_app.generation_attempts);
            }
        }
    }

    if (g_app.candidate_count == 0 && g_app.generation_attempts >= max_attempts) {
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


/* -------------------------------------------------- */
/* app lifecycle                                      */
/* -------------------------------------------------- */

static int app_init(void)
{
    // 1. Zero out the entire struct first (safest starting point)
    memset(&g_app, 0, sizeof(g_app));

    // ─────────────────────────────────────────────────────────────
    //  Config / Files
    // ─────────────────────────────────────────────────────────────
    g_app.input_file  = NULL;
    g_app.output_file = NULL;
    g_app.syn_file    = "synonyms.txt";           // default value

    // ─────────────────────────────────────────────────────────────
    //  Generation parameters
    // ─────────────────────────────────────────────────────────────
    g_app.min_gen_len     = DEFAULT_MIN_TEXT_LENGTH;
    g_app.max_gen_len     = DEFAULT_MAX_TEXT_LENGTH;
    g_app.max_candidates  = MAX_CANDIDATES;
    g_app.rep_window      = 5;
    g_app.rep_penalty     = 0.3;

    // ─────────────────────────────────────────────────────────────
    //  Threading & performance
    // ─────────────────────────────────────────────────────────────
    g_app.num_threads = NUM_THREADS;

    // ─────────────────────────────────────────────────────────────
    //  Randomness & logging
    // ─────────────────────────────────────────────────────────────
    g_app.seed    = (unsigned)time(NULL);
    g_app.verbose = 1;                    // default: info level

    srand(g_app.seed);

    // ─────────────────────────────────────────────────────────────
    //  Keyword / Theme filtering
    // ─────────────────────────────────────────────────────────────
    g_app.keyword_count        = 0;
    g_app.keywords             = NULL;                    // will be allocated later if needed
    g_app.keyword_syn_pool_count = 0;
    memset(g_app.keyword_syn_pool, 0, sizeof(g_app.keyword_syn_pool));

    // ─────────────────────────────────────────────────────────────
    //  Model table sizes (power-of-2 preferred for fast modulo)
    // ─────────────────────────────────────────────────────────────
    g_app.hashtable_size     = HASH_TABLE_SIZE;       // 1<<21 = ~2M buckets
    g_app.starter_table_size = STARTER_TABLE_SIZE;    // 1<<17 = 131072
    g_app.intern_table_size  = INTERN_TABLE_SIZE;     // 1<<20 = ~1M buckets

    // ─────────────────────────────────────────────────────────────
    //  Model state & statistics
    // ─────────────────────────────────────────────────────────────
    g_app.hashtable       = NULL;                     // allocated in build_model_mt
    g_app.starter_table   = NULL;                     // allocated in build_model_mt
    g_app.intern_table    = NULL;                     // ← we allocate it here now
    g_app.vocab_size      = 0;
    g_app.total_starters  = 0;

    // ─────────────────────────────────────────────────────────────
    //  Candidate generation state
    // ─────────────────────────────────────────────────────────────
    g_app.candidate_count       = 0;
    g_app.best_candidate_index  = -1;
    g_app.best_score            = 0.0;
    g_app.generation_attempts   = 0;
    memset(g_app.candidates, 0, sizeof(g_app.candidates));

    // ─────────────────────────────────────────────────────────────
    //  Memory management (arena)
    // ─────────────────────────────────────────────────────────────
    g_app.arena_head    = NULL;
    g_app.arena_current = NULL;
    arena_init();           // creates the first block
    
    // Custom freelists for Entry and Next nodes
	g_app.entry_freelist = NULL;
	g_app.next_freelist  = NULL;

	// Optional: reset counters if you track them
	g_app.entry_alloc_count       = 0;
	g_app.next_alloc_count        = 0;
	g_app.entry_from_freelist     = 0;
	g_app.next_from_freelist      = 0;

    // ─────────────────────────────────────────────────────────────
    //  Allocate the interning hash table (double pointer style)
    // ─────────────────────────────────────────────────────────────
    g_app.intern_table = calloc(g_app.intern_table_size, sizeof(InternEntry *));
    if (!g_app.intern_table) {
        fprintf(stderr, "app_init: failed to allocate intern_table (%zu entries)\n",
                g_app.intern_table_size);
        return -1;
    }

    // Optional: if you want to be extra paranoid during development
    // for (size_t i = 0; i < g_app.intern_table_size; i++) {
    //     g_app.intern_table[i] = NULL;
    // }

    return 0;
}

void display_global_debug(void) {
    printf("\n\n=== App Global Debug Dump ===\n\n");

    printf("--- Configuration ---\n");
    printf("  input_file     : %s\n", g_app.input_file     ? g_app.input_file     : "(null)");
    printf("  output_file    : %s\n", g_app.output_file    ? g_app.output_file    : "(null)");
    printf("  syn_file       : %s\n", g_app.syn_file       ? g_app.syn_file       : "(null)");
    printf("  min_gen_len    : %d\n", g_app.min_gen_len);
    printf("  max_gen_len    : %d\n", g_app.max_gen_len);
    printf("  max_candidates : %d\n", g_app.max_candidates);
    printf("  num_threads    : %d\n", g_app.num_threads);
    printf("  seed           : %u\n", g_app.seed);
    printf("  verbose        : %d\n", g_app.verbose);
    printf("\n");

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
    printf("\n");

    printf("--- Synonym / Theme Pool ---\n");
    printf("  keyword_syn_pool_count : %d\n", g_app.keyword_syn_pool_count);

    if (g_app.keyword_syn_pool_count > 0) {
        // Count how many are original keywords (for breakdown)
        int counted_keywords = 0;
        for (int i = 0; i < g_app.keyword_syn_pool_count; i++) {
            const char *w = g_app.keyword_syn_pool[i];
            bool is_orig_keyword = false;
            for (int k = 0; k < g_app.keyword_count; k++) {
                if (w == g_app.keywords[k]) {
                    is_orig_keyword = true;
                    break;
                }
            }
            if (is_orig_keyword) counted_keywords++;
        }

        int counted_synonyms = g_app.keyword_syn_pool_count - counted_keywords;

        printf("    └─ original keywords : %d\n", counted_keywords);
        printf("    └─ added from groups : %d\n", counted_synonyms);
        printf("\n");

        // Show full pool (or limit if too big)
        const int MAX_SHOWN = 10;  // prevent flooding terminal
        printf("  Pool content (%d total):\n", g_app.keyword_syn_pool_count);
        for (int i = 0; i < g_app.keyword_syn_pool_count && i < MAX_SHOWN; i++) {
            printf("    %s\n", g_app.keyword_syn_pool[i]);
        }
        if (g_app.keyword_syn_pool_count > MAX_SHOWN) {
            printf("    ... (and %d more)\n", g_app.keyword_syn_pool_count - MAX_SHOWN);
        }
    } else {
        printf("    (empty - no keywords or matching synonym groups loaded)\n");
    }
    printf("\n");
    
	// Thematic Starters (precomputed)
	printf("--- Thematic Starters ---\n");
	printf(" thematic_starter_count : %d\n", g_app.thematic_starter_count);
	printf(" total_thematic_weight  : %ld\n", g_app.total_thematic_weight);

	if (g_app.thematic_starter_count > 0 && g_app.thematic_starters) {
		// Optional: count how many are original keywords
		int counted_keywords = 0;
		for (int i = 0; i < g_app.thematic_starter_count; i++) {
		    const char *w = g_app.thematic_starters[i].word;
		    bool is_orig = false;
		    for (int k = 0; k < g_app.keyword_count; k++) {
		        if (w == g_app.keywords[k]) {
		            is_orig = true;
		            break;
		        }
		    }
		    if (is_orig) counted_keywords++;
		}
		int counted_extra = g_app.thematic_starter_count - counted_keywords;

		printf(" └─ from original keywords : %d\n", counted_keywords);
		printf(" └─ from synonym groups    : %d\n", counted_extra);
		printf("\n");
		
		// sort by count
		for (int i = 0; i < g_app.thematic_starter_count - 1; i++) {
		    for (int j = 0; j < g_app.thematic_starter_count - i - 1; j++) {
		        if (g_app.thematic_starters[j].count < g_app.thematic_starters[j+1].count) {
		            ThematicStarter temp = g_app.thematic_starters[j];
		            g_app.thematic_starters[j] = g_app.thematic_starters[j+1];
		            g_app.thematic_starters[j+1] = temp;
		        }
		    }
		}

		// Show list (limited)
		const int MAX_SHOWN = 10;
		printf(" Thematic starters (%d total):\n", g_app.thematic_starter_count);
		for (int i = 0; i < g_app.thematic_starter_count && i < MAX_SHOWN; i++) {
		    printf("   %s (count: %ld)\n",
		           g_app.thematic_starters[i].word,
		           g_app.thematic_starters[i].count);
		}
		if (g_app.thematic_starter_count > MAX_SHOWN) {
		    printf("   ... (and %d more)\n", g_app.thematic_starter_count - MAX_SHOWN);
		}
	} else {
		printf(" (none - no matching starters in synonym pool)\n");
	}
	printf("\n");

    printf("--- Model Stats ---\n");
    printf("  hashtable         : %p\n", (void *)g_app.hashtable);
    printf("  hashtable_size    : %zu\n", g_app.hashtable_size);   // use %zu for size_t
    printf("  intern_table_size : %zu\n", g_app.intern_table_size);
    printf("  starter_table     : %p\n", (void *)g_app.starter_table);
    printf("  starter_table_size: %zu\n", g_app.starter_table_size);
    printf("  vocab_size        : %zu\n", g_app.vocab_size);
    printf("  total_starters    : %zu\n", g_app.total_starters);
    printf("\n");

    printf("--- Memory Arena ---\n");
    printf("  head       : %p\n", (void *)g_app.arena_head);
    printf("  current    : %p\n", (void *)g_app.arena_current);
    printf("  block_size : %zu\n", (size_t)ARENA_BLOCK_SIZE);
    printf("\n");
    
	printf("--- Candidates ---\n");
	printf("  Attempts made:          %d\n", g_app.generation_attempts);
	printf("  Valid candidates kept:  %d\n", g_app.candidate_count);
	if (g_app.candidate_count > 0) {
		printf("  Best score found:       %.4f\n", g_app.best_score);
	} else {
		printf("  (none generated - check corpus, keywords or parameters)\n");
	}
	printf("\n");

    printf("\n=== End of debug dump ===\n\n");
}

bool is_in_syn_pool(const char *interned_str) {
    for (int i = 0; i < g_app.keyword_syn_pool_count; i++) {
        if (g_app.keyword_syn_pool[i] == interned_str) return true;
    }
    return false;
}

void precompute_thematic_starters(void)
{
    // Clear previous allocation if any (for safety / re-init)
    if (g_app.thematic_starters) {
        free(g_app.thematic_starters);
        g_app.thematic_starters = NULL;
    }

    g_app.thematic_starter_count = 0;
    g_app.total_thematic_weight = 0;

    if (g_app.keyword_syn_pool_count == 0 || g_app.total_starters == 0) {
        printf("[INFO] No thematic starters to precompute (empty pool or no starters).\n");
        return;
    }

    // Pass 1: count + sum weights
    for (size_t i = 0; i < g_app.starter_table_size; i++) {
        for (Starter *s = g_app.starter_table[i]; s; s = s->next) {
            if (is_keyword_related(s->word)) {
                g_app.thematic_starter_count++;
                g_app.total_thematic_weight += s->count;
            }
        }
    }

    if (g_app.thematic_starter_count == 0) {
        printf("[INFO] No starters match the synonym pool.\n");
        return;
    }

    // Allocate
    g_app.thematic_starters = malloc(g_app.thematic_starter_count * sizeof(ThematicStarter));
    if (!g_app.thematic_starters) {
        fprintf(stderr, "[ERROR] Failed to allocate thematic starters array\n");
        g_app.thematic_starter_count = 0;
        g_app.total_thematic_weight = 0;
        return;
    }

    // Pass 2: fill the array
    int idx = 0;
    for (size_t i = 0; i < g_app.starter_table_size; i++) {
        for (Starter *s = g_app.starter_table[i]; s; s = s->next) {
            if (is_keyword_related(s->word)) {
                g_app.thematic_starters[idx].word = s->word;
                g_app.thematic_starters[idx].count = s->count;
                idx++;
            }
        }
    }
}
