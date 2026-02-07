/* markov.c - Thematic keyword-biased Markov chain text generator
 *
 * Multi-threaded corpus training, arena allocation, string interning,
 * synonym-guided generation.
 *
 * Copyright (c) 2025-2026 Marën Derveld
 * SPDX-License-Identifier: MIT
 */

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

// Configuration constants
#define NUM_THREADS 8
#define MAX_LINE_LEN 4096
#define WORD_MAX_LEN 128
#define MAX_WORDS_PER_GENERATION 1000
#define MAX_CANDIDATES 300

// Table sizes — powers of 2 for fast modulo
#define HASH_TABLE_SIZE (1 << 21)    // ~2M buckets
#define STARTER_TABLE_SIZE (1 << 17) // 131k buckets
#define INTERN_TABLE_SIZE (1 << 20)  // ~1M buckets

#define ARENA_BLOCK_SIZE (1UL << 22) // 4 MiB blocks
#define CONTEXT_BUF_SIZE 1024

// Keyword / synonym limits
#define MAX_KEYWORDS 8
#define MAX_SYNONYMS_PER_WORD 32
#define MAX_SYN_ENTRIES 8192

// Generation defaults
#define DEFAULT_MIN_TEXT_LENGTH 50
#define DEFAULT_MAX_TEXT_LENGTH 60

typedef struct App App;

// ──────────────────────────────────────────────────────────────────────────────
// Core data structures
// ──────────────────────────────────────────────────────────────────────────────

typedef struct
{
    const char** words;
    int length;
    double ppl;
    int rep_count;
    double score;
} Candidate;

typedef struct
{
    const char* w[4];
    uint8_t order;
} ContextKey;

typedef struct ArenaBlock
{
    struct ArenaBlock* next;
    size_t used;
    char data[ARENA_BLOCK_SIZE];
} ArenaBlock;

typedef struct InternEntry
{
    const char* str;
    struct InternEntry* next;
} InternEntry;

typedef struct Next
{
    const char* word;
    int count;
    struct Next* next;
} Next;

typedef struct Entry
{
    ContextKey key;
    Next* nexts;
    uint32_t total_count;
    struct Entry* next;
} Entry;

typedef struct Starter
{
    const char* word;
    int count;
    struct Starter* next;
} Starter;

typedef struct
{
    const char* word;
    const char* syns[MAX_SYNONYMS_PER_WORD];
    int num_syns;
} SynEntry;

typedef struct
{
    const char* filename;
    long start_byte;
    long end_byte;
    Entry** local_hashtable;
    Starter** local_starters;
    long local_total_starters;
    long local_vocab;
    ArenaBlock* local_arena;
} ThreadData;

typedef struct
{
    Entry** local;
    Entry** global;
    size_t start_bucket;
    size_t end_bucket;
} MergeRangeArgs;

typedef struct
{
    const char* word;
    long count;
} ThematicStarter;

// ──────────────────────────────────────────────────────────────────────────────
// Global application state
// ──────────────────────────────────────────────────────────────────────────────

typedef struct App
{
    // ── Configuration ───────────────────────────────────────────────────────
    char* input_file;
    char* output_file;
    char* syn_file;

    int min_gen_len;
    int max_gen_len;
    int max_candidates;

    int num_threads;
    unsigned int seed;
    int verbose;

    // ── Theme / keyword filtering ───────────────────────────────────────────
    const char** keywords;
    int keyword_count;

    const char* keyword_syn_pool[2048];
    int keyword_syn_pool_count;

    ThematicStarter* thematic_starters;
    int thematic_starter_count;
    long total_thematic_weight;

    // ── Generation quality controls ─────────────────────────────────────────
    int rep_window;
    double rep_penalty;
    double temperature;
    double average_theme_density;
    int top_k;

    // ── Model tables ────────────────────────────────────────────────────────
    Entry** hashtable;
    size_t hashtable_size;

    Starter** starter_table;
    size_t starter_table_size;

    InternEntry** intern_table;
    size_t intern_table_size;

    long vocab_size;
    long total_starters;

    atomic_long total_lines_processed;
    long total_lines;

    // ── Candidate generation state ──────────────────────────────────────────
    Candidate candidates[MAX_CANDIDATES];
    int candidate_count;
    int best_candidate_index;
    double best_score;
    int generation_attempts;

    double theme_force_probability;
    double min_theme_density;

    const char** prefix_words;
    int prefix_len;

    // ── Arena allocator ─────────────────────────────────────────────────────
    ArenaBlock* arena_head;
    ArenaBlock* arena_current;

    size_t arena_total_bytes_used;
    size_t arena_block_count;

    // ── Object freelists & debug counters ───────────────────────────────────
    Entry* entry_freelist;
    Next* next_freelist;

    long entry_alloc_count;
    long next_alloc_count;
    long entry_from_freelist;
    long next_from_freelist;
} App;

static App g_app;
static pthread_mutex_t intern_mutex = PTHREAD_MUTEX_INITIALIZER;

// ──────────────────────────────────────────────────────────────────────────────
// Function prototypes (grouped by responsibility)
// ──────────────────────────────────────────────────────────────────────────────

// Debug / output
void display_global_debug(void);
void display_candidates(void);
static void print_words_properly(FILE* out, const char** words, int count);

// Initialization & argument parsing
static int app_init(void);
static int parse_arguments(int argc, char** argv);

// Model building & destruction
static void free_model(void);
int build_model_mt(void);

// Text & token utilities
static inline bool is_punct(const char* tok);
int split_into_words(char* line, const char** tokens_out, int max_tokens);
long get_lines_nb(const char* filename);

// Keyword / synonym handling
int load_synonyms(void);
static inline bool is_keyword_related(const char* w);
static inline bool is_in_syn_pool(const char* interned_str);
static const char* select_thematic_starter(void);
void precompute_thematic_starters(void);

// Generation & scoring
static int generate_sequence(const char** out_words);
static void generate_multiple_candidates(void);
static double score_candidate(const char** words, int nw);
static double compute_perplexity(const char** words, int nw);
static void select_and_print_best(int n);

// Hashing & context utilities
static inline uint64_t xxhash64(const char* str);
static inline unsigned long fnv64_hash(const char* str);
static inline uint64_t hash_context(const ContextKey* c);
static inline bool context_equal(const ContextKey* a, const ContextKey* b);
static inline const char* intern(const char* s);

// Transition table operations
void add_transition_local_thread(Entry** table, const char* window[4],
                                 const char* next_word, ArenaBlock** arena);
void merge_local_into_global(Entry** local, Entry** global);
void merge_starters(Starter** local, Starter** global);
void add_starter_local(Starter** table, long* total, const char* word);

// Thread worker
void* process_chunk(void* arg);

// Arena allocator
void arena_init(void);
char* arena_alloc(size_t size);
char* arena_alloc_thread(ArenaBlock** arena, size_t size);
Next* alloc_next_thread_arena(ArenaBlock** arena);
Entry* alloc_entry_thread_arena(ArenaBlock** arena);

// Object pool (freelist) alloc/free
Next* alloc_next(void);
void free_next(Next* n);
Entry* alloc_entry(void);
void free_entry(Entry* e);

static int app_init(void)
{
    // Zero the entire global state — safest starting point
    memset(&g_app, 0, sizeof(g_app));

    // ── Core configuration ──────────────────────────────────────────────────
    g_app.syn_file = "synonyms.txt";

    // Generation quality defaults (tunable)
    g_app.min_gen_len = DEFAULT_MIN_TEXT_LENGTH;
    g_app.max_gen_len = DEFAULT_MAX_TEXT_LENGTH;
    g_app.max_candidates = MAX_CANDIDATES;

    g_app.temperature = 1.2; // slight creativity boost
    g_app.top_k = 8;
    g_app.rep_penalty = 0.05;             // strong repetition suppression
    g_app.rep_window = 12;                // look-back for detecting repeats
    g_app.theme_force_probability = 0.40; // moderate thematic pressure
    g_app.min_theme_density = 0.05;       // minimum acceptable theme ratio

    // Threading & randomness
    g_app.num_threads = NUM_THREADS;
    g_app.seed = (unsigned) time(NULL);
    srand(g_app.seed);

    // Verbosity (0 = silent, 1 = info, 2+ = debug)
    g_app.verbose = 1;

    // ── Model & memory ──────────────────────────────────────────────────────
    g_app.hashtable_size = HASH_TABLE_SIZE;
    g_app.starter_table_size = STARTER_TABLE_SIZE;
    g_app.intern_table_size = INTERN_TABLE_SIZE;

    // Allocate permanent interning table (never resized)
    g_app.intern_table = calloc(g_app.intern_table_size, sizeof(InternEntry*));
    if (!g_app.intern_table)
    {
        fprintf(stderr,
                "app_init: failed to allocate intern_table (%zu buckets)\n",
                g_app.intern_table_size);
        return -1;
    }

    // Small fixed array for user-provided keywords
    g_app.keywords = calloc(MAX_KEYWORDS, sizeof(const char*));
    if (!g_app.keywords)
    {
        fprintf(stderr, "app_init: failed to allocate keywords array\n");
        free(g_app.intern_table);
        g_app.intern_table = NULL;
        return -1;
    }

    // ── Initialize arena (first block created lazily) ───────────────────────
    arena_init();

    // Optional: very light startup trace
    if (g_app.verbose >= 2)
    {
        fprintf(
            stderr,
            "[init] app state zeroed, intern table allocated, arena ready\n");
    }

    return 0;
}

static void print_str_or_null(FILE* out, const char* label, const char* value)
{
    fprintf(out, "%-22s : %s\n", label, value ? value : "(null)");
}

static void print_ptr_or_null(FILE* out, const char* label, const void* ptr)
{
    fprintf(out, "%-22s : %p\n", label, ptr);
}

static void print_int_field(FILE* out, const char* label, int value)
{
    fprintf(out, "%-22s : %d\n", label, value);
}

static void print_long_field(FILE* out, const char* label, long value)
{
    fprintf(out, "%-22s : %ld\n", label, value);
}

static void print_size_t_field(FILE* out, const char* label, size_t value)
{
    fprintf(out, "%-22s : %zu\n", label, value);
}

static void print_double_field(FILE* out, const char* label, double value,
                               int prec)
{
    fprintf(out, "%-22s : %.*f\n", label, prec, value);
}

static void print_keywords(FILE* out)
{
    if (g_app.keyword_count == 0 || !g_app.keywords)
    {
        fprintf(out, "  keywords             : (none)\n");
        return;
    }

    fprintf(out, "  keywords             : ");
    for (int i = 0; i < g_app.keyword_count; i++)
    {
        fprintf(out, "%s%s", g_app.keywords[i],
                (i < g_app.keyword_count - 1) ? ", " : "\n");
    }
}

static void print_syn_pool_summary(FILE* out)
{
    int n = g_app.keyword_syn_pool_count;
    if (n == 0)
    {
        fprintf(out, "  synonym pool         : (empty)\n");
        return;
    }

    int orig_count = 0;
    for (int i = 0; i < n; i++)
    {
        const char* w = g_app.keyword_syn_pool[i];
        for (int k = 0; k < g_app.keyword_count; k++)
        {
            if (w == g_app.keywords[k])
            {
                orig_count++;
                break;
            }
        }
    }

    fprintf(out,
            "  synonym pool         : %d total (%d original keywords, %d from "
            "groups)\n",
            n, orig_count, n - orig_count);

    const int MAX_SHOWN = 10;
    fprintf(out, "    first %d:\n", (n < MAX_SHOWN) ? n : MAX_SHOWN);
    for (int i = 0; i < n && i < MAX_SHOWN; i++)
    {
        fprintf(out, "      %s\n", g_app.keyword_syn_pool[i]);
    }
    if (n > MAX_SHOWN)
    {
        fprintf(out, "      ... and %d more\n", n - MAX_SHOWN);
    }
}

static void print_thematic_starters(FILE* out)
{
    int n = g_app.thematic_starter_count;
    if (n == 0 || !g_app.thematic_starters)
    {
        fprintf(out, "  thematic starters    : (none)\n");
        return;
    }

    int orig_count = 0;
    for (int i = 0; i < n; i++)
    {
        const char* w = g_app.thematic_starters[i].word;
        for (int k = 0; k < g_app.keyword_count; k++)
        {
            if (w == g_app.keywords[k])
            {
                orig_count++;
                break;
            }
        }
    }

    fprintf(out, "  thematic starters    : %d total, weight %ld\n", n,
            g_app.total_thematic_weight);
    fprintf(out, "    └─ %d from keywords, %d from synonyms\n", orig_count,
            n - orig_count);

    const int MAX_SHOWN = 10;
    fprintf(out, "    first %d:\n", (n < MAX_SHOWN) ? n : MAX_SHOWN);
    for (int i = 0; i < n && i < MAX_SHOWN; i++)
    {
        fprintf(out, "      %-20s (count: %ld)\n",
                g_app.thematic_starters[i].word,
                g_app.thematic_starters[i].count);
    }
    if (n > MAX_SHOWN)
    {
        fprintf(out, "      ... and %d more\n", n - MAX_SHOWN);
    }
}

void display_global_debug(void)
{
    puts("\n=== Application Debug Snapshot ===\n");

    puts("Configuration");
    print_str_or_null(stdout, "input_file", g_app.input_file);
    print_str_or_null(stdout, "output_file", g_app.output_file);
    print_str_or_null(stdout, "synonyms file", g_app.syn_file);
    print_int_field(stdout, "min_gen_len", g_app.min_gen_len);
    print_int_field(stdout, "max_gen_len", g_app.max_gen_len);
    print_int_field(stdout, "max_candidates", g_app.max_candidates);
    print_int_field(stdout, "rep_window", g_app.rep_window);
    print_double_field(stdout, "rep_penalty", g_app.rep_penalty, 2);
    print_int_field(stdout, "num_threads", g_app.num_threads);
    print_int_field(stdout, "seed", (int) g_app.seed);
    print_int_field(stdout, "verbose", g_app.verbose);

    puts("\nKeywords & Theme");
    print_keywords(stdout);
    print_syn_pool_summary(stdout);
    print_thematic_starters(stdout);

    puts("\nModel & Tables");
    print_long_field(stdout, "total_lines", g_app.total_lines);
    print_ptr_or_null(stdout, "hashtable", g_app.hashtable);
    print_size_t_field(stdout, "hashtable_size", g_app.hashtable_size);
    print_ptr_or_null(stdout, "intern_table", g_app.intern_table);
    print_size_t_field(stdout, "intern_table_size", g_app.intern_table_size);
    print_ptr_or_null(stdout, "starter_table", g_app.starter_table);
    print_size_t_field(stdout, "starter_table_size", g_app.starter_table_size);
    print_long_field(stdout, "vocab_size", g_app.vocab_size);
    print_long_field(stdout, "total_starters", g_app.total_starters);
    print_long_field(stdout, "lines_processed",
                     atomic_load(&g_app.total_lines_processed));

    puts("\nArena Allocator");
    print_ptr_or_null(stdout, "arena_head", g_app.arena_head);
    print_ptr_or_null(stdout, "arena_current", g_app.arena_current);
    print_size_t_field(stdout, "block_size", ARENA_BLOCK_SIZE);
    print_size_t_field(stdout, "bytes_used", g_app.arena_total_bytes_used);
    print_size_t_field(stdout, "block_count", g_app.arena_block_count);

    puts("\nPrefix / Context");
    print_int_field(stdout, "prefix_len", g_app.prefix_len);
    if (g_app.prefix_len > 0 && g_app.prefix_words)
    {
        printf("  prefix_words         : ");
        for (int i = 0; i < g_app.prefix_len; i++)
        {
            printf("%s%s", g_app.prefix_words[i],
                   (i < g_app.prefix_len - 1) ? " " : "\n");
        }
    }
    else
    {
        puts("  prefix_words         : (none)");
    }

    puts("\nGeneration Controls");
    print_int_field(stdout, "rep_window", g_app.rep_window);
    print_double_field(stdout, "rep_penalty", g_app.rep_penalty, 2);
    print_double_field(stdout, "temperature", g_app.temperature, 2);
    print_int_field(stdout, "top_k", g_app.top_k);
    print_double_field(stdout, "theme_force_prob",
                       g_app.theme_force_probability, 2);
    print_double_field(stdout, "avg_theme_density", g_app.average_theme_density,
                       3);

    puts("\nCandidate Generation");
    print_int_field(stdout, "generation_attempts", g_app.generation_attempts);
    print_int_field(stdout, "candidate_count", g_app.candidate_count);
    if (g_app.candidate_count > 0)
    {
        print_double_field(stdout, "best_score", g_app.best_score, 4);
    }
    else
    {
        puts("  best_score           : (no candidates generated)");
    }

    puts("\nAllocation Stats");
    print_ptr_or_null(stdout, "entry_freelist", g_app.entry_freelist);
    print_ptr_or_null(stdout, "next_freelist", g_app.next_freelist);
    print_long_field(stdout, "entry_alloc_count", g_app.entry_alloc_count);
    print_long_field(stdout, "next_alloc_count", g_app.next_alloc_count);
    print_long_field(stdout, "entry_from_freelist", g_app.entry_from_freelist);
    print_long_field(stdout, "next_from_freelist", g_app.next_from_freelist);

    puts("\n=== End of debug snapshot ===\n");
}

/**
 * Count the number of lines in a text file.
 *
 * @param filename  Path to the file (must not be NULL or empty string)
 * @return          Number of lines on success (> 0 or 0 for empty file)
 *                  -1 on failure (file not found, permission denied, read
 * error, etc.)
 */
long get_lines_nb(const char* restrict filename)
{
    if (!filename || !*filename)
    {
        fprintf(stderr, "get_lines_nb: invalid filename (NULL or empty)\n");
        return -1;
    }

    FILE* f = fopen(filename, "r");
    if (!f)
    {
        fprintf(stderr, "get_lines_nb: cannot open '%s': %s\n", filename,
                strerror(errno));
        return -1;
    }

    long line_count = 0;
    char buffer[8192];

    // Lecture efficace ligne par ligne
    while (fgets(buffer, sizeof(buffer), f))
    {
        line_count++;
    }

    // Vérification d'erreur de lecture (disque plein, I/O error, etc.)
    if (ferror(f))
    {
        fprintf(stderr, "get_lines_nb: read error on '%s': %s\n", filename,
                strerror(errno));
        fclose(f);
        return -1;
    }

    if (fclose(f) == EOF)
    {
        fprintf(stderr, "get_lines_nb: fclose failed on '%s': %s\n", filename,
                strerror(errno));
        return -1;
    }

    return line_count;
}

/* ----------------------------------------------------------------------------
   Object freelists (Next & Entry nodes)
   ----------------------------------------------------------------------------
 */

/**
 * Allocate a Next node — prefer freelist reuse when possible.
 * Returns NULL on allocation failure (caller must handle).
 */
Next* alloc_next(void)
{
    Next* n;

    if (g_app.next_freelist)
    {
        n = g_app.next_freelist;
        g_app.next_freelist = n->next;

        /* Clear visible fields to avoid leaking stale data */
        n->word = NULL;
        n->count = 0;
        n->next = NULL;

        if (g_app.verbose >= 3)
        {
            g_app.next_from_freelist++;
        }

        return n;
    }

    n = calloc(1, sizeof(*n));
    if (!n)
    {
        fprintf(stderr, "[alloc_next] calloc failed: %s\n", strerror(errno));
        return NULL;
    }

    if (g_app.verbose >= 3)
    {
        g_app.next_alloc_count++;
    }

    return n;
}

/**
 * Return a Next node to the freelist. No-op if NULL.
 */
void free_next(Next* restrict n)
{
    if (!n)
        return;

    n->word = NULL;
    n->count = 0;

    n->next = g_app.next_freelist;
    g_app.next_freelist = n;
}

/**
 * Allocate an Entry node — prefer freelist reuse when possible.
 * Returns NULL on allocation failure (caller must handle).
 */
Entry* alloc_entry(void)
{
    Entry* e;

    if (g_app.entry_freelist)
    {
        e = g_app.entry_freelist;
        g_app.entry_freelist = e->next;

        /* Clear visible fields to avoid leaking stale data */
        e->key = (ContextKey) {0};
        e->nexts = NULL;
        e->total_count = 0;
        e->next = NULL;

        if (g_app.verbose >= 3)
        {
            g_app.entry_from_freelist++;
        }

        return e;
    }

    e = calloc(1, sizeof(*e));
    if (!e)
    {
        fprintf(stderr, "[alloc_entry] calloc failed: %s\n", strerror(errno));
        return NULL;
    }

    if (g_app.verbose >= 3)
    {
        g_app.entry_alloc_count++;
    }

    return e;
}

/**
 * Return an Entry node to the freelist. No-op if NULL.
 */
void free_entry(Entry* restrict e)
{
    if (!e)
        return;

    e->nexts = NULL;
    e->total_count = 0;

    e->next = g_app.entry_freelist;
    g_app.entry_freelist = e;
}

/* ----------------------------------------------------------------------------
   Arena-based allocator (grow-only, block-chained)
   ----------------------------------------------------------------------------
 */

/**
 * Initialize the global arena if not already done.
 * Creates the first 4 MiB block.
 */
void arena_init(void)
{
    if (g_app.arena_head)
    {
        return; // already initialized
    }

    ArenaBlock* block = calloc(1, sizeof(ArenaBlock));
    if (!block)
    {
        fprintf(stderr, "[arena_init] calloc failed: %s\n", strerror(errno));
        exit(EXIT_FAILURE); // fatal — arena is critical
    }

    g_app.arena_head = block;
    g_app.arena_current = block;
    block->used = 0;

    if (g_app.verbose >= 2)
    {
        fprintf(stderr, "[arena] first block allocated (%zu bytes)\n",
                ARENA_BLOCK_SIZE);
    }
}

/**
 * Allocate size bytes from the arena (grow if needed).
 * Adds +1 byte padding (helps with null-termination when needed).
 * Never returns NULL — fatal exit on failure.
 */
char* arena_alloc(size_t size)
{
    if (!g_app.arena_current)
    {
        arena_init();
    }

    /* Need new block? */
    if (g_app.arena_current->used + size + 1 > ARENA_BLOCK_SIZE)
    {
        ArenaBlock* new_block = calloc(1, sizeof(ArenaBlock));
        if (!new_block)
        {
            fprintf(stderr,
                    "[arena_alloc] failed to grow arena (%zu bytes): %s\n",
                    size, strerror(errno));
            exit(EXIT_FAILURE); // fatal — arena is critical
        }

        g_app.arena_current->next = new_block;
        g_app.arena_current = new_block;

        if (g_app.verbose >= 3)
        {
            g_app.arena_block_count++;
        }
    }

    char* p = g_app.arena_current->data + g_app.arena_current->used;
    g_app.arena_current->used += size + 1;

    if (g_app.verbose >= 3)
    {
        g_app.arena_total_bytes_used += size + 1;
    }

    return p;
}

/**
 * Archive all generated candidates to archive.txt (append mode),
 * then sort them by score (lowest first, longer preferred on tie),
 * and print the top N best ones to stdout.
 *
 * Updates g_app.best_score with the top candidate's score.
 */
static void select_and_print_best(int n)
{
    if (g_app.candidate_count == 0)
    {
        fputs("\n[CRITICAL] No valid candidates were generated.\n", stderr);
        return;
    }

    // Limit n to actual number of candidates
    if (n > g_app.candidate_count)
    {
        n = g_app.candidate_count;
    }

    // ── Step 1: Archive everything ──────────────────────────────────────────

    FILE* archive = fopen("archive.txt", "a");
    if (archive)
    {
        for (int i = 0; i < g_app.candidate_count; i++)
        {
            const Candidate* c = &g_app.candidates[i];
            print_words_properly(archive, c->words, c->length);
            fputs("\n---\n", archive);
        }
        fclose(archive);
    }
    else
    {
        fprintf(stderr, "[WARNING] Cannot open archive.txt for appending: %s\n",
                strerror(errno));
        // Continue anyway — archiving is not critical
    }

    // ── Step 2: Score all candidates ────────────────────────────────────────

    typedef struct
    {
        int idx;
        double score;
    } ScoredCandidate;

    ScoredCandidate scored[g_app.candidate_count];

    for (int i = 0; i < g_app.candidate_count; i++)
    {
        scored[i].idx = i;
        scored[i].score = score_candidate(g_app.candidates[i].words,
                                          g_app.candidates[i].length);
    }

    // ── Step 3: Sort (lowest score first, longer on tie) ────────────────────
    // Bubble sort is acceptable: MAX_CANDIDATES around 300 → ~45k comparisons
    // max

    for (int i = 0; i < g_app.candidate_count - 1; i++)
    {
        for (int j = i + 1; j < g_app.candidate_count; j++)
        {
            const Candidate* ci = &g_app.candidates[scored[i].idx];
            const Candidate* cj = &g_app.candidates[scored[j].idx];

            if (scored[j].score < scored[i].score ||
                (scored[j].score == scored[i].score && cj->length > ci->length))
            {
                ScoredCandidate tmp = scored[i];
                scored[i] = scored[j];
                scored[j] = tmp;
            }
        }
    }

    // ── Step 4: Print top N ─────────────────────────────────────────────────

    putchar('\n');

    for (int i = 0; i < n; i++)
    {
        const Candidate* c = &g_app.candidates[scored[i].idx];

        printf("[Candidate %d] Score: %.4f\n", i + 1, scored[i].score);
        print_words_properly(stdout, c->words, c->length);
        putchar('\n');
        fputs("---\n", stdout);
    }

    // Remember best score for later use (debug / logging / etc.)
    g_app.best_score = scored[0].score;
}

/**
 * Merge a range of buckets from one thread-local hashtable into the global one.
 * Thread-safe for disjoint bucket ranges.
 *
 * - First pass: locate or insert each local Entry into global buckets
 * - Second pass: merge successor lists (Next nodes), sum counts
 * - Frees local Entry nodes after merging (arena-allocated)
 *
 * @param arg  MergeRangeArgs* (thread argument)
 * @return     always NULL (pthread requirement)
 */
void* merge_local_into_global_range(void* arg)
{
    MergeRangeArgs* mra = arg;
    Entry** local = mra->local;
    Entry** global = mra->global;

    const size_t range_size = mra->end_bucket - mra->start_bucket;

    // Optional per-bucket hint to avoid full scans on repeated keys
    Entry** hints = calloc(range_size, sizeof(Entry*));
    if (!hints)
    {
        hints = NULL; // fallback: always scan
    }

    // Temporary array to remember which local entries map to which global ones
    struct
    {
        Entry* local;
        Entry* global;
    }* mappings = NULL;

    size_t map_count = 0;
    size_t map_capacity = 0;

    // ── Pass 1: Place or find each local entry in global ─────────────────────

    for (size_t bucket_idx = mra->start_bucket; bucket_idx < mra->end_bucket;
         bucket_idx++)
    {
        Entry* local_e = local[bucket_idx];

        while (local_e)
        {
            Entry* next_local = local_e->next;

            // Compute global bucket (re-hash because pointers differ)
            uint64_t h = hash_context(&local_e->key);
            size_t global_bucket = h % g_app.hashtable_size;

            Entry* global_match = NULL;

            // Fast path: use hint if available and still valid
            size_t hint_idx = bucket_idx - mra->start_bucket;
            if (hints && hints[hint_idx] &&
                context_equal(&hints[hint_idx]->key, &local_e->key))
            {
                global_match = hints[hint_idx];
            }

            // Slow path: scan the global collision chain
            if (!global_match)
            {
                Entry* g = global[global_bucket];
                while (g)
                {
                    if (context_equal(&g->key, &local_e->key))
                    {
                        global_match = g;
                        if (hints)
                            hints[hint_idx] = g;
                        break;
                    }
                    g = g->next;
                }
            }

            // Remember mapping for second pass
            if (map_count >= map_capacity)
            {
                map_capacity = map_capacity ? map_capacity * 2 : 65536;
                void* new_mem =
                    realloc(mappings, map_capacity * sizeof(*mappings));
                if (!new_mem)
                {
                    free(mappings);
                    free(hints);
                    fprintf(stderr,
                            "[merge_range] realloc failed for mappings\n");
                    return NULL;
                }
                mappings = new_mem;
            }

            mappings[map_count++] = (typeof(*mappings)) {local_e, global_match};

            // Insert new entry if no match found
            if (!global_match)
            {
                local_e->next = global[global_bucket];
                global[global_bucket] = local_e;
                if (hints)
                    hints[hint_idx] = local_e;
            }

            local_e = next_local;
        }
    }

    // ── Pass 2: Merge successor lists (Next*) for matched entries
    // ─────────────

    for (size_t i = 0; i < map_count; i++)
    {
        Entry* local_e = mappings[i].local;
        Entry* global_e = mappings[i].global;

        if (!global_e)
        {
            // New entry already inserted → nothing to merge
            continue;
        }

        Next* local_next = local_e->nexts;
        while (local_next)
        {
            Next* next_local = local_next->next;

            bool merged = false;
            Next** insert_pos = &global_e->nexts;
            Next* global_next = global_e->nexts;

            while (global_next)
            {
                // Pointer equality: all words are interned
                if (global_next->word == local_next->word)
                {
                    global_next->count += local_next->count;
                    global_e->total_count += local_next->count;
                    free_next(local_next);
                    merged = true;
                    break;
                }
                insert_pos = &global_next->next;
                global_next = global_next->next;
            }

            if (!merged)
            {
                local_next->next = *insert_pos;
                *insert_pos = local_next;
                global_e->total_count += local_next->count;
            }

            local_next = next_local;
        }

        // Local Entry node is no longer needed (merged or inserted)
        free_entry(local_e);
    }

    free(mappings);
    free(hints);

    return NULL;
}

/**
 * Merge thread-local starter tables into the global starter table.
 * If a word already exists globally, its count is incremented.
 * Otherwise, the starter node is inserted directly (ownership transfer).
 *
 * Called after all threads finish parsing.
 */
void merge_starters(Starter** local, Starter** global)
{
    for (size_t bucket = 0; bucket < g_app.starter_table_size; bucket++)
    {
        Starter* local_s = local[bucket];

        while (local_s)
        {
            Starter* next_local = local_s->next;
            Starter* global_s = global[bucket];
            bool merged = false;

            while (global_s)
            {
                if (global_s->word == local_s->word)
                { // interned strings → pointer comparison safe
                    global_s->count += local_s->count;
                    merged = true;
                    free(local_s);
                    break;
                }
                global_s = global_s->next;
            }

            if (!merged)
            {
                local_s->next = global[bucket];
                global[bucket] = local_s;
            }

            local_s = next_local;
        }
    }

    if (g_app.verbose >= 2)
    {
        fprintf(stderr, "[merge_starters] processed %zu buckets\n",
                g_app.starter_table_size);
    }
}

/**
 * Add a starter word to a thread-local starter table.
 * If the word already exists, increment its count.
 * Otherwise, insert a new Starter node.
 */
void add_starter_local(Starter** table, long* total, const char* word)
{
    if (!word || !*word)
        return; // ignore null or empty strings
    if (strlen(word) == 1 && ispunct((unsigned char) word[0]))
        return; // skip single-char punctuation

    unsigned long h = fnv64_hash(word) % g_app.starter_table_size;
    Starter* s = table[h];

    while (s)
    {
        if (s->word == word)
        { // interned string → pointer comparison safe
            s->count++;
            (*total)++;
            return;
        }
        s = s->next;
    }

    Starter* new_s = calloc(1, sizeof(Starter));
    if (!new_s)
    {
        fprintf(stderr, "[add_starter_local] Memory allocation failed\n");
        exit(1);
    }

    new_s->word = word;
    new_s->count = 1;
    new_s->next = table[h];
    table[h] = new_s;
    (*total)++;
}

/**
 * Thread-local arena allocation functions.
 * Allocates memory from per-thread ArenaBlock chains.
 */
char* arena_alloc_thread(ArenaBlock** arena, size_t size)
{
    if (!arena || !*arena)
    {
        fprintf(stderr, "[arena_alloc_thread] Invalid arena pointer\n");
        exit(1);
    }

    // Allocate new block if current one cannot fit requested size (+1 for
    // safety)
    if ((*arena)->used + size + 1 > ARENA_BLOCK_SIZE)
    {
        ArenaBlock* new_block = calloc(1, sizeof(ArenaBlock));
        if (!new_block)
        {
            fprintf(stderr,
                    "[arena_alloc_thread] Failed to allocate ArenaBlock\n");
            exit(1);
        }
        (*arena)->next = new_block;
        *arena = new_block;
    }

    char* ptr = (*arena)->data + (*arena)->used;
    (*arena)->used += size + 1;
    return ptr;
}

Next* alloc_next_thread_arena(ArenaBlock** arena)
{
    Next* n = (Next*) arena_alloc_thread(arena, sizeof(Next));
    memset(n, 0, sizeof(Next));
    return n;
}

Entry* alloc_entry_thread_arena(ArenaBlock** arena)
{
    Entry* e = (Entry*) arena_alloc_thread(arena, sizeof(Entry));
    memset(e, 0, sizeof(Entry));
    return e;
}

/**
 * Add a word transition to a thread-local hashtable using a thread-local arena.
 * Builds all 1–4 word contexts from the sliding window.
 * Updates existing entries or allocates new Entry/Next nodes as needed.
 */
void add_transition_local_thread(
    Entry** table,         // local hashtable
    const char* window[4], // last 4 words (sliding context)
    const char* next_word, // next word to add
    ArenaBlock** arena     // thread-local arena
)
{
    if (!next_word || !table || !arena || !*arena)
        return;

    // Build 1–4 length contexts from the window
    ContextKey ctxs[4];
    int ctx_count = 0;
    for (int len = 1; len <= 4; len++)
    {
        if (!window[len - 1])
            break;
        ContextKey* c = &ctxs[ctx_count++];
        c->order = len;
        for (int i = 0; i < len; i++)
            c->w[i] = window[i];
    }

    for (int i = 0; i < ctx_count; i++)
    {
        ContextKey* key = &ctxs[i];
        size_t idx = hash_context(key) & (g_app.hashtable_size - 1);

        Entry* e = table[idx];
        while (e)
        {
            if (context_equal(&e->key, key))
            {
                // Update existing Next list
                Next* n = e->nexts;
                while (n)
                {
                    if (n->word == next_word)
                    {
                        n->count++;
                        e->total_count++;
                        goto next_context;
                    }
                    n = n->next;
                }

                // Add new Next node
                Next* new_n = alloc_next_thread_arena(arena);
                new_n->word = next_word;
                new_n->count = 1;
                new_n->next = e->nexts;
                e->nexts = new_n;
                e->total_count++;
                goto next_context;
            }
            e = e->next;
        }

        // Entry not found: allocate new Entry and first Next node
        Entry* new_e = alloc_entry_thread_arena(arena);
        new_e->key = *key;
        new_e->total_count = 1;

        Next* new_n = alloc_next_thread_arena(arena);
        new_n->word = next_word;
        new_n->count = 1;
        new_n->next = NULL;

        new_e->nexts = new_n;
        new_e->next = table[idx];
        table[idx] = new_e;

    next_context:;
    }
}

/**
 * Add a word transition to a hashtable using the global allocator (freelist /
 * malloc). Builds all 1–4 word contexts from the sliding window. Updates
 * existing entries or allocates new Entry/Next nodes as needed.
 */
void add_transition_local(
    Entry** table,         // hashtable (global or local)
    const char* window[4], // last 4 words (sliding context)
    const char* next_word  // next word to add
)
{
    if (!table || !next_word)
        return;

    // Build 1–4 length contexts from the sliding window
    ContextKey ctxs[4];
    int ctx_count = 0;
    for (int len = 1; len <= 4; len++)
    {
        if (!window[len - 1])
            break;
        ContextKey* c = &ctxs[ctx_count++];
        c->order = len;
        for (int i = 0; i < len; i++)
            c->w[i] = window[i];
    }

    for (int i = 0; i < ctx_count; i++)
    {
        ContextKey* key = &ctxs[i];
        size_t idx = hash_context(key) & (g_app.hashtable_size - 1);

        Entry* e = table[idx];
        while (e)
        {
            if (context_equal(&e->key, key))
            {
                // Update existing Next list
                Next* n = e->nexts;
                while (n)
                {
                    if (n->word == next_word)
                    {
                        n->count++;
                        e->total_count++;
                        goto next_context;
                    }
                    n = n->next;
                }

                // Add new Next node
                Next* new_n = alloc_next();
                new_n->word = next_word;
                new_n->count = 1;
                new_n->next = e->nexts;
                e->nexts = new_n;
                e->total_count++;
                goto next_context;
            }
            e = e->next;
        }

        // Entry not found: allocate new Entry and first Next node
        Entry* new_e = alloc_entry();
        new_e->key = *key;
        new_e->total_count = 1;

        Next* new_n = alloc_next();
        new_n->word = next_word;
        new_n->count = 1;
        new_n->next = NULL;

        new_e->nexts = new_n;
        new_e->next = table[idx];
        table[idx] = new_e;

    next_context:;
    }
}

/**
 * Thread worker: process a file chunk and build a local Markov model.
 * Updates thread-local starters, hashtable, and vocabulary.
 */
void* process_chunk(void* arg)
{
    ThreadData* td = (ThreadData*) arg;
    FILE* f = fopen(td->filename, "r");
    if (!f)
    {
        fprintf(stderr, "[Thread %ld] Cannot open %s at byte %ld\n",
                (long) pthread_self(), td->filename, td->start_byte);
        return NULL;
    }

    if (fseek(f, td->start_byte, SEEK_SET) != 0)
    {
        fprintf(stderr, "[Thread %ld] Seek failed to %ld in %s\n",
                (long) pthread_self(), td->start_byte, td->filename);
        fclose(f);
        return NULL;
    }

    char line[MAX_LINE_LEN];
    const char* words[256];
    const char* window[4] = {NULL};

    char* seen = calloc(g_app.intern_table_size, 1);
    if (!seen)
    {
        fprintf(stderr,
                "[Thread %ld] Failed to allocate seen array (%zu bytes)\n",
                (long) pthread_self(), g_app.intern_table_size);
        fclose(f);
        return NULL;
    }

    long current_pos = td->start_byte;
    long last_reported = 0;

    while (fgets(line, sizeof(line), f))
    {
        atomic_fetch_add(&g_app.total_lines_processed, 1);

        size_t line_len = strlen(line);
        current_pos += line_len + 1;
        if (current_pos > td->end_byte)
            break;

        line[strcspn(line, "\n")] = '\0';
        int nw = split_into_words(line, words, 256);

        for (int j = 0; j < nw; j++)
        {
            const char* curr = words[j];

            // Track unique vocabulary words
            unsigned long vh = fnv64_hash(curr) % g_app.intern_table_size;
            if (!seen[vh])
            {
                seen[vh] = 1;
                td->local_vocab++;
            }

            // Determine sentence start
            bool sent_start =
                (j == 0) ||
                (window[0] && strchr(".!?", window[0][strlen(window[0]) - 1]));
            if (sent_start)
            {
                add_starter_local(td->local_starters, &td->local_total_starters,
                                  curr);
            }

            // Add Markov transition using thread-local arena
            add_transition_local_thread(td->local_hashtable, window, curr,
                                        &td->local_arena);

            // Slide the window
            window[3] = window[2];
            window[2] = window[1];
            window[1] = window[0];
            window[0] = curr;
        }

        // Progress report
        if (g_app.verbose >= 1)
        {
            long global_processed = atomic_load(&g_app.total_lines_processed);
            if (global_processed - last_reported >= 10000)
            {
                last_reported = global_processed;
                double percent =
                    (double) global_processed / g_app.total_lines * 100.0;
                if (percent > 100.0)
                    percent = 100.0;
                printf("\r\033[K%.1f%%", percent);
                fflush(stdout);
            }
        }
    }

    fclose(f);
    free(seen);
    return NULL;
}

/**
 * Build Markov model from input file using multiple threads.
 * - Each thread parses a chunk of the file into a local hashtable and starter
 * table.
 * - After parsing, all local tables are merged into global tables.
 */
int build_model_mt(void)
{
    const char* filename = g_app.input_file;
    if (!filename || !*filename)
    {
        fprintf(stderr, "Error: No input file specified\n");
        return -1;
    }

    // --- Count total lines ---
    g_app.total_lines = get_lines_nb(filename);
    if (g_app.total_lines == 0)
    {
        fprintf(stderr, "Input file is empty or unreadable\n");
        return -1;
    }

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    // --- Determine file size ---
    FILE* f = fopen(filename, "r");
    if (!f)
    {
        perror("Cannot open input file");
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    if (file_size <= 0)
    {
        fprintf(stderr, "File is empty or cannot get size\n");
        fclose(f);
        return -1;
    }
    fclose(f);

    // --- Allocate / clear global hashtable ---
    if (!g_app.hashtable)
    {
        g_app.hashtable = calloc(g_app.hashtable_size, sizeof(Entry*));
        if (!g_app.hashtable)
        {
            perror("calloc failed");
            return -1;
        }
    }
    else
    {
        memset(g_app.hashtable, 0, g_app.hashtable_size * sizeof(Entry*));
    }

    // --- Allocate / clear starter table ---
    if (!g_app.starter_table)
    {
        g_app.starter_table =
            calloc(g_app.starter_table_size, sizeof(Starter*));
        if (!g_app.starter_table)
        {
            free(g_app.hashtable);
            g_app.hashtable = NULL;
            perror("calloc failed for starter_table");
            return -1;
        }
    }
    else
    {
        memset(g_app.starter_table, 0,
               g_app.starter_table_size * sizeof(Starter*));
    }

    g_app.vocab_size = 0;
    g_app.total_starters = 0;

    // --- Initialize global arena ---
    g_app.arena_head = g_app.arena_current = NULL;
    arena_init();

    // --- Spawn parsing threads ---
    pthread_t threads[NUM_THREADS];
    ThreadData td[NUM_THREADS];
    long bytes_per_thread = file_size / NUM_THREADS;

    for (int t = 0; t < NUM_THREADS; t++)
    {
        td[t].filename = filename;
        td[t].start_byte = t * bytes_per_thread;
        td[t].end_byte =
            (t == NUM_THREADS - 1) ? file_size : (t + 1) * bytes_per_thread;
        td[t].local_hashtable = calloc(g_app.hashtable_size, sizeof(Entry*));
        td[t].local_starters =
            calloc(g_app.starter_table_size, sizeof(Starter*));
        td[t].local_total_starters = 0;
        td[t].local_vocab = 0;
        td[t].local_arena = calloc(1, sizeof(ArenaBlock));

        if (!td[t].local_hashtable || !td[t].local_starters ||
            !td[t].local_arena)
        {
            fprintf(stderr, "Thread %d allocation failed\n", t);
            for (int i = 0; i <= t; i++)
            {
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

    for (int t = 0; t < NUM_THREADS; t++)
        pthread_join(threads[t], NULL);

    long final_processed = atomic_load(&g_app.total_lines_processed);
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double read_parse_time = (ts_end.tv_sec - ts_start.tv_sec) +
                             (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;

    if (g_app.verbose >= 1)
    {
        printf("\nReading & parsing finished:\n");
        printf("  - Total lines processed: %ld\n", final_processed);
        printf("  - Time: %.2f seconds\n\n", read_parse_time);
        fflush(stdout);
    }

    // --- Multi-threaded merge of hashtables ---
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    pthread_t mthreads[NUM_THREADS];

    typedef struct
    {
        Entry** local;
        Entry** global;
        size_t start_bucket;
        size_t end_bucket;
    } MergeRangeArgs;

    MergeRangeArgs args[NUM_THREADS];
    size_t buckets_per_thread = g_app.hashtable_size / NUM_THREADS;

    for (int i = 0; i < NUM_THREADS; i++)
    {
        args[i].local = td[i].local_hashtable;
        args[i].global = g_app.hashtable;
        args[i].start_bucket = i * buckets_per_thread;
        args[i].end_bucket = (i == NUM_THREADS - 1)
                                 ? g_app.hashtable_size
                                 : (i + 1) * buckets_per_thread;
        pthread_create(&mthreads[i], NULL, merge_local_into_global_range,
                       &args[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(mthreads[i], NULL);

    // --- Merge starters and thread arenas sequentially ---
    for (int t = 0; t < NUM_THREADS; t++)
    {
        merge_starters(td[t].local_starters, g_app.starter_table);
        g_app.total_starters += td[t].local_total_starters;
        g_app.vocab_size += td[t].local_vocab;

        // --- Merge thread-local arenas into global arena ---
        ArenaBlock* b = td[t].local_arena;
        while (b)
        {
            ArenaBlock* next_b = b->next;
            b->next = NULL;
            if (!g_app.arena_head)
            {
                g_app.arena_head = g_app.arena_current = b;
            }
            else
            {
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

    if (g_app.verbose >= 1)
    {
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

/**
 * xxHash64: fast, non-cryptographic 64-bit hash
 * - Lightweight, good for short strings
 * - TODO: Replace FNV with this for performance if needed
 */
static inline uint64_t xxhash64(const char* str)
{
    if (!str)
        return 0;

    uint64_t h = 0x517cc1b727220a95ULL;

    while (*str)
    {
        h ^= (uint64_t) (unsigned char) *str++;
        h *= 0x5bd1e995ULL;
        h ^= h >> 24;
    }

    return h;
}

/**
 * FNV-1a 64-bit hash with finalizer
 * - Safe, widely used
 * - Good distribution for short and medium strings
 */
static inline uint64_t fnv64_hash(const char* str)
{
    if (!str || !*str)
        return 0;

    uint64_t h = 0xcbf29ce484222325ULL; // FNV-1a offset basis

    while (*str)
    {
        h ^= (unsigned char) *str++;
        h *= 0x100000001b3ULL; // FNV-1a prime
    }

    // Finalization improves avalanche / distribution
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;

    return h;
}

/**
 * Hash a ContextKey struct
 * - Uses pointer values of interned strings for speed
 * - Not suitable for un-interned or mutable strings
 */
static inline uint64_t hash_context(const ContextKey* ctx)
{
    if (!ctx || ctx->order <= 0)
        return 0;

    uint64_t h = 0xcbf29ce484222325ULL;

    for (int i = 0; i < ctx->order; i++)
    {
        uintptr_t p = (uintptr_t) ctx->w[i];
        h ^= p;
        h *= 0x100000001b3ULL;
    }

    return h;
}

// ------------------------------------------------------------------------
// Context / String Utilities
// ------------------------------------------------------------------------

/**
 * Compare two ContextKey structs for equality
 * - Returns true if both order and word pointers match
 */
static inline bool context_equal(const ContextKey* a, const ContextKey* b)
{
    if (!a || !b)
        return false;
    if (a->order != b->order)
        return false;

    for (int i = 0; i < a->order; i++)
    {
        if (a->w[i] != b->w[i])
            return false;
    }

    return true;
}

/**
 * Intern a string and return a persistent pointer.
 * - Strings are stored in arena memory
 * - Duplicate strings are not reallocated
 * - Thread-safe via intern_mutex
 */
static inline const char* intern(const char* s)
{
    if (!s || !*s)
        return "";

    size_t len = strlen(s);
    uint64_t h = fnv64_hash(s) % g_app.intern_table_size;

    pthread_mutex_lock(&intern_mutex);

    // --- Lookup existing interned string ---
    InternEntry* e = g_app.intern_table[h];
    while (e)
    {
        if (strcmp(e->str, s) == 0)
        {
            pthread_mutex_unlock(&intern_mutex);
            return e->str;
        }
        e = e->next;
    }

    // --- Allocate and insert new interned string ---
    char* copy = arena_alloc(len + 1);
    memcpy(copy, s, len);
    copy[len] = '\0';

    InternEntry* new_e = malloc(sizeof(InternEntry));
    if (!new_e)
    {
        pthread_mutex_unlock(&intern_mutex);
        fprintf(stderr, "[intern] malloc failed\n");
        exit(1);
    }

    new_e->str = copy;
    new_e->next = g_app.intern_table[h];
    g_app.intern_table[h] = new_e;

    pthread_mutex_unlock(&intern_mutex);
    return copy;
}

/**
 * Load synonym groups and build the keyword synonym pool.
 * - Always includes original keywords
 * - Only synonym groups containing at least one keyword are used
 * - All strings are interned
 * - Duplicate entries are ignored
 *
 * @return 0 on success, -1 on file error, -2 if no synonyms loaded
 */
int load_synonyms(void)
{
    const char* filename = g_app.syn_file ? g_app.syn_file : "synonyms.txt";

    FILE* f = fopen(filename, "r");
    if (!f)
    {
        fprintf(stderr, "Cannot open synonyms file: %s\n", filename);
        return -1;
    }

    g_app.keyword_syn_pool_count = 0;

    /* Add original keywords to pool */
    for (int i = 0; i < g_app.keyword_count; i++)
    {
        const char* kw = g_app.keywords[i];
        if (!kw)
            continue;

        bool dup = false;
        for (int j = 0; j < g_app.keyword_syn_pool_count; j++)
        {
            if (g_app.keyword_syn_pool[j] == kw)
            {
                dup = true;
                break;
            }
        }

        if (!dup && g_app.keyword_syn_pool_count < 2048)
        {
            g_app.keyword_syn_pool[g_app.keyword_syn_pool_count++] = kw;
        }
    }

    char line[8192];

    while (fgets(line, sizeof(line), f))
    {
        line[strcspn(line, "\n")] = '\0';
        if (strlen(line) < 3)
            continue;

        char* words[128];
        int word_count = 0;

        /* Split comma-separated synonym group */
        char* token = strtok(line, ",");
        while (token && word_count < 128)
        {
            while (isspace((unsigned char) *token))
                token++;
            char* end = token + strlen(token) - 1;
            while (end >= token && isspace((unsigned char) *end))
                *end-- = '\0';

            if (*token)
                words[word_count++] = token;
            token = strtok(NULL, ",");
        }

        if (word_count == 0)
            continue;

        /* Check if group contains any original keyword */
        bool relevant = false;
        for (int w = 0; w < word_count && !relevant; w++)
        {
            const char* candidate = intern(words[w]);
            for (int k = 0; k < g_app.keyword_count; k++)
            {
                if (g_app.keywords[k] == candidate)
                {
                    relevant = true;
                    break;
                }
            }
        }
        if (!relevant)
            continue;

        /* Add synonyms to global pool */
        for (int w = 0; w < word_count; w++)
        {
            const char* syn = intern(words[w]);

            bool dup = false;
            for (int j = 0; j < g_app.keyword_syn_pool_count; j++)
            {
                if (g_app.keyword_syn_pool[j] == syn)
                {
                    dup = true;
                    break;
                }
            }

            if (!dup && g_app.keyword_syn_pool_count < 2048)
            {
                g_app.keyword_syn_pool[g_app.keyword_syn_pool_count++] = syn;
            }
        }
    }

    fclose(f);
    return (g_app.keyword_syn_pool_count > 0) ? 0 : -2;
}

/**
 * Check if a token is a single punctuation character.
 * - Recognized punctuation: .,!?;:
 * - Returns true if tok is non-NULL and a single-character punctuation
 */
static inline bool is_punct(const char* tok)
{
    return tok && strlen(tok) == 1 && strchr(".,!?;:", tok[0]);
}

/**
 * Split a line into words and punctuation tokens.
 * - Each token is interned and lowercased
 * - Punctuation marks are treated as separate tokens
 * - Returns the number of tokens placed in tokens_out (up to max_tokens)
 *
 * @param line        Input line to tokenize (modified in-place)
 * @param tokens_out  Output array of interned token pointers
 * @param max_tokens  Maximum number of tokens to store
 * @return            Number of tokens written to tokens_out
 */
int split_into_words(char* line, const char** tokens_out, int max_tokens)
{
    if (!line || !tokens_out || max_tokens <= 0)
        return 0;

    int count = 0;
    char* p = line;

    while (*p && count < max_tokens)
    {
        /* Skip leading whitespace */
        while (*p && isspace((unsigned char) *p))
            p++;
        if (!*p)
            break;

        /* Single punctuation as token */
        if (strchr(".,!?;:", *p))
        {
            char tmp[2] = {*p, '\0'};
            tokens_out[count++] = intern(tmp);
            p++;
            continue;
        }

        /* Regular word */
        char* start = p;
        int len = 0;

        while (*p && !isspace((unsigned char) *p) && !strchr(".,!?;:", *p))
        {
            len++;
            p++;
        }

        /* Trim punctuation from start/end, keep apostrophes */
        while (len > 0 && ispunct((unsigned char) *start) && *start != '\'')
        {
            start++;
            len--;
        }
        while (len > 0 && ispunct((unsigned char) *(start + len - 1)) &&
               *(start + len - 1) != '\'')
            len--;

        if (len <= 0)
            continue;
        if (len >= WORD_MAX_LEN)
            len = WORD_MAX_LEN - 1;

        char tmp[WORD_MAX_LEN];
        strncpy(tmp, start, len);
        tmp[len] = '\0';

        /* Convert to lowercase */
        for (int i = 0; i < len; i++)
            tmp[i] = tolower((unsigned char) tmp[i]);

        tokens_out[count++] = intern(tmp);
    }

    return count;
}

/**
 * Compute perplexity of a sequence of words using interpolated n-grams.
 * - Uses Lidstone smoothing (alpha = 0.01)
 * - Interpolates up to 4-grams with fixed weights
 * - Backoff to lower orders if high-order context is missing
 * - Words are interned pointers
 *
 * @param words Array of interned word pointers
 * @param nw    Number of words in the sequence
 * @return      Perplexity score (large number if invalid or unseen)
 */
static double compute_perplexity(const char** words, int nw)
{
    if (!words || nw < 2)
        return 1e12;
    if (g_app.vocab_size <= 0)
        return 1e12;

    double log_sum = 0.0;
    int valid_count = 0;

    /* Interpolation weights (order 0 → 4) */
    static const double lambdas[5] = {0.0, 0.15, 0.25, 0.35, 0.25};

    /* Lidstone smoothing constant */
    const double alpha = 0.01;

    for (int pos = 1; pos < nw; pos++)
    {
        int ctx_len = (pos < 4) ? pos : 4; /* max 4-gram */
        double prob = 0.0;
        double remaining = 1.0;

        /* Backoff from high-order → low-order n-grams */
        for (int ord = ctx_len; ord >= 1; ord--)
        {
            if (remaining < 1e-9)
                break;

            /* Build context key */
            ContextKey ctx = {0};
            ctx.order = (uint8_t) ord;
            for (int k = 0; k < ord; k++)
                ctx.w[k] = words[pos - 1 - k];

            /* Hash lookup */
            uint64_t h = hash_context(&ctx);
            size_t bucket = h & (g_app.hashtable_size - 1);

            long observed = 0;
            long match_count = 0;

            /* Search bucket */
            for (Entry* e = g_app.hashtable[bucket]; e; e = e->next)
            {
                if (context_equal(&e->key, &ctx))
                {
                    observed = e->total_count;
                    for (Next* n = e->nexts; n; n = n->next)
                    {
                        if (n->word == words[pos])
                        {
                            match_count = n->count;
                            break;
                        }
                    }
                    break;
                }
            }

            /* Lidstone probability */
            double denom = observed + alpha * g_app.vocab_size;
            if (denom <= 0)
                denom = 1.0;
            double p =
                (observed > 0) ? (match_count + alpha) / denom : alpha / denom;

            /* Weighted interpolation */
            double w = lambdas[ord] * remaining;
            prob += w * p;
            remaining -= w;
        }

        /* Fallback uniform probability for remaining mass */
        double uniform_p = 1.0 / (g_app.vocab_size + 10000.0);
        prob += remaining * uniform_p;

        if (prob > 0.0)
        {
            log_sum += log(prob);
            valid_count++;
        }
    }

    if (valid_count == 0)
        return 1e12;

    double avg_log_p = log_sum / (double) valid_count;
    double perplexity = exp(-avg_log_p);

    /* Safety: prevent inf/NaN */
    if (!isfinite(perplexity) || perplexity < 1e-9)
        return 1e12;

    return perplexity;
}

/**
 * Check if a word is related to the keyword pool.
 * - Comparison is done via pointer equality (all keywords are interned)
 *
 * @param w  Word to check
 * @return   true if the word exists in keyword_syn_pool, false otherwise
 */
static inline bool is_keyword_related(const char* w)
{
    if (!w)
        return false;

    for (int i = 0; i < g_app.keyword_syn_pool_count; i++)
    {
        if (g_app.keyword_syn_pool[i] == w)
            return true;
    }

    return false;
}

/**
 * Select a starter word for sentence generation.
 * - Prefers thematic starters if available (weighted random)
 * - Falls back to global starter table if no thematic starter is selected
 * - Ultimate fallback returns a default word ("hello")
 *
 * @return  Pointer to interned starter word
 */
static const char* select_thematic_starter(void)
{
    /* --- Prefer thematic starters --- */
    if (g_app.thematic_starter_count > 0 && g_app.total_thematic_weight > 0)
    {
        long r = rand() % g_app.total_thematic_weight;
        long sum = 0;

        for (int i = 0; i < g_app.thematic_starter_count; i++)
        {
            sum += g_app.thematic_starters[i].count;
            if (r < sum)
                return g_app.thematic_starters[i].word;
        }
    }

    /* --- Fallback: full starter table weighted selection --- */
    if (g_app.total_starters > 0)
    {
        long r = rand() % g_app.total_starters;
        long sum = 0;

        for (size_t i = 0; i < g_app.starter_table_size; i++)
        {
            for (Starter* s = g_app.starter_table[i]; s; s = s->next)
            {
                sum += s->count;
                if (r < sum)
                    return s->word;
            }
        }
    }

    /* --- Ultimate fallback --- */
    return intern("hello");
}

/**
 * Generate a sequence of words using the Markov model.
 * - Respects user-provided prefix if available
 * - Performs 4→1-gram backoff for candidate selection
 * - Biases toward thematic / keyword-related words
 * - Applies temperature scaling, repetition penalty, and top-k sampling
 * - Ensures sentence capitalization and proper ending punctuation
 *
 * @param out_words  Pre-allocated array to receive generated interned words
 * @return           Number of words generated
 */
int generate_sequence(const char** out_words)
{
    int max_tokens = MAX_WORDS_PER_GENERATION;
    int generated = 0;

    // ----------------------------
    // Context / Starter selection
    // ----------------------------
    if (g_app.prefix_len > 0 && g_app.prefix_words)
    {
        // Push user-provided prefix directly
        for (int i = 0; i < g_app.prefix_len && generated < max_tokens; i++)
            out_words[generated++] = g_app.prefix_words[i];
    }
    else
    {
        // Default: select a thematic starter
        const char* starter = select_thematic_starter();
        if (starter && generated < max_tokens)
            out_words[generated++] = starter;
    }

    // ----------------------------
    // Main generation loop
    // ----------------------------
    while (generated < max_tokens)
    {
        const char* candidates[4] = {NULL};
        int best_syn_slot = -1;
        bool has_syn_candidate = false;

        // --- Collect candidates from 4→1-grams ---
        for (int order = 4; order >= 1; order--)
        {
            if (generated < order)
                continue;

            ContextKey ctx = {0};
            ctx.order = (uint8_t) order;
            for (int k = 0; k < order; k++)
                ctx.w[k] = out_words[generated - 1 - k];

            uint64_t h = hash_context(&ctx);
            size_t idx = h & (g_app.hashtable_size - 1);

            for (Entry* e = g_app.hashtable[idx]; e; e = e->next)
            {
                if (!context_equal(&e->key, &ctx))
                    continue;

                const char* best_next = NULL;
                long max_count = -1;

                for (Next* n = e->nexts; n; n = n->next)
                {
                    if (n->count > max_count)
                    {
                        max_count = n->count;
                        best_next = n->word;
                    }
                }

                if (best_next)
                {
                    int slot = 4 - order;
                    candidates[slot] = best_next;

                    if (is_keyword_related(best_next))
                    {
                        has_syn_candidate = true;
                        if (best_syn_slot == -1 || slot < best_syn_slot)
                            best_syn_slot = slot;
                    }
                }
                break; // Stop after first matching context
            }
        }

        // ----------------------------
        // Decide next word
        // ----------------------------
        const char* next = NULL;

        // --- Priority 1: synonym bias ---
        if (has_syn_candidate && best_syn_slot != -1 && (rand() % 100 < 70))
            next = candidates[best_syn_slot];

        // --- Priority 2: temperature + top-k sampling ---
        if (!next)
        {
            typedef struct
            {
                const char* word;
                double weight;
            } ScoredCand;
            ScoredCand scored[4];
            double temp_weights[4];
            int num_valid = 0;
            static const double base_weights[4] = {0.9, 0.7, 0.4, 0.2};

            for (int slot = 0; slot < 4; slot++)
            {
                if (!candidates[slot])
                    continue;

                double w = base_weights[slot];
                for (int k = 1; k <= g_app.rep_window && generated - k >= 0;
                     k++)
                {
                    if (out_words[generated - k] == candidates[slot])
                    {
                        w *= g_app.rep_penalty;
                        break;
                    }
                }

                scored[num_valid++] = (ScoredCand) {candidates[slot], w};
            }

            if (num_valid > 0)
            {
                double sum = 0.0;
                for (int i = 0; i < num_valid; i++)
                {
                    double p = scored[i].weight;
                    if (p <= 0)
                        p = 1e-10;
                    temp_weights[i] = pow(p, 1.0 / g_app.temperature);
                    sum += temp_weights[i];
                }
                for (int i = 0; i < num_valid; i++)
                    temp_weights[i] /= sum;

                int sample_from = num_valid;
                if (g_app.top_k > 0 && g_app.top_k < num_valid)
                    sample_from = g_app.top_k;

                double r = (double) rand() / RAND_MAX;
                double cum = 0.0;
                for (int i = 0; i < sample_from; i++)
                {
                    cum += temp_weights[i];
                    if (r <= cum)
                    {
                        next = scored[i].word;
                        break;
                    }
                }
            }
        }

        // --- Priority 3: fallback to period ---
        if (!next)
            next = intern(".");

        out_words[generated++] = next;

        // --- Stop conditions ---
        if (generated >= g_app.max_gen_len)
            break;
        if (generated >= g_app.min_gen_len &&
            (strcmp(next, ".") == 0 || strcmp(next, "!") == 0 ||
             strcmp(next, "?") == 0))
            break;
    }

    // ----------------------------
    // Post-processing: capitalization & final punctuation
    // ----------------------------
    if (generated == 0)
        return 0;

    bool capitalize = true;
    for (int i = 0; i < generated; i++)
    {
        if (capitalize && isalpha((unsigned char) out_words[i][0]))
        {
            char buf[WORD_MAX_LEN];
            strncpy(buf, out_words[i], sizeof(buf) - 1);
            buf[sizeof(buf) - 1] = '\0';
            buf[0] = toupper((unsigned char) buf[0]);
            out_words[i] = intern(buf);
            capitalize = false;
        }
        if (strcmp(out_words[i], ".") == 0)
            capitalize = true;
    }

    if (strcmp(out_words[generated - 1], ".") != 0)
        out_words[generated++] = intern(".");

    return generated;
}

/**
 * Print an array of interned words to the given FILE stream
 * - Automatically handles spacing, capitalization, and punctuation
 * - First word is always capitalized if alphabetical
 * - No spaces added before closing punctuation or after opening punctuation
 *
 * @param out     FILE stream to print to
 * @param words   Array of interned words
 * @param count   Number of words in the array
 */
static void print_words_properly(FILE* out, const char** words, int count)
{
    for (int i = 0; i < count; i++)
    {
        const char* w = words[i];
        if (!w)
            continue;

        // --- Capitalize first word if alphabetical ---
        if (i == 0)
        {
            if (isalpha((unsigned char) w[0]))
            {
                fputc(toupper((unsigned char) w[0]), out);
                fputs(w + 1, out);
            }
            else
            {
                fputs(w, out);
            }
            continue;
        }

        const char* prev = words[i - 1];

        // --- Determine if space is needed ---
        bool space_before = true;

        // No space after opening punctuation
        if (prev && strchr("([{‘“\"'", prev[0]))
            space_before = false;

        // No space before closing or sentence-ending punctuation
        if (strchr(".,!?;:)]’”\"'", w[0]))
            space_before = false;

        if (space_before)
            fputc(' ', out);
        fputs(w, out);
    }
}

/**
 * Free all allocated memory in the Markov model
 * - Safely releases candidates, keywords, starter table, intern table, arena
 * blocks, and hashtable
 * - Resets all model stats and freelists
 * - Must be called before program exit or re-loading a new model
 */
void free_model(void)
{
    printf("[free_model] Starting cleanup...\n");

    // ----------------------------
    // 1. Free candidate arrays
    // ----------------------------
    printf("[free_model] Freeing candidates (%d)...\n", g_app.candidate_count);
    for (int i = 0; i < g_app.candidate_count; i++)
    {
        if (g_app.candidates[i].words)
        {
            free((void*) g_app.candidates[i].words);
            g_app.candidates[i].words = NULL;
        }
    }
    g_app.candidate_count = 0;
    printf("[free_model] Candidates freed.\n");

    // ----------------------------
    // 2. Free keyword & thematic starter arrays
    // ----------------------------
    printf("[free_model] Freeing keywords & thematic starters...\n");
    if (g_app.keywords)
    {
        free(g_app.keywords);
        g_app.keywords = NULL;
    }
    if (g_app.thematic_starters)
    {
        free(g_app.thematic_starters);
        g_app.thematic_starters = NULL;
    }
    printf("[free_model] Keywords & thematic starters freed.\n");

    // ----------------------------
    // 3. Free starter table
    // ----------------------------
    if (g_app.starter_table)
    {
        printf("[free_model] Freeing starter_table (%zu buckets)...\n",
               g_app.starter_table_size);
        for (size_t i = 0; i < g_app.starter_table_size; i++)
        {
            Starter* s = g_app.starter_table[i];
            while (s)
            {
                Starter* next_s = s->next;
                free(s);
                s = next_s;
            }
        }
        free(g_app.starter_table);
        g_app.starter_table = NULL;
        printf("[free_model] Starter table freed.\n");
    }

    // ----------------------------
    // 4. Free intern table
    // ----------------------------
    if (g_app.intern_table)
    {
        printf("[free_model] Freeing intern_table (%zu buckets)...\n",
               g_app.intern_table_size);
        for (size_t i = 0; i < g_app.intern_table_size; i++)
        {
            InternEntry* e = g_app.intern_table[i];
            while (e)
            {
                InternEntry* next_e = e->next;
                free(e);
                e = next_e;
            }
        }
        free(g_app.intern_table);
        g_app.intern_table = NULL;
        printf("[free_model] Intern table freed.\n");
    }

    // ----------------------------
    // 5. Free arena blocks
    // ----------------------------
    if (g_app.arena_head)
    {
        printf("[free_model] Freeing arena (%zu bytes used, %zu blocks)...\n",
               g_app.arena_total_bytes_used, g_app.arena_block_count);

        ArenaBlock* block = g_app.arena_head;
        size_t block_count = 0;
        while (block)
        {
            ArenaBlock* next = block->next;
            free(block);
            block = next;
            block_count++;
            if (block_count % 10 == 0)
            {
                printf("[free_model] Freed %zu arena blocks...\n", block_count);
            }
        }

        g_app.arena_head = NULL;
        g_app.arena_current = NULL;
        printf("[free_model] Arena fully freed (%zu blocks).\n", block_count);
    }

    // ----------------------------
    // 6. Free hashtable array (entries live in arena)
    // ----------------------------
    if (g_app.hashtable)
    {
        printf("[free_model] Freeing hashtable array pointer only...\n");
        free(g_app.hashtable);
        g_app.hashtable = NULL;
        printf("[free_model] Hashtable array freed.\n");
    }

    // ----------------------------
    // 7. Reset model stats & freelists
    // ----------------------------
    printf("[free_model] Resetting stats & freelists...\n");
    g_app.vocab_size = 0;
    g_app.total_starters = 0;
    g_app.entry_freelist = NULL;
    g_app.next_freelist = NULL;
    g_app.best_candidate_index = -1;
    g_app.best_score = 0.0;
    g_app.generation_attempts = 0;

    printf("[free_model] Cleanup finished successfully.\n");
}

/**
 * Parse command-line arguments
 * - Required: input file (corpus)
 * - Optional: --verbose N, --seed N
 *
 * @param argc  Argument count
 * @param argv  Argument vector
 * @return 0 on success, 1 on failure
 */
static int parse_arguments(int argc, char** argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <corpus.txt> [--verbose N] [--seed N]\n",
                argv[0]);
        return 1;
    }

    g_app.input_file = argv[1];

    // ----------------------------
    // Optional flags
    // ----------------------------
    for (int i = 2; i < argc; i++)
    {
        if (strcmp(argv[i], "--verbose") == 0 && i + 1 < argc)
        {
            g_app.verbose = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
        {
            g_app.seed = (unsigned) atoi(argv[++i]);
            srand(g_app.seed);
        }
        else
        {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
    }

    return 0;
}

/**
 * Main program entry point
 * - Initializes application
 * - Parses arguments
 * - Builds Markov model (multi-threaded)
 * - Handles interactive user input loop for:
 *     • Keywords
 *     • Synonym pool
 *     • Min/max generation length
 *     • Optional starting context
 * - Generates text candidates and prints the best
 * - Frees all allocated resources before exit
 */
int main(int argc, char** argv)
{
    // ----------------------------
    // 1. Initialize application
    // ----------------------------
    if (app_init() != 0)
    {
        fprintf(stderr, "Fatal: initialization failed\n");
        return 1;
    }

    // ----------------------------
    // 2. Parse CLI arguments
    // ----------------------------
    if (parse_arguments(argc, argv) != 0)
        goto shutdown;

    // ----------------------------
    // 3. Build Markov model
    // ----------------------------
    if (build_model_mt() != 0)
    {
        fprintf(stderr, "Fatal: model build failed\n");
        goto shutdown;
    }

    if (g_app.vocab_size == 0)
        fprintf(stderr,
                "CRITICAL: vocab_size is 0 → may cause div-by-zero errors\n");

    if (g_app.total_starters == 0)
        fprintf(stderr,
                "WARNING: no starter words detected → fallback to 'hello'\n");

    // ----------------------------
    // 4. Interactive input loop
    // ----------------------------
    char keyword_input[1024];
    char context_input[1024];
    char len_input[256];
    static const char* parsed_context[256];

    while (1)
    {
        // --- 4a. Ask for keywords ---
        printf("\nEnter keywords (comma separated, empty = keep previous): ");
        fflush(stdout);
        if (!fgets(keyword_input, sizeof(keyword_input), stdin))
            break;
        keyword_input[strcspn(keyword_input, "\n")] = '\0';

        if (strcmp(keyword_input, "quit") == 0 ||
            strcmp(keyword_input, "exit") == 0)
            break;

        // --- Reset keyword-related state ---
        g_app.keyword_count = 0;
        g_app.keyword_syn_pool_count = 0;
        memset(g_app.keyword_syn_pool, 0, sizeof(g_app.keyword_syn_pool));
        if (g_app.thematic_starters)
        {
            free(g_app.thematic_starters);
            g_app.thematic_starters = NULL;
        }
        g_app.thematic_starter_count = 0;
        g_app.total_thematic_weight = 0;

        // --- Parse comma-separated keywords ---
        if (strlen(keyword_input) > 0)
        {
            char* tok = strtok(keyword_input, ",");
            while (tok && g_app.keyword_count < MAX_KEYWORDS)
            {
                while (isspace((unsigned char) *tok))
                    tok++;
                char* end = tok + strlen(tok) - 1;
                while (end >= tok && isspace((unsigned char) *end))
                    *end-- = '\0';
                if (*tok)
                    g_app.keywords[g_app.keyword_count++] = intern(tok);
                tok = strtok(NULL, ",");
            }
        }

        // --- Load synonym pool ---
        load_synonyms();

        // --- 4b. Ask for min/max generation length ---
        printf("Enter min_length (default %d, empty = keep current): ",
               g_app.min_gen_len);
        fflush(stdout);
        if (fgets(len_input, sizeof(len_input), stdin))
        {
            len_input[strcspn(len_input, "\n")] = '\0';
            if (strlen(len_input) > 0)
            {
                int new_min = atoi(len_input);
                if (new_min > 0 && new_min < 200)
                    g_app.min_gen_len = new_min;
            }
        }

        printf("Enter max_length (default %d, empty = keep current): ",
               g_app.max_gen_len);
        fflush(stdout);
        if (fgets(len_input, sizeof(len_input), stdin))
        {
            len_input[strcspn(len_input, "\n")] = '\0';
            if (strlen(len_input) > 0)
            {
                int new_max = atoi(len_input);
                if (new_max > g_app.min_gen_len && new_max < 1000)
                    g_app.max_gen_len = new_max;
            }
        }

        // --- 4c. Ask for starting context ---
        printf("Enter starting context (or press Enter to use thematic "
               "starter): ");
        fflush(stdout);
        if (!fgets(context_input, sizeof(context_input), stdin))
            context_input[0] = '\0';
        context_input[strcspn(context_input, "\n")] = '\0';

        g_app.prefix_len = 0;
        g_app.prefix_words = NULL;

        if (strlen(context_input) > 0)
        {
            char ctx_copy[1024];
            strncpy(ctx_copy, context_input, sizeof(ctx_copy) - 1);
            ctx_copy[sizeof(ctx_copy) - 1] = '\0';
            int nw = split_into_words(ctx_copy, parsed_context, 256);
            if (nw > 0)
            {
                g_app.prefix_words = parsed_context;
                g_app.prefix_len = nw;
            }
        }

        // --- 4d. Precompute thematic starters ---
        precompute_thematic_starters();

        // --- 4e. Generate text candidates and print ---
        generate_multiple_candidates();
        select_and_print_best(5);

        // --- 4f. Display global debug info ---
        display_global_debug();
    }

shutdown:
    // ----------------------------
    // 5. Cleanup
    // ----------------------------
    free_model();
    return 0;
}

/*
 * Score a candidate sequence based on theme, repetition, perplexity, and length
 * Lower score = better candidate
 */
static double score_candidate(const char** words, int nw)
{
    if (nw < g_app.min_gen_len)
        return 1e15;

    int theme_count = 0, exact_keyword_count = 0;
    for (int i = 0; i < nw; i++)
    {
        if (is_keyword_related(words[i]))
        {
            theme_count++;
            for (int k = 0; k < g_app.keyword_count; k++)
            {
                if (words[i] == g_app.keywords[k])
                {
                    exact_keyword_count++;
                    break;
                }
            }
        }
    }
    double theme_ratio = (double) theme_count / nw;
    double theme_score = 8.0 * (1.0 - theme_ratio);

    double exact_bonus = exact_keyword_count * 0.5;

    int local_rep = 0;
    for (int j = 2; j < nw; j++)
        if (words[j] == words[j - 2] && words[j - 1] == words[j + 1])
            local_rep++;

    int global_rep = 0;
    for (int i = 0; i < nw - 2; i++)
        for (int j = i + 3; j < nw - 2; j++)
            if (words[i] == words[j] && words[i + 1] == words[j + 1] &&
                words[i + 2] == words[j + 2])
                global_rep++;

    double rep_penalty = 1.0 + 1.0 * local_rep + 2.0 * global_rep;

    double ppl = compute_perplexity(words, nw);
    double norm_ppl = ppl / 100.0;

    double ideal_len = (g_app.min_gen_len + g_app.max_gen_len) / 2.0;
    double len_diff = fabs(nw - ideal_len);
    double len_reward = 1.0 + 0.8 * exp(-len_diff / 5.0);

    double end_bonus = 0.0;
    if (nw > 0 &&
        (strcmp(words[nw - 1], ".") == 0 || strcmp(words[nw - 1], "!") == 0 ||
         strcmp(words[nw - 1], "?") == 0))
        end_bonus = 0.7;

    double final_score = (norm_ppl * rep_penalty * theme_score) /
                         (len_reward + end_bonus + exact_bonus + 0.5);

    return final_score;
}

/*
 * Generate multiple candidates and store them in g_app.candidates
 * Applies min/max length, theme density, and repetition checks
 */
static void generate_multiple_candidates(void)
{
    g_app.candidate_count = 0;
    g_app.generation_attempts = 0;
    g_app.average_theme_density = 0.0;
    double theme_density_sum = 0.0;

    const int max_attempts = g_app.max_candidates * 30;

    while (g_app.candidate_count < g_app.max_candidates &&
           g_app.generation_attempts < max_attempts)
    {
        g_app.generation_attempts++;

        const char* words[MAX_WORDS_PER_GENERATION + 16] = {0};
        int nw = generate_sequence(words);

        if (nw < g_app.min_gen_len || nw > g_app.max_gen_len)
            continue;

        int theme_count = 0;
        for (int i = 0; i < nw; i++)
            if (is_keyword_related(words[i]))
                theme_count++;

        double density = (nw > 0) ? (double) theme_count / nw : 0.0;
        if (density < g_app.min_theme_density)
            continue;

        const char** copy = malloc(nw * sizeof(const char*));
        if (!copy)
        {
            fprintf(stderr, "[generate] malloc failed\n");
            continue;
        }
        memcpy(copy, words, nw * sizeof(const char*));

        Candidate* c = &g_app.candidates[g_app.candidate_count];
        c->words = copy;
        c->length = nw;
        c->ppl = compute_perplexity(words, nw);

        g_app.candidate_count++;
        theme_density_sum += density;

        if (g_app.candidate_count >= g_app.max_candidates)
            break;
    }

    if (g_app.candidate_count > 0)
        g_app.average_theme_density = theme_density_sum / g_app.candidate_count;

    if (g_app.candidate_count == 0)
        fprintf(stderr,
                "[generate] No valid candidates after %d attempts "
                "(min_len=%d max_len=%d min_density=%.3f)\n",
                max_attempts, g_app.min_gen_len, g_app.max_gen_len,
                g_app.min_theme_density);
    else
        printf("[generate] Generated %d candidates (%.1f%% success rate)\n",
               g_app.candidate_count,
               100.0 * g_app.candidate_count / g_app.generation_attempts);
}

/*
 * Display all candidates in a readable format using print_words_properly
 */
void display_candidates(void)
{
    printf("\n=== Candidate Sequences (%d total) ===\n\n",
           g_app.candidate_count);

    for (int i = 0; i < g_app.candidate_count; i++)
    {
        Candidate* c = &g_app.candidates[i];
        const char* temp_words[MAX_WORDS_PER_GENERATION + 16];
        int temp_len = c->length;
        for (int j = 0; j < temp_len; j++)
            temp_words[j] = c->words[j];

        printf("Candidate #%d  (cleaned length: %d words)\n", i + 1, temp_len);
        print_words_properly(stdout, temp_words, temp_len);
        printf("\n\n");
    }

    printf("====================================\n\n");
}

/*
 * Check if a given interned string is in the keyword synonym pool
 */
static inline bool is_in_syn_pool(const char* interned_str)
{
    if (!interned_str)
        return false;
    for (int i = 0; i < g_app.keyword_syn_pool_count; i++)
        if (g_app.keyword_syn_pool[i] == interned_str)
            return true;
    return false;
}

/*
 * Precompute thematic starters from keyword synonym pool
 * Populates g_app.thematic_starters and total thematic weight
 */
void precompute_thematic_starters(void)
{
    if (g_app.thematic_starters)
    {
        free(g_app.thematic_starters);
        g_app.thematic_starters = NULL;
    }
    g_app.thematic_starter_count = 0;
    g_app.total_thematic_weight = 0;
    if (g_app.keyword_syn_pool_count == 0 || g_app.total_starters == 0)
        return;

    int count = 0;
    long total_weight = 0;

    for (size_t i = 0; i < g_app.starter_table_size; i++)
        for (Starter* s = g_app.starter_table[i]; s; s = s->next)
            if (is_in_syn_pool(s->word))
            {
                count++;
                if (s->count > 0)
                    total_weight += s->count;
            }

    if (count == 0)
        return;

    g_app.thematic_starters = malloc(count * sizeof(ThematicStarter));
    if (!g_app.thematic_starters)
        return;

    g_app.thematic_starter_count = count;
    g_app.total_thematic_weight = total_weight;

    int idx = 0;
    for (size_t i = 0; i < g_app.starter_table_size && idx < count; i++)
        for (Starter* s = g_app.starter_table[i]; s; s = s->next)
            if (is_in_syn_pool(s->word))
            {
                g_app.thematic_starters[idx].word = s->word;
                g_app.thematic_starters[idx].count = s->count;
                idx++;
            }
}
