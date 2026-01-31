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
#define MAX_CANDIDATES 10
#define HASH_TABLE_SIZE (1 << 21)
#define STARTER_TABLE_SIZE (1 << 17)
#define INTERN_TABLE_SIZE (1 << 20)
#define ARENA_BLOCK_SIZE (1ULL << 24)
#define CONTEXT_BUF_SIZE 1024
#define MAX_KEYWORDS 8
#define MAX_SYNONYMS_PER_WORD 32
#define MAX_SYN_ENTRIES 8192

#define DEFAULT_MIN_TEXT_LENGTH 50
#define DEFAULT_MAX_TEXT_LENGTH 60
#define HASH_TABLE_SIZE (1 << 21)

// TODO Forward declaration mess as to be fixed
typedef struct App App;
static int  app_init(void);
static void print_words_properly(FILE *out, const char **words, int count);

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

ArenaBlock *arena_head = NULL;
ArenaBlock *arena_current = NULL;

void arena_init(void) {
    arena_head = calloc(1, sizeof(ArenaBlock));
    if (!arena_head) {
        fprintf(stderr, "arena_init failed\n");
        exit(1);
    }
    arena_current = arena_head;
}

char *arena_alloc(size_t size) {
    if (arena_current->used + size + 1 > ARENA_BLOCK_SIZE) {
        ArenaBlock *new_block = calloc(1, sizeof(ArenaBlock));
        arena_current->next = new_block;
        arena_current = new_block;
    }
    char *p = arena_current->data + arena_current->used;
    arena_current->used += size + 1;
    return p;
}

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

Next *next_freelist = NULL;

Next *alloc_next(void) {
    if (next_freelist) {
        Next *n = next_freelist;
        next_freelist = n->next;
        memset(n, 0, sizeof(Next));
        return n;
    }
    return calloc(1, sizeof(Next));
}

void free_next(Next *n) {
    if (!n) return;
    n->next = next_freelist;
    next_freelist = n;
}

Entry *entry_freelist = NULL;

Entry *alloc_entry(void) {
    if (entry_freelist) {
        Entry *e = entry_freelist;
        entry_freelist = e->next;
        memset(e, 0, sizeof(Entry));
        return e;
    }
    return calloc(1, sizeof(Entry));
}

void free_entry(Entry *e) {
    if (!e) return;
    e->next = entry_freelist;
    entry_freelist = e;
}

typedef struct App {

    /* config */
    char *input_file;
    char *output_file;
    char *syn_file;
    int min_gen_len;
    int max_gen_len;
    int max_candidates;
    int num_threads;
    unsigned int seed;
    int verbose;
    
    /* filters */
    const char **keywords;
    int keyword_count;
    
    /* model */
    Entry **hashtable;
    long hashtable_size;
    Starter **starter_table;
    long starter_table_size;
    long vocab_size;
    long total_starters;
    long intern_table_size;
    
    /* synonyms */
    SynEntry syn_table[MAX_SYN_ENTRIES];
    int syn_count;
    
    /* Generation candidates */
    Candidate   candidates[MAX_CANDIDATES];
    int         candidate_count;
    int         best_candidate_index;
    
    /* memory */
    ArenaBlock arena;
} App;

static App g_app;

// TODO clean big declaration mess 

void display_global_debug(void);

static const double lambda[5] = {0.0, 0.15, 0.35, 0.60, 0.85};
InternEntry *intern_table[INTERN_TABLE_SIZE] = {0};
Entry *hashtable[HASH_TABLE_SIZE] = {0};
Starter *starter_table[STARTER_TABLE_SIZE] = {0};
SynEntry syn_table[MAX_SYN_ENTRIES];
static int generate_sequence(const char **out_words);
int syn_count = 0;
long total_starters = 0;
long vocab_size = 0;

int MIN_GEN_LEN = 12;
int MAX_GEN_LEN = 28;

atomic_long total_lines_processed = 0;
long total_lines_global = 0;
static inline bool context_equal(const ContextKey *a, const ContextKey *b);
static inline uint64_t hash_context(const ContextKey *c);
static void generate_multiple_candidates(void);
void display_candidates(void);
static double score_candidate(const char **words, int nw);

static inline unsigned long hash_str(const char *str);
const char *intern(const char *s);
int split_into_words(char *line, const char **tokens_out, int max_tokens);
static void select_and_print_best(void);

static void select_and_print_best(void) { // very naive for now
    if (g_app.candidate_count == 0) {
        printf("No valid candidates generated.\n");
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
        printf("No best candidate found.\n");
        return;
    }

    Candidate *winner = &g_app.candidates[best_idx];

    printf("\nBest candidate selected (index %d, length %d, score %.4f):\n\n",
           best_idx + 1, winner->length, best_score);

    print_words_properly(stdout, winner->words, winner->length);
    putchar('\n');
}

void merge_local_into_global(Entry **local, Entry **global) { // TODO check for overflow, put boundaries
    struct TempMapping {
        Entry *local_entry;
        Entry *global_match;
    };

    struct TempMapping *mappings = NULL;
    size_t map_count = 0;
    size_t map_cap = 0;

    for (size_t i = 0; i < g_app.hashtable_size; i++) {
        Entry *e = local[i];
        while (e) {
            Entry *next_e = e->next;

            uint64_t h = hash_context(&e->key);
            size_t bucket = h % g_app.hashtable_size;

            Entry *g = global[bucket];
            Entry *match = NULL;

            while (g) {
                if (context_equal(&g->key, &e->key)) {
                    match = g;
                    break;
                }
                g = g->next;
            }

            if (map_count >= map_cap) {
                map_cap = map_cap ? map_cap * 2 : 65536;
                mappings = realloc(mappings, map_cap * sizeof(*mappings));
                if (!mappings) {
                    perror("realloc failed");
                    exit(1);
                }
            }

            mappings[map_count++] = (struct TempMapping){e, match};

            if (!match) {
                e->next = global[bucket];
                global[bucket] = e;
            }

            e = next_e;
        }
    }

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
                if (gn->word == ln->word) {
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
        free_entry(local_e);
    }

    free(mappings);
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
        atomic_fetch_add(&total_lines_processed, 1);
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

        // Progress reporting every 50k lines (adjust number if too noisy or too quiet)
        if (local_lines % 50000 == 0 && g_app.verbose >= 1) {
            long global_processed = atomic_load(&total_lines_processed);
            fprintf(stderr, "[Thread %ld] Processed %ld lines so far (global: %ld)\n",
                    (long)pthread_self(), local_lines, global_processed);
            fflush(stderr);
        }
    }

    fclose(f);
    free(seen);

    if (g_app.verbose >= 1) {
        fprintf(stderr, "[Thread %ld] Finished — processed %ld lines\n",
                (long)pthread_self(), local_lines);
    }

    return NULL;
}

int build_model_mt(void){

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
    arena_head    = NULL;
    arena_current = NULL;
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
    
	long final_processed = atomic_load(&total_lines_processed);
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double read_parse_time = (ts_end.tv_sec - ts_start.tv_sec) + (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;

    if (g_app.verbose >= 1) {
        printf("\nReading & parsing finished:\n");
        printf("  - Total lines processed: %ld\n", final_processed);
        printf("  - Time: %.2f seconds\n", read_parse_time);
        fflush(stdout);
    }

    clock_gettime(CLOCK_MONOTONIC, &ts_start); // reset time for mergin phase

    for (int t = 0; t < NUM_THREADS; t++) {
        if (g_app.verbose >= 2) {
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
            if (!arena_head) {
                arena_head = b;
                arena_current = b;
                arena_current->next = NULL;
            } else {
                arena_current->next = b;
                arena_current = b;
                arena_current->next = NULL;
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
    InternEntry *e = intern_table[h];
    while (e) {
        if (strcmp(e->str, s) == 0) return e->str;
        e = e->next;
    }
    char *copy = arena_alloc(len + 1);
    memcpy(copy, s, len);
    copy[len] = '\0';
    InternEntry *new_e = malloc(sizeof(InternEntry));
    new_e->str = copy;
    new_e->next = intern_table[h];
    intern_table[h] = new_e;
    return copy;
}

int load_synonyms(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        return -1;
    }
    int loaded_before = g_app.syn_count;
    char line[4096];
    while (fgets(line, sizeof(line), f)) {
        line[strcspn(line, "\n")] = 0;
        if (!*line || strlen(line) < 3)
            continue;
        char *head = strtok(line, ",");
        if (!head || !*head)
            continue;
        if (g_app.syn_count >= MAX_SYN_ENTRIES)
            break;
        SynEntry *e = &g_app.syn_table[g_app.syn_count++];
        e->word = intern(head);
        e->num_syns = 0;
        char *s;
        while ((s = strtok(NULL, ","))) {
            while (isspace(*s)) s++;
            char *end = s + strlen(s) - 1;
            while (end >= s && isspace(*end))
                *end-- = '\0';
            if (*s && e->num_syns < MAX_SYNONYMS_PER_WORD) {
                e->syns[e->num_syns++] = intern(s);
            }
        }
        if (e->num_syns == 0) {
            g_app.syn_count--;
        }
    }
    fclose(f);
    if (g_app.syn_count == loaded_before)
        return -2;
    return 0;
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

void add_starter(const char *word) {
    if (!word || !*word) return;
    unsigned long h = hash_str(word) % g_app.starter_table_size;
    Starter *s = starter_table[h];
    while (s) {
        if (s->word == word) {
            s->count++;
            total_starters++;
            return;
        }
        s = s->next;
    }
    Starter *new_s = malloc(sizeof(Starter));
    new_s->word = word;
    new_s->count = 1;
    new_s->next = starter_table[h];
    starter_table[h] = new_s;
    total_starters++;
}

double get_smoothed_prob(const char **prev_words, int prev_count, const char *word) {
    if (!word) return 0.0;

    double prob = 0.0;
    double remaining_mass = 1.0;

    int max_order = prev_count > 4 ? 4 : prev_count;
    for (int ord = max_order; ord >= 1; ord--) {
        if (remaining_mass <= 1e-9) break;

        ContextKey ctx = {0};
        ctx.order = (uint8_t)ord;
        for (int i = 0; i < ord; i++) {
            ctx.w[i] = prev_words[prev_count - 1 - i];
        }

        uint64_t h = hash_context(&ctx);
        size_t bucket = h % g_app.hashtable_size;

        long count = 0;
        long observed = 0;

        for (Entry *e = hashtable[bucket]; e; e = e->next) {
            if (context_equal(&e->key, &ctx)) {
                observed = e->total_count;
                for (Next *n = e->nexts; n; n = n->next) {
                    if (n->word == word) {
                        count = n->count;
                        break;
                    }
                }
                break;
            }
        }

        double local_p = (observed > 0) ? (count + 0.01) / (observed + 0.01 * vocab_size): 0.0;
        double w = lambda[ord] * remaining_mass;
        prob += w * local_p;
        remaining_mass -= w;
    }

    double uniform = 1.0 / vocab_size;
    prob += remaining_mass * uniform;

    return prob;
}

double compute_perplexity(const char **words, int nw) {
    if (nw < 2) return 1e12;

    double log_sum = 0.0;
    int transitions = 0;

    for (int i = 1; i < nw; i++) {
        int prev_count = (i >= 4) ? 4 : i;
        double p = get_smoothed_prob(&words[i - prev_count], prev_count, words[i]);

        if (p <= 0) p = 1e-10;
        log_sum += log(p);
        transitions++;
    }

    if (transitions == 0) return 1e12;

    double avg_nll = -log_sum / transitions;
    return exp(avg_nll);
}

static const char *choose_biased(Next *list, const char **keywords, int kw_count) {
    if (!list) return NULL;
    if (kw_count <= 0) return NULL;

    typedef struct { const char *word; double score; } Scored;
    Scored scored[256];
    int scount = 0;
    long total_raw = 0;

    for (Next *n = list; n; n = n->next) {
        if (!n->word) continue;
        total_raw += n->count;
    }
    if (total_raw == 0) return NULL;

    for (Next *n = list; n; n = n->next) {
        if (!n->word) continue;
        double base_p = (double)n->count / total_raw;
        double boost = 1.0;

        for (int k = 0; k < kw_count; k++) {
            const char *target = keywords[k];
            if (n->word == target) {
                boost *= 3.0;
                goto boosted;
            }
            for (int s = 0; s < syn_count; s++) {
                if (syn_table[s].word == target) {
                    for (int i = 0; i < syn_table[s].num_syns; i++) {
                        if (n->word == syn_table[s].syns[i]) {
                            boost *= 2.5;
                            goto boosted;
                        }
                    }
                }
            }
        }
    boosted:
        if (scount < 256) {
            scored[scount++] = (Scored){n->word, base_p * boost};
        }
    }

    double total_boosted = 0;
    for (int i = 0; i < scount; i++) total_boosted += scored[i].score;
    if (total_boosted <= 0) return NULL;

    double r = ((double)rand() / RAND_MAX) * total_boosted;
    double sum = 0;
    for (int i = 0; i < scount; i++) {
        sum += scored[i].score;
        if (r < sum) return scored[i].word;
    }
    return NULL;
}

static int generate_sequence(const char **out_words)
{
    const char **keywords  = g_app.keywords;
    int kw_count           = g_app.keyword_count;
    int max_tokens         = MAX_WORDS_PER_GENERATION;   // safety cap ? 

    const char *prev[4] = {NULL};
    int generated = 0;

    // Select starter
    if (g_app.total_starters > 0) {
        long r = rand() % g_app.total_starters;
        long sum = 0;
        for (int i = 0; i < g_app.starter_table_size; i++) {
            for (Starter *s = g_app.starter_table[i]; s; s = s->next) {
                sum += s->count;
                if (r < sum) {
                    prev[0] = s->word;
                    goto starter_found;
                }
            }
        }
    }

    prev[0] = intern("hello");

starter_found:;

    // Push initial context (up to 4 words)
    for (int k = 3; k >= 0; k--) {
        if (prev[k]) {
            if (generated >= max_tokens) break;
            out_words[generated++] = prev[k];
        }
    }

    // Main generation loop
    while (generated < max_tokens) {
        const char *candidates[4] = {NULL};
        double cand_weights[4]    = {0.0};
        double base_weights[4]    = {1.0, 0.7, 0.4, 0.2};
        double total_w = 0.0;

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
                    const char *w = choose_biased(e->nexts, keywords, kw_count);
                    if (w) {
                        int slot = 4 - order;
                        candidates[slot] = w;
                        double wgt = base_weights[slot];
                        if (kw_count > 0 && order < 4) wgt *= 1.2;
                        cand_weights[slot] = wgt;
                        total_w += wgt;
                    }
                    break;
                }
            }
        }

        const char *next = NULL;

        if (total_w > 0.0) {
            double r = ((double)rand() / RAND_MAX) * total_w;
            double acc = 0.0;
            for (int i = 0; i < 4; i++) {
                if (!candidates[i]) continue;
                acc += cand_weights[i];
                if (r <= acc) {
                    next = candidates[i];
                    break;
                }
            }
        }

        if (!next) {
            if (g_app.total_starters > 0) {
                long r = rand() % g_app.total_starters;
                long sum = 0;
                for (int i = 0; i < g_app.starter_table_size; i++) {
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
            next = intern(".");  // ultimate safety ?
        }

        out_words[generated++] = next;

        // Early stopping: once min length reached, stop at first sentence ender
        if (generated >= g_app.min_gen_len) {
            if (strcmp(next, ".") == 0 ||
                strcmp(next, "!") == 0 ||
                strcmp(next, "?") == 0) {
                break;  // natural sentence end → stop here
            }
        }
    }

    // Post-processing remains unchanged for now
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

    // Clamp length to min/max boundaries (still useful as final guard)
    if (len > g_app.max_gen_len) len = g_app.max_gen_len;
    if (len < g_app.min_gen_len) len = g_app.min_gen_len;

    // Optional: trim to last period if possible
    for (int i = len - 1; i >= 0; i--) {
        if (strcmp(out_words[start + i], ".") == 0) {
            len = i + 1;
            break;
        }
    }

    for (int i = 0; i < len; i++)
        out_words[i] = out_words[start + i];

    generated = len;

    // capitalization after '.'
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

    // force final dot
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

void free_model(void) {
    // Free global hashtable
    for (int i = 0; i < g_app.hashtable_size; i++) {
        Entry *e = g_app.hashtable[i];
        while (e) {
            Entry *ne = e->next;
            Next *n = e->nexts;
            while (n) {
                Next *nn = n->next;
                free_next(n);
                n = nn;
            }
            free_entry(e);
            e = ne;
        }
    }

    // Free starter table
    for (int i = 0; i < g_app.starter_table_size; i++) {
        Starter *s = g_app.starter_table[i];
        while (s) {
            Starter *ns = s->next;
            free(s);
            s = ns;
        }
    }

    // Free intern table
    for (int i = 0; i < g_app.intern_table_size; i++) {
        InternEntry *e = intern_table[i];
        while (e) {
            InternEntry *ne = e->next;
            free(e);
            e = ne;
        }
    }

    // Free arena blocks
    ArenaBlock *b = arena_head;
    while (b) {
        ArenaBlock *nb = b->next;
        free(b);
        b = nb;
    }

    // free all stored candidates
    for (int i = 0; i < g_app.candidate_count; i++) {
        free(g_app.candidates[i].words);
        g_app.candidates[i].words = NULL;
    }
    g_app.candidate_count = 0;

    g_app.hashtable = NULL;
    g_app.starter_table = NULL;
    arena_head = NULL;
    arena_current = NULL;
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
		
	if (load_synonyms(g_app.syn_file) < 0)
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
    
    generate_multiple_candidates();

	if (g_app.verbose >= 1) {
		display_global_debug();
		display_candidates();
	}
		
	select_and_print_best();

shutdown:
	free_model();
	return 0;
}

static double score_candidate(const char **words, int nw) {
    if (nw < 2) return 1e15;  // very bad

    int rep_count = 0;
    for (int j = 2; j < nw - 1; j++) {
        if (words[j] == words[j - 2] && words[j + 1] == words[j - 1]) {
            rep_count++;
        }
    }

    double rep_penalty = (rep_count > 4) ? 2.0 : 1.0;
    double ppl = compute_perplexity(words, nw);
    double length_bonus = 1.0 + 0.5 * log(nw + 1.0);

    return (ppl * rep_penalty) / length_bonus;
}

static void generate_multiple_candidates(void) {
    g_app.candidate_count = 0;

    int attempts = 0;
    const int max_attempts = g_app.max_candidates * 10; // while boundary 

    while (g_app.candidate_count < g_app.max_candidates && attempts < max_attempts) {
        attempts++;

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
                       g_app.candidate_count, nw, attempts);
            }
        }
    }

    if (g_app.verbose >= 1) {
        printf("\n[INFO] Finished generation: %d valid candidates collected "
               "(after %d total attempts, target was %d)\n",
               g_app.candidate_count, attempts, g_app.max_candidates);
    }

    if (g_app.candidate_count == 0 && attempts >= max_attempts) {
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

static int app_init(void) {
    memset(&g_app, 0, sizeof(g_app));
    g_app.input_file = NULL;
    g_app.output_file = NULL;
    g_app.syn_file = "synonyms.txt";
    g_app.min_gen_len = DEFAULT_MIN_TEXT_LENGTH;
    g_app.max_gen_len = DEFAULT_MAX_TEXT_LENGTH;
    g_app.max_candidates = MAX_CANDIDATES;
    g_app.num_threads = NUM_THREADS;
    g_app.seed = (unsigned)time(NULL);
    g_app.verbose = 1; // info level
    g_app.keyword_count = 0;
    g_app.keywords = NULL;
    g_app.hashtable_size = HASH_TABLE_SIZE; 
    g_app.starter_table_size = STARTER_TABLE_SIZE;
    g_app.intern_table_size = INTERN_TABLE_SIZE;
    g_app.vocab_size = 0;
    g_app.total_starters = 0;
    g_app.syn_count = 0;
    srand(g_app.seed);
    memset(g_app.candidates, 0, sizeof(g_app.candidates));
	g_app.candidate_count = 0;
	g_app.best_candidate_index = -1;
    arena_init();
    return 0;
}

void display_global_debug(void) {
    printf("\nApp Debug Dump:\n");
    
    printf("\n--- Config ---\n\n");
    printf("input_file: %s\n", g_app.input_file ? g_app.input_file : "NULL");
    printf("output_file: %s\n", g_app.output_file ? g_app.output_file : "NULL");
    printf("syn_file: %s\n", g_app.syn_file ? g_app.syn_file : "NULL");
    printf("min_gen_len: %d\n", g_app.min_gen_len);
    printf("max_gen_len: %d\n", g_app.max_gen_len);
    printf("max_candidates: %d\n", g_app.max_candidates);
    printf("num_threads: %d\n", g_app.num_threads);
    printf("seed: %u\n", g_app.seed);
    printf("verbose: %d\n", g_app.verbose);
    
    printf("\n--- Filters ---\n\n");
    printf("keyword_count: %d\n", g_app.keyword_count);
    if (g_app.keyword_count > 0 && g_app.keywords) {
        printf("keywords: ");
        for (int i = 0; i < g_app.keyword_count; i++) {
            printf("%s%s", g_app.keywords[i], (i < g_app.keyword_count - 1) ? ", " : "\n");
        }
    } else {
        printf("keywords: (empty)\n");
    }
    
    printf("\n--- Model ---\n\n");
    printf("hashtable: %p\n", (void *)g_app.hashtable);
    printf("hashtable_size: %ld\n", g_app.hashtable_size);
    printf("intern_table_size: %ld\n", g_app.intern_table_size);
    printf("starter_table: %p\n", (void *)g_app.starter_table);
    printf("starter_table_size: %p\n", (void *)g_app.starter_table_size);
    printf("vocab_size: %ld\n", g_app.vocab_size);
    printf("total_starters: %ld\n", g_app.total_starters);
    
	printf("\n--- Synonyms ---\n\n");
    printf("syn_count: %d\n", g_app.syn_count);
    if (g_app.syn_count > 0) {
        int total_synonyms = 0;
        for (int i = 0; i < 5; i++) { // for debug i just display the first 5 that's enough 
            int ns = g_app.syn_table[i].num_syns;
            total_synonyms += ns;
            printf("  syn_table[%d]: word='%s', num_syns=%d\n", i, g_app.syn_table[i].word, ns);
        }
        printf("  ...\n");
    } else {
        printf("syn_table: (empty)\n");
    }
    
    printf("\n--- Memory ---\n\n");
    printf("arena: head=%p, current=%p, block_size=%zu\n", (void *)arena_head, (void *)arena_current, (size_t)ARENA_BLOCK_SIZE);
    printf("\n");
}
