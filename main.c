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

#define NUM_THREADS 8
#define MAX_LINE_LEN 4096
#define WORD_MAX_LEN 128
#define MAX_WORDS_PER_GENERATION 500
#define MAX_CANDIDATES 20
#define HASH_TABLE_SIZE (1 << 21)
#define STARTER_TABLE_SIZE (1 << 17)
#define INTERN_TABLE_SIZE (1 << 20)
#define ARENA_BLOCK_SIZE (1ULL << 24)
#define CONTEXT_BUF_SIZE 1024
#define MAX_KEYWORDS 8
#define MAX_SYNONYMS_PER_WORD 32
#define MAX_SYN_ENTRIES 8192

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

static const double lambda[5] = {0.0, 0.15, 0.35, 0.60, 0.85};
InternEntry *intern_table[INTERN_TABLE_SIZE] = {0};
Entry *hashtable[HASH_TABLE_SIZE] = {0};
Starter *starter_table[STARTER_TABLE_SIZE] = {0};
SynEntry syn_table[MAX_SYN_ENTRIES];

int syn_count = 0;
long total_starters = 0;
long vocab_size = 0;

int MIN_GEN_LEN = 12;
int MAX_GEN_LEN = 28;

atomic_long total_lines_processed = 0;
long total_lines_global = 0;
static inline bool context_equal(const ContextKey *a, const ContextKey *b);
static inline uint64_t hash_context(const ContextKey *c);


static inline unsigned long hash_str(const char *str);
const char *intern(const char *s);
int split_into_words(char *line, const char **tokens_out, int max_tokens);


void merge_local_into_global(Entry **local, Entry **global) {
    struct TempMapping {
        Entry *local_entry;
        Entry *global_match;
    };

    struct TempMapping *mappings = NULL;
    size_t map_count = 0;
    size_t map_cap = 0;

    for (size_t i = 0; i < HASH_TABLE_SIZE; i++) {
        Entry *e = local[i];
        while (e) {
            Entry *next_e = e->next;

            uint64_t h = hash_context(&e->key);
            size_t bucket = h % HASH_TABLE_SIZE;

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
    for (size_t i = 0; i < STARTER_TABLE_SIZE; i++) {
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
    unsigned long h = hash_str(word) % STARTER_TABLE_SIZE;
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
        size_t idx = h & (HASH_TABLE_SIZE - 1);

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
        fprintf(stderr, "[Thread] Cannot open %s from byte %ld\n", td->filename, td->start_byte);
        return NULL;
    }

    if (fseek(f, td->start_byte, SEEK_SET) != 0) {
        fprintf(stderr, "[Thread] Seek failed to %ld\n", td->start_byte);
        fclose(f);
        return NULL;
    }

    char line[MAX_LINE_LEN];
    const char *words[256];
    const char *window[4] = {NULL};
    char seen[INTERN_TABLE_SIZE] = {0};
    long current_pos = td->start_byte;

    while (fgets(line, sizeof(line), f)) {
        size_t line_len = strlen(line);
        current_pos += line_len + 1;

        if (current_pos > td->end_byte) {
            break;
        }

        line[strcspn(line, "\n")] = '\0';

        int nw = split_into_words(line, words, 256);

        for (int j = 0; j < nw; j++) {
            const char *curr = words[j];

            unsigned long vh = hash_str(curr) % INTERN_TABLE_SIZE;
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

            window[3] = window[2];
            window[2] = window[1];
            window[1] = window[0];
            window[0] = curr;
        }
    }

    fclose(f);

    return NULL;
}

int build_model_mt(const char *filename) {
    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    FILE *f = fopen(filename, "r");
    if (!f) {
        perror("Cannot open file");
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

    memset(hashtable, 0, sizeof(hashtable));
    pthread_t threads[NUM_THREADS];
    ThreadData td[NUM_THREADS];

    long bytes_per_thread = file_size / NUM_THREADS;

    for (int t = 0; t < NUM_THREADS; t++) {
        td[t].filename = filename;
        td[t].start_byte = t * bytes_per_thread;
        td[t].end_byte   = (t == NUM_THREADS - 1) ? file_size : (t + 1) * bytes_per_thread;

        td[t].local_hashtable     = calloc(HASH_TABLE_SIZE, sizeof(Entry*));
        td[t].local_starters      = calloc(STARTER_TABLE_SIZE, sizeof(Starter*));
        td[t].local_total_starters = 0;
        td[t].local_vocab         = 0;
        td[t].local_arena         = calloc(1, sizeof(ArenaBlock));

        pthread_create(&threads[t], NULL, process_chunk, &td[t]);
    }

    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }

    long final_processed = atomic_load(&total_lines_processed);
	printf("All threads done — total lines processed: %ld\n", final_processed);
	fflush(stdout);

    for (int t = 0; t < NUM_THREADS; t++) {
        printf("Merging thread %d local table...\n", t);
        fflush(stdout);

        merge_local_into_global(td[t].local_hashtable, hashtable);
        merge_starters(td[t].local_starters, starter_table);

        total_starters += td[t].local_total_starters;
        vocab_size += td[t].local_vocab;

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
    double elapsed = (ts_end.tv_sec - ts_start.tv_sec) +
                     (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;

    printf("Total build time: %.2f s\n", elapsed);
    fflush(stdout);

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

    // Final avalanche mix (optional but helps distribute better)
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
    unsigned long h = hash_str(s) % INTERN_TABLE_SIZE;
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

void load_synonyms(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        printf("[Warning] Could not load synonyms.txt\n");
        return;
    }
    char line[4096];
    while (fgets(line, sizeof(line), f)) {
        line[strcspn(line, "\n")] = 0;
        if (!*line || strlen(line) < 3) continue;
        char *head = strtok(line, ",");
        if (!head || !*head) continue;
        if (syn_count >= MAX_SYN_ENTRIES) break;
        SynEntry *e = &syn_table[syn_count++];
        e->word = intern(head);
        e->num_syns = 0;
        char *s;
        while ((s = strtok(NULL, ","))) {
            while (isspace(*s)) s++;
            char *end = s + strlen(s) - 1;
            while (end >= s && isspace(*end)) *end-- = '\0';
            if (*s && e->num_syns < MAX_SYNONYMS_PER_WORD) {
                e->syns[e->num_syns++] = intern(s);
            }
        }
        if (e->num_syns == 0) {
            syn_count--;
        }
    }
    fclose(f);
    //printf("[Info] Loaded %d synonym entries.\n", syn_count);
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
    unsigned long h = hash_str(word) % STARTER_TABLE_SIZE;
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
        size_t bucket = h % HASH_TABLE_SIZE;

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

static int generate_sequence(const char **keywords,
                             int kw_count,
                             const char **out_words,
                             int max_tokens)
{
    const char *prev[4] = {NULL};
    int generated = 0;

    if (total_starters > 0) {
        long r = rand() % total_starters;
        long sum = 0;

        for (int i = 0; i < STARTER_TABLE_SIZE; i++) {
            for (Starter *s = starter_table[i]; s; s = s->next) {
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

    // push prev words into output
    for (int k = 3; k >= 0; k--) {
        if (prev[k]) {
            if (generated >= max_tokens) {
                break;
            }
            out_words[generated++] = prev[k];
        }
    }

    // -----------------------------
    // Generate rest of sequence
    // -----------------------------
	while (generated < max_tokens) {
		const char *candidates[4] = {0};
		double cand_weights[4]   = {0};
		double base_weights[4]   = {1.0, 0.7, 0.4, 0.2};
		double total_w = 0.0;

		for (int order = 4; order >= 1; order--) {
		    if (generated < order)
		        continue;

		    ContextKey ctx = {0};
		    ctx.order = order;
		    for (int k = 0; k < order; k++)
		        ctx.w[k] = out_words[generated - 1 - k];

		    uint64_t h = hash_context(&ctx);
		    size_t idx = h & (HASH_TABLE_SIZE - 1);

		    for (Entry *e = hashtable[idx]; e; e = e->next) {
		        if (context_equal(&e->key, &ctx)) {

		            const char *w = choose_biased(e->nexts, keywords, kw_count);

		            if (w) {
		                int slot = 4 - order;
		                candidates[slot] = w;
		                double wgt = base_weights[slot];
		                if (kw_count > 0 && order < 4)
		                    wgt *= 1.2;
		                cand_weights[slot] = wgt;
		                total_w += wgt;
		            }
		            break;
		        }
		    }
		}

		// ===== Defensive fallback =====
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

		// if no next candidate, pick random starter
		if (!next) {
		    long r = rand() % total_starters;
		    long sum = 0;
		    for (int i = 0; i < STARTER_TABLE_SIZE; i++) {
		        for (Starter *s = starter_table[i]; s; s = s->next) {
		            sum += s->count;
		            if (r < sum) {
		                next = s->word;
		                goto starter_chosen;
		            }
		        }
		    }
		    next = intern("hello");
		}

	starter_chosen:
		out_words[generated++] = next;
	}


    // -----------------------------
    // POST-PROCESSING (COSMETIC)
    // -----------------------------
    if (generated == 0) {
        printf("[DEBUG] generated == 0, returning 0\n");
        return 0;
    }

    // remove leading punctuation
    int start = 0;
    while (start < generated &&
           ispunct((unsigned char)out_words[start][0]) &&
           strcmp(out_words[start], ".") != 0){
               start++;
           }

	int len = generated - start;

	// Clamp length to min/max boundaries
	if (len > MAX_GEN_LEN) len = MAX_GEN_LEN;
	if (len < MIN_GEN_LEN) len = MIN_GEN_LEN;

	// Optional: ensure final word is a period if possible
	for (int i = len-1; i >= 0; i--) {
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
        if (!w) {
            printf("[DEBUG] Warning: out_words[%d] is NULL\n", i);
            continue;
        }
        if (cap && isalpha((unsigned char)w[0])) {
            char buf[WORD_MAX_LEN];
            strncpy(buf, w, sizeof(buf)-1);
            buf[sizeof(buf)-1] = '\0';
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
    bool first = true;
    for (int i = 0; i < count; i++) {
        const char *w = words[i];
        bool punct = is_punct(w);
        if (first) {
            char buf[WORD_MAX_LEN];
            strncpy(buf, w, sizeof(buf)-1);
            buf[sizeof(buf)-1] = '\0';
            if (isalpha((unsigned char)buf[0])) buf[0] = toupper((unsigned char)buf[0]);
            fprintf(out, "%s", buf);
            first = false;
        } else {
            if (punct) {
                fprintf(out, "%s", w);
            } else {
                fprintf(out, " %s", w);
            }
        }
        if (strchr(",;:", w[0])) fprintf(out, " ");
    }
}

// ---------------------- Main generation ----------------------
void generate(FILE *out, const char *user_input) {

    const char *keywords[MAX_KEYWORDS] = {0};
    int kw_count = 0;

    if (user_input && *user_input) {
        char tmp[MAX_LINE_LEN];
        strncpy(tmp, user_input, sizeof(tmp)-1);
        tmp[sizeof(tmp)-1] = '\0';

        char *tok = strtok(tmp, ",");
        while (tok && kw_count < MAX_KEYWORDS) {
            while (isspace(*tok)) tok++;
            char *end = tok + strlen(tok) - 1;
            while (end >= tok && isspace(*end)) *end-- = '\0';
            if (*tok) {
                keywords[kw_count] = intern(tok);
                kw_count++;
            }
            tok = strtok(NULL, ",");
        }
    }

    double best_score = 1e15;
    int best_len = 0;
    const char *best_words[MAX_WORDS_PER_GENERATION + 16] = {0};
    int best_count = 0;

    for (int cand = 0; cand < MAX_CANDIDATES; cand++) {
        const char *words[MAX_WORDS_PER_GENERATION + 16] = {0};
        int nw = generate_sequence(keywords, kw_count, words, MAX_WORDS_PER_GENERATION);

        if (nw < 40) {
            continue;
        }

        int rep_count = 0;
        for (int j = 2; j < nw - 1; j++) {
            if (words[j] == words[j-2] && words[j+1] == words[j-1]) rep_count++;
        }

        double rep_penalty = (rep_count > 4) ? 2.0 : 1.0;

		double ppl = compute_perplexity(words, nw);
		double score = (ppl * rep_penalty) / (1.0 + 0.5 * log(nw + 1));

        if (score < best_score || (score == best_score && nw > best_len)) {
            best_score = score;
            best_len = nw;
            best_count = nw;
            for (int i = 0; i < nw; i++) best_words[i] = words[i];
        }
    }

    if (best_count > 0) {
        print_words_properly(out, best_words, best_count);
        fprintf(out, "\n");
    }
}

void free_model(void) {
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        Entry *e = hashtable[i];
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
    for (int i = 0; i < STARTER_TABLE_SIZE; i++) {
        Starter *s = starter_table[i];
        while (s) {
            Starter *ns = s->next;
            free(s);
            s = ns;
        }
    }
    for (int i = 0; i < INTERN_TABLE_SIZE; i++) {
        InternEntry *e = intern_table[i];
        while (e) {
            InternEntry *ne = e->next;
            free(e);
            e = ne;
        }
    }
    ArenaBlock *b = arena_head;
    while (b) {
        ArenaBlock *nb = b->next;
        free(b);
        b = nb;
    }
}

static void parse_arguments(									// should be int return succes or 1
    int argc,
    char **argv,
    const char **keywords,
    int *kw_count
) {
    *kw_count = 0;

    for (int i = 2; i < argc; i++) {
        if (strncmp(argv[i], "keywords=", 9) == 0) {
            char tmp[MAX_LINE_LEN];
            strncpy(tmp, argv[i] + 9, sizeof(tmp) - 1); 		// snprintf() prevents buffer overflows and ensures null-termination,
            tmp[sizeof(tmp) - 1] = '\0';

            char *tok = strtok(tmp, ",");
            while (tok && *kw_count < MAX_KEYWORDS) {
                while (isspace(*tok)) tok++;

                char *end = tok + strlen(tok) - 1;
                while (end >= tok && isspace(*end)) {
                    *end-- = '\0';
                }

                if (*tok) {
                    keywords[(*kw_count)++] = intern(tok);
                }
                tok = strtok(NULL, ",");
            }

        } else if (strncmp(argv[i], "min_length=", 11) == 0) {
            MIN_GEN_LEN = atoi(argv[i] + 11); 					// atoi could be replace by strtol() safer and more robust

        } else if (strncmp(argv[i], "max_length=", 11) == 0) {
            MAX_GEN_LEN = atoi(argv[i] + 11);					// atoi could be replace by strtol() safer and more robust
        }
    }
}

static int find_best_candidate(
    const char **keywords,
    int kw_count,
    const char **best_words_out
) {
    int best_len = 0;
    double best_score = 1e15;

    for (int cand = 0; cand < MAX_CANDIDATES; cand++) {
        const char *words[MAX_WORDS_PER_GENERATION + 16] = {0};
        int nw = generate_sequence(keywords, kw_count, words, MAX_WORDS_PER_GENERATION);

        if (nw < MIN_GEN_LEN) continue;

        int rep_count = 0;
        for (int j = 2; j < nw - 1; j++) {
            if (words[j] == words[j - 2] &&
                words[j + 1] == words[j - 1]) {
                rep_count++;
            }
        }

        double rep_penalty = (rep_count > 4) ? 2.0 : 1.0;
        double ppl = compute_perplexity(words, nw);
        double score = (ppl * rep_penalty) / (1.0 + 0.5 * log(nw + 1));

        if (score < best_score || (score == best_score && nw > best_len)) {
            best_score = score;
            best_len = nw;
            for (int i = 0; i < nw; i++) {
                best_words_out[i] = words[i];
            }
        }
    }

    return best_len;
}

int main(int argc, char **argv) {
    srand(time(NULL));
    arena_init();

    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s corpus.txt [keywords=key1,key2] [min_length=XX] [max_length=XX]\n",
            argv[0]
        );
        return 1;
    }

    const char *filename = argv[1];
    const char *keywords[MAX_KEYWORDS] = {0};
    int kw_count = 0;

    parse_arguments(argc, argv, keywords, &kw_count);

    printf("Parsed %d keyword(s):\n", kw_count);
    for (int i = 0; i < kw_count; i++) {
        printf("  [%d]: '%s'\n", i, keywords[i]);
    }

    printf("=== Keyword-Guided Markov Chain ===\n");
    printf("Loading %s ...\n", filename);

    if (build_model_mt(filename) != 0) {
        fprintf(stderr, "Multi-threaded model build failed.\n");
        free_model();
        return 1;
    }

    load_synonyms("synonyms.txt");

    const char *best_words[MAX_WORDS_PER_GENERATION + 16] = {0};
    int best_len = find_best_candidate(keywords, kw_count, best_words);

    if (best_len > 0) {
        print_words_properly(stdout, best_words, best_len);
        printf("\n");
    }

    free_model();
    return 0;
}



