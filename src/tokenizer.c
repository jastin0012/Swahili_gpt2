// BPE tokenizer: loads `vocab.json` (token->id) and `merges.txt` BPE rules.
#include "tokenizer.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>

// Byte-level mapping used by GPT-2 / HF ByteLevel tokenizer
static char *byte_to_unicode[256] = {0};

// encode a Unicode codepoint into UTF-8 bytes (out must have >=4 bytes)
static int encode_utf8(int cp, char *out) {
    if (cp <= 0x7F) { out[0] = (char)cp; return 1; }
    else if (cp <= 0x7FF) {
        out[0] = (char)(0xC0 | ((cp >> 6) & 0x1F));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    } else if (cp <= 0xFFFF) {
        out[0] = (char)(0xE0 | ((cp >> 12) & 0x0F));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    } else {
        out[0] = (char)(0xF0 | ((cp >> 18) & 0x07));
        out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[3] = (char)(0x80 | (cp & 0x3F));
        return 4;
    }
}

static void bytes_to_unicode_init(void) {
    // build list of bytes considered "printable" in original implementation
    int bs_len = 0;
    unsigned char bs[512];
    for (int i = 33; i <= 126; ++i) bs[bs_len++] = (unsigned char)i;
    for (int i = 161; i <= 172; ++i) bs[bs_len++] = (unsigned char)i;
    for (int i = 174; i <= 255; ++i) bs[bs_len++] = (unsigned char)i;

    int cs[256];
    int n = 0;
    for (int b = 0; b < 256; ++b) cs[b] = -1;
    for (int i = 0; i < bs_len; ++i) cs[bs[i]] = bs[i];
    for (int b = 0; b < 256; ++b) {
        if (cs[b] == -1) {
            cs[b] = 256 + n;
            n++;
        }
    }

    for (int b = 0; b < 256; ++b) {
        int cp = cs[b];
        char tmp[5]; int L = encode_utf8(cp, tmp);
        tmp[L] = '\0';
        byte_to_unicode[b] = strdup(tmp);
    }
}

static void bytes_to_unicode_free(void) {
    for (int i = 0; i < 256; ++i) { free(byte_to_unicode[i]); byte_to_unicode[i] = NULL; }
}

typedef struct {
    char *key;
    int id;
} vocab_entry;

static vocab_entry *vocab_table = NULL;
static size_t vocab_table_size = 0; // number of slots
static size_t vocab_count = 0; // number of entries

static char **merges_left = NULL;
static char **merges_right = NULL;
static int *merge_rank = NULL;
static size_t merges_count = 0;

static size_t next_pow2(size_t v) {
    size_t n = 1; while (n < v) n <<= 1; return n;
}

static unsigned long hash_str(const char *s) {
    unsigned long h = 5381; int c; while ((c = (unsigned char)*s++)) h = ((h << 5) + h) + c; return h;
}

static int vocab_table_init(size_t expected) {
    vocab_table_size = next_pow2(expected * 2 + 1);
    vocab_table = calloc(vocab_table_size, sizeof(vocab_entry));
    return vocab_table ? 0 : -1;
}

static int vocab_insert(const char *k, int id) {
    if (!vocab_table) return -1;
    unsigned long h = hash_str(k);
    size_t idx = h & (vocab_table_size - 1);
    for (size_t i = 0; i < vocab_table_size; ++i) {
        size_t j = (idx + i) & (vocab_table_size - 1);
        if (vocab_table[j].key == NULL) {
            vocab_table[j].key = strdup(k);
            vocab_table[j].id = id;
            vocab_count++;
            return 0;
        }
    }
    return -1;
}

static int vocab_find(const char *k) {
    if (!vocab_table) return -1;
    unsigned long h = hash_str(k);
    size_t idx = h & (vocab_table_size - 1);
    for (size_t i = 0; i < vocab_table_size; ++i) {
        size_t j = (idx + i) & (vocab_table_size - 1);
        if (vocab_table[j].key == NULL) return -1;
        if (strcmp(vocab_table[j].key, k) == 0) return vocab_table[j].id;
    }
    return -1;
}

static void free_vocab_table(void) {
    if (!vocab_table) return;
    for (size_t i = 0; i < vocab_table_size; ++i) if (vocab_table[i].key) free(vocab_table[i].key);
    free(vocab_table); vocab_table = NULL; vocab_table_size = 0; vocab_count = 0;
}

static int load_vocab(const char *path) {
    if (!path) return -1;
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)sz + 1); if (!buf) { fclose(f); return -1; }
    fread(buf, 1, (size_t)sz, f); buf[sz] = '\0'; fclose(f);

    // estimate entries by counting occurrences of '":'
    size_t approx = 0;
    char *pp = buf;
    while ((pp = strstr(pp, "\":")) != NULL) { approx++; pp += 2; }
    if (approx < 128) approx = 128;
    vocab_table_init(approx + 16);

    char *p = buf;
    while ((p = strchr(p, '"')) != NULL) {
        p++; char *q = strchr(p, '"'); if (!q) break;
        size_t len = q - p;
        char *token = malloc(len + 1); memcpy(token, p, len); token[len] = '\0';
        p = q + 1;
        // find ':' after this point
        char *c = strchr(p, ':'); if (!c) { free(token); break; }
        c++;
        while (*c && isspace((unsigned char)*c)) c++;
        int id = (int)strtol(c, NULL, 10);
        vocab_insert(token, id);
        free(token);
    }
    free(buf);
    return 0;
}

// Prefer tokenizer.json if present: parse its "vocab" object (token->id)
static int load_tokenizer_json(const char *path) {
    if (!path) return -1;
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)sz + 1); if (!buf) { fclose(f); return -1; }
    fread(buf, 1, (size_t)sz, f); buf[sz] = '\0'; fclose(f);

    char *v = strstr(buf, "\"vocab\"");
    if (!v) { free(buf); return -1; }
    char *start = strchr(v, '{'); if (!start) { free(buf); return -1; }
    start++;
    // estimate approx entries
    size_t approx = 0; char *pp = start; while ((pp = strchr(pp, ':')) != NULL) { approx++; pp++; }
    if (approx < 128) approx = 128;
    vocab_table_init(approx + 16);

    char *p = start;
    while (1) {
        // find next quote
        char *q = strchr(p, '"'); if (!q) break; q++;
        char *r = strchr(q, '"'); if (!r) break;
        size_t len = r - q;
        char *token = malloc(len + 1); memcpy(token, q, len); token[len] = '\0';
        // find ':' after r
        char *c = strchr(r, ':'); if (!c) { free(token); break; }
        c++;
        while (*c && isspace((unsigned char)*c)) c++;
        int id = (int)strtol(c, NULL, 10);
        vocab_insert(token, id);
        free(token);
        p = c;
    }
    free(buf);
    return 0;
}

static int load_merges(const char *path) {
    if (!path) return -1;
    FILE *f = fopen(path, "r"); if (!f) return -1;
    char line[1024];
    size_t cap = 1024;
    merges_left = malloc(cap * sizeof(char*));
    merges_right = malloc(cap * sizeof(char*));
    merge_rank = malloc(cap * sizeof(int));
    merges_count = 0;
    int rank = 0;
    while (fgets(line, sizeof(line), f)) {
        // strip newline
        char *s = line; while (*s && (*s==' '||*s=='\t' || *s=='\r' || *s=='\n')) s++;
        if (*s == '#' || *s == '\0') continue;
        // split by space
        char *sp = strchr(s, ' ');
        if (!sp) continue;
        // left
        size_t l1 = sp - s;
        while (l1 > 0 && s[l1-1] == ' ') l1--;
        char *left = malloc(l1 + 1); memcpy(left, s, l1); left[l1] = '\0';
        // right
        char *rstart = sp + 1;
        while (*rstart == ' ') rstart++;
        char *rend = rstart; while (*rend && *rend!='\r' && *rend!='\n') rend++;
        size_t l2 = rend - rstart;
        char *right = malloc(l2 + 1); memcpy(right, rstart, l2); right[l2] = '\0';
        if (merges_count + 1 >= cap) {
            cap *= 2;
            merges_left = realloc(merges_left, cap * sizeof(char*));
            merges_right = realloc(merges_right, cap * sizeof(char*));
            merge_rank = realloc(merge_rank, cap * sizeof(int));
        }
        merges_left[merges_count] = left;
        merges_right[merges_count] = right;
        merge_rank[merges_count] = rank++;
        merges_count++;
    }
    fclose(f);
    return 0;
}

// Helper: read next UTF-8 char
static int utf8_char_len(const char *s) {
    unsigned char c = (unsigned char)s[0];
    if (c < 0x80) return 1;
    else if ((c >> 5) == 0x6) return 2;
    else if ((c >> 4) == 0xE) return 3;
    else if ((c >> 3) == 0x1E) return 4;
    return 1;
}

// Find merge rank for a pair, -1 if not found.
static int find_pair_rank(const char *a, const char *b) {
    for (size_t i = 0; i < merges_count; ++i) {
        if (strcmp(merges_left[i], a) == 0 && strcmp(merges_right[i], b) == 0) return merge_rank[i];
    }
    return -1;
}

int init_tokenizer(const char *vocab_path) {
    // try load tokenizer.json first (preferred)
    if (load_tokenizer_json("swahili_onnx_bundle/tokenizer.json") != 0) {
        // fallback to provided vocab.json
        load_vocab(vocab_path);
    }
    // init byte->unicode mapping
    bytes_to_unicode_init();
    // try merges in model/merges.txt or swahili_onnx_bundle/merges.txt
    if (load_merges("model/merges.txt") != 0) {
        load_merges("swahili_onnx_bundle/merges.txt");
    }
    return 0;
}

// BPE merge algorithm for a single word piece (using byte-level mapping)
static int bpe_tokenize_word(const char *word, int64_t *out_ids, size_t max_out) {
    // map each byte of word through byte_to_unicode to form initial symbols
    size_t parts_cap = 512; size_t parts = 0;
    char **seg = malloc(parts_cap * sizeof(char*));
    const unsigned char *p = (const unsigned char*)word;
    while (*p) {
        unsigned char b = *p++;
        char *s = byte_to_unicode[b] ? strdup(byte_to_unicode[b]) : strdup("?");
        if (parts + 1 >= parts_cap) { parts_cap *= 2; seg = realloc(seg, parts_cap * sizeof(char*)); }
        seg[parts++] = s;
    }

    // merge loop
    while (1) {
        int best_rank = -1; int best_i = -1;
        for (size_t i = 0; i + 1 < parts; ++i) {
            int r = find_pair_rank(seg[i], seg[i+1]);
            if (r >= 0 && (best_rank < 0 || r < best_rank)) { best_rank = r; best_i = (int)i; }
        }
        if (best_i < 0) break;
        // merge seg[best_i] and seg[best_i+1]
        size_t a_len = strlen(seg[best_i]); size_t b_len = strlen(seg[best_i+1]);
        char *m = malloc(a_len + b_len + 1);
        memcpy(m, seg[best_i], a_len); memcpy(m + a_len, seg[best_i+1], b_len); m[a_len + b_len] = '\0';
        free(seg[best_i]); free(seg[best_i+1]);
        seg[best_i] = m;
        // shift
        for (size_t j = best_i + 1; j + 1 < parts; ++j) seg[j] = seg[j+1];
        parts--;
    }

    // map segments to ids
    size_t out_idx = 0;
    for (size_t i = 0; i < parts && out_idx < max_out; ++i) {
        int id = vocab_find(seg[i]);
        if (id < 0) {
            // fallback: try the raw slice; if still no id, use 0
            id = vocab_find(seg[i]);
            if (id < 0) id = 0;
        }
        out_ids[out_idx++] = (int64_t)id;
    }

    for (size_t i = 0; i < parts; ++i) free(seg[i]); free(seg);
    return (int)out_idx;
}

int tokenize(const char *text, int64_t *out_ids, size_t max_ids) {
    if (!text || !out_ids) return 0;
    // HF ByteLevel often adds a prefix space to the input
    int add_prefix = 1;
    if (text[0] == ' ') add_prefix = 0;
    // build initial segments from bytes
    size_t parts_cap = 1024; size_t parts = 0;
    char **seg = malloc(parts_cap * sizeof(char*));
    const unsigned char *p = (const unsigned char*)text;
    if (add_prefix) {
        // prefix space byte
        unsigned char b = (unsigned char)' ';
        char *s = strdup(byte_to_unicode[b]);
        if (parts + 1 >= parts_cap) { parts_cap *= 2; seg = realloc(seg, parts_cap * sizeof(char*)); }
        seg[parts++] = s;
    }
    while (*p) {
        unsigned char b = *p++;
        char *s = byte_to_unicode[b] ? strdup(byte_to_unicode[b]) : strdup("?");
        if (parts + 1 >= parts_cap) { parts_cap *= 2; seg = realloc(seg, parts_cap * sizeof(char*)); }
        seg[parts++] = s;
    }

    // global merge loop: find best pair rank across entire sequence
    while (1) {
        int best_rank = -1; int best_i = -1;
        for (size_t i = 0; i + 1 < parts; ++i) {
            int r = find_pair_rank(seg[i], seg[i+1]);
            if (r >= 0 && (best_rank < 0 || r < best_rank)) { best_rank = r; best_i = (int)i; }
        }
        if (best_i < 0) break;
        // merge
        size_t a_len = strlen(seg[best_i]); size_t b_len = strlen(seg[best_i+1]);
        char *m = malloc(a_len + b_len + 1);
        memcpy(m, seg[best_i], a_len); memcpy(m + a_len, seg[best_i+1], b_len); m[a_len + b_len] = '\0';
        free(seg[best_i]); free(seg[best_i+1]);
        seg[best_i] = m;
        for (size_t j = best_i + 1; j + 1 < parts; ++j) seg[j] = seg[j+1];
        parts--;
    }

    // map segments to ids
    size_t out_idx = 0;
    for (size_t i = 0; i < parts && out_idx < max_ids; ++i) {
        int id = vocab_find(seg[i]);
        if (id < 0) {
            id = 0;
        }
        out_ids[out_idx++] = (int64_t)id;
    }

    for (size_t i = 0; i < parts; ++i) free(seg[i]); free(seg);
    return (int)out_idx;
}

int decode_ids(const int64_t *ids, size_t n, char *out, size_t out_size) {
    if (!out) return -1;
    size_t used = 0; out[0] = '\0';
    for (size_t i = 0; i < n; ++i) {
        int id = (int)ids[i];
        // find token by scanning vocab_table
        const char *tok = NULL;
        if (vocab_table) {
            for (size_t j = 0; j < vocab_table_size; ++j) if (vocab_table[j].key && vocab_table[j].id == id) { tok = vocab_table[j].key; break; }
        }
        if (!tok) tok = "";
        size_t len = strlen(tok);
        if (used + len + 2 >= out_size) break;
        if (used > 0) out[used++] = ' ';
        memcpy(out + used, tok, len); used += len; out[used] = '\0';
    }
    return 0;
}

const char *vocab_token_for_id(int id) {
    if (!vocab_table) return NULL;
    for (size_t j = 0; j < vocab_table_size; ++j) if (vocab_table[j].key && vocab_table[j].id == id) return vocab_table[j].key;
    return NULL;
}

void free_tokenizer(void) {
    if (merges_left) { for (size_t i = 0; i < merges_count; ++i) { free(merges_left[i]); free(merges_right[i]); } free(merges_left); free(merges_right); free(merge_rank); merges_left = NULL; merges_right = NULL; merge_rank = NULL; merges_count = 0; }
    free_vocab_table();
    bytes_to_unicode_free();
}
