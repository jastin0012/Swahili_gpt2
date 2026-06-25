#include "json_simple.h"
#include <string.h>
#include <stdlib.h>

static const char *find_key(const char *s, const char *key) {
    return strstr(s, key);
}

int parse_max_tokens(const char *body) {
    const char *p = find_key(body, "\"max_tokens\"");
    if (!p) return 0;
    p = strchr(p, ':');
    if (!p) return 0;
    p++;
    while (*p && (*p == ' ' || *p == '\n' || *p == '\r')) p++;
    int val = atoi(p);
    return val;
}

// simple extraction of all "content" string values concatenated with spaces
int extract_messages_content(const char *body, char *out, size_t out_size) {
    const char *p = body;
    size_t used = 0;
    out[0] = '\0';
    while ((p = find_key(p, "\"content\"")) != NULL) {
        const char *q = strchr(p, ':');
        if (!q) break;
        q++;
        while (*q == ' ' || *q == '\n' || *q == '\r') q++;
        if (*q == '"') {
            q++;
            const char *end = q;
            while (*end && *end != '"') end++;
            size_t len = (size_t)(end - q);
            if (len == 0) { p = end + 1; continue; }
            if (used + len + 2 >= out_size) break;
            if (used > 0) { out[used++] = ' '; }
            memcpy(out + used, q, len);
            used += len;
            out[used] = '\0';
            p = end + 1;
        } else {
            p = q;
        }
    }
    return 0;
}

// Parse a top-level JSON array called "token_ids": [n, n, ...]
int parse_token_ids(const char *body, int64_t *out, size_t max_out) {
    const char *p = strstr(body, "\"token_ids\"");
    if (!p) return 0;
    p = strchr(p, '[');
    if (!p) return 0;
    p++;
    size_t idx = 0;
    while (*p && *p != ']') {
        while (*p && (*p == ' ' || *p == '\n' || *p == '\r' || *p == ',')) p++;
        if (*p == ']') break;
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        if (idx < max_out) out[idx++] = (int64_t)v;
        p = end;
    }
    return (int)idx;
}
