#include <stdio.h>
#include <stdlib.h>
#include "tokenizer.h"

int main(int argc, char **argv) {
    const char *text = (argc > 1) ? argv[1] : "Hujambo dunia";
    if (init_tokenizer("model/vocab.json") != 0) {
        fprintf(stderr, "init_tokenizer failed\n");
        return 1;
    }
    int64_t ids[1024];
    int n = tokenize(text, ids, sizeof(ids)/sizeof(ids[0]));
    printf("Token IDs (%d):", n);
    for (int i = 0; i < n; ++i) printf(" %lld", (long long)ids[i]);
    printf("\n");
    char out[4096];
    decode_ids(ids, n, out, sizeof(out));
    printf("Decoded: %s\n", out);
    // debug: print token string for each id
    printf("Tokens:\n");
    for (int i = 0; i < n; ++i) {
        const char *t = vocab_token_for_id((int)ids[i]);
        if (t) printf("  id=%lld -> '%s'\n", (long long)ids[i], t);
        else printf("  id=%lld -> <missing>\n", (long long)ids[i]);
    }
    free_tokenizer();
    return 0;
}
