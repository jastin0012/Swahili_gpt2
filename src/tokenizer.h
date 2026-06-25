#pragma once
#include <stddef.h>
#include <stdint.h>
int init_tokenizer(const char *vocab_path);
int tokenize(const char *text, int64_t *out_ids, size_t max_ids);
int decode_ids(const int64_t *ids, size_t n, char *out, size_t out_size);
void free_tokenizer(void);
const char *vocab_token_for_id(int id);