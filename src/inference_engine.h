#pragma once
#include <stddef.h>
int init_engine(const char *model_path);
int generate_text(const char *prompt, int max_tokens, char *out, size_t out_size);
int generate_text_from_ids(const int64_t *input_ids, int input_len, int max_tokens, char *out, size_t out_size);