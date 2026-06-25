#pragma once
#include <stddef.h>
int parse_max_tokens(const char *body);
// concatenates all message.content fields into out (size out_size). Returns 0 on success.
int extract_messages_content(const char *body, char *out, size_t out_size);
// parse token_ids array from JSON body into out (max_out entries). Returns count.
int parse_token_ids(const char *body, int64_t *out, size_t max_out);