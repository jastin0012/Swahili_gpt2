#include "mongoose.h"
#include "inference_engine.h"
#include "tokenizer.h"
#include "json_simple.h"
#include <stdlib.h>
#include <string.h>

static const char *MODEL_PATH = "model/model.onnx";
static const char *VOCAB_PATH = "model/vocab.json";
static int g_engine_ready = 0;
static int g_tokenizer_ready = 0;

static int ensure_ready(void) {
    if (!g_tokenizer_ready) {
        if (init_tokenizer(VOCAB_PATH) != 0) return 0;
        g_tokenizer_ready = 1;
    }
    if (!g_engine_ready) {
        if (init_engine(MODEL_PATH) != 0) return 0;
        g_engine_ready = 1;
    }
    return 1;
}

static void copy_mg_str(char *dst, size_t dst_sz, struct mg_str src) {
    size_t n = src.len < dst_sz - 1 ? src.len : dst_sz - 1;
    memcpy(dst, src.ptr, n);
    dst[n] = '\0';
}

static void escape_json(const char *src, char *dst, size_t dst_sz) {
    size_t pos = 0;
    while (*src && pos + 1 < dst_sz) {
        if (*src == '"' || *src == '\\' || *src == '/') {
            if (pos + 2 >= dst_sz) break;
            dst[pos++] = '\\';
            dst[pos++] = *src;
        } else if (*src == '\b') {
            if (pos + 2 >= dst_sz) break;
            dst[pos++] = '\\'; dst[pos++] = 'b';
        } else if (*src == '\f') {
            if (pos + 2 >= dst_sz) break;
            dst[pos++] = '\\'; dst[pos++] = 'f';
        } else if (*src == '\n') {
            if (pos + 2 >= dst_sz) break;
            dst[pos++] = '\\'; dst[pos++] = 'n';
        } else if (*src == '\r') {
            if (pos + 2 >= dst_sz) break;
            dst[pos++] = '\\'; dst[pos++] = 'r';
        } else if (*src == '\t') {
            if (pos + 2 >= dst_sz) break;
            dst[pos++] = '\\'; dst[pos++] = 't';
        } else {
            dst[pos++] = *src;
        }
        src++;
    }
    dst[pos] = '\0';
}

static void handle_chat_completion(struct mg_connection *c, struct mg_http_message *hm) {
    char body[8192];
    char prompt[4096];
    char result[8192];
    int64_t token_ids[1024];
    int token_count;

    copy_mg_str(body, sizeof(body), hm->body);
    token_count = parse_token_ids(body, token_ids, sizeof(token_ids) / sizeof(token_ids[0]));

    if (!ensure_ready()) {
        mg_http_reply(c, 500, "Content-Type: application/json\r\n", "{\"error\":\"engine_not_ready\"}");
        return;
    }

    if (token_count > 0) {
        if (generate_text_from_ids(token_ids, token_count, 32, result, sizeof(result)) != 0) {
            mg_http_reply(c, 500, "Content-Type: application/json\r\n", "{\"error\":\"inference_failed\"}");
            return;
        }
    } else {
        extract_messages_content(body, prompt, sizeof(prompt));
        if (prompt[0] == '\0') {
            mg_http_reply(c, 400, "Content-Type: application/json\r\n", "{\"error\":\"invalid_request\"}");
            return;
        }
        if (generate_text(prompt, 32, result, sizeof(result)) != 0) {
            mg_http_reply(c, 500, "Content-Type: application/json\r\n", "{\"error\":\"inference_failed\"}");
            return;
        }
    }

    char escaped[16384];
    escape_json(result, escaped, sizeof(escaped));
    mg_http_reply(c, 200, "Content-Type: application/json\r\n",
                  "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\"}}]}",
                  escaped);
}

static void event_handler(struct mg_connection *c, int ev, void *ev_data, void *fn_data) {
    if (ev == MG_EV_HTTP_MSG) {
        struct mg_http_message *hm = (struct mg_http_message *) ev_data;
        if (mg_http_match_uri(hm, "/v1/chat/completions")) {
            handle_chat_completion(c, hm);
        } else {
            mg_http_reply(c, 404, "Content-Type: application/json\r\n", "{\"error\":\"not_found\"}");
        }
    }
}

int main(void) {
    struct mg_mgr mgr;
    mg_mgr_init(&mgr);
    mg_http_listen(&mgr, "http://0.0.0.0:8080", event_handler, NULL);
    while (1) {
        mg_mgr_poll(&mgr, 1000);
    }
    mg_mgr_free(&mgr);
    return 0;
}
