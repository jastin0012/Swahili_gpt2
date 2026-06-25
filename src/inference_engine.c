#include "inference_engine.h"
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static int engine_ready = 0;
static int64_t global_vocab_size = 0;
static int64_t global_eos_token_id = -1;

#ifdef USE_ORT
#include "ort_loader.h"
#include <onnxruntime_c_api.h>

static const OrtApi* ort = NULL;
static OrtEnv* env = NULL;
static OrtSession* session = NULL;
static OrtAllocator* allocator = NULL;
static char *session_input_name = NULL;
static char *session_output_name = NULL;
static size_t session_output_count = 0;

// KV-cache
static OrtValue **kv_cache = NULL;
static char **kv_input_names = NULL;
static size_t kv_count = 0;

int init_engine(const char *model_path) {
    if (engine_ready) return 0;
    if (ort_init_from_dll(NULL) != 0) return -1;
    ort = ort_get();
    if (!ort) return -1;
    if (ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "slm", &env) != NULL) return -1;
    OrtSessionOptions* opts = NULL;
    ort->CreateSessionOptions(&opts);
    ort->SetIntraOpNumThreads(opts, 1);
    ort->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_BASIC);
    if (ort->CreateSession(env, model_path, opts, &session) != NULL) {
        return -1;
    }
    ort->ReleaseSessionOptions(opts);
    ort->GetAllocatorWithDefaultOptions(&allocator);
    size_t inp_count = 0, out_count = 0;
    ort->SessionGetInputCount(session, &inp_count);
    ort->SessionGetOutputCount(session, &out_count);
    for (size_t i = 0; i < inp_count; ++i) {
        char *name = NULL;
        ort->SessionGetInputName(session, (size_t)i, allocator, &name);
        if (name) {
            if (i == 0) session_input_name = strdup(name);
            allocator->Free(allocator, name);
        }
    }
    for (size_t i = 0; i < out_count; ++i) {
        char *name = NULL;
        ort->SessionGetOutputName(session, (size_t)i, allocator, &name);
        if (name) {
            if (i == 0) session_output_name = strdup(name);
            allocator->Free(allocator, name);
        }
    }
    session_output_count = out_count;
    FILE *fv = fopen("model/vocab.json", "r");
    if (fv) { fclose(fv); init_tokenizer("model/vocab.json"); }
    else { init_tokenizer(NULL); }

    // try to parse model/config.json for vocab_size and eos_token_id
    FILE *fc = fopen("model/config.json", "r");
    if (fc) {
        fseek(fc, 0, SEEK_END);
        long sz = ftell(fc);
        fseek(fc, 0, SEEK_SET);
        char *buf = malloc((size_t)sz + 1);
        if (buf) {
            if (fread(buf, 1, (size_t)sz, fc) == (size_t)sz) {
                buf[sz] = '\0';
                const char *p = strstr(buf, "\"vocab_size\"");
                if (p) {
                    const char *c = strchr(p, ':');
                    if (c) global_vocab_size = atoll(c+1);
                }
                p = strstr(buf, "\"eos_token_id\"");
                if (p) {
                    const char *c = strchr(p, ':');
                    if (c) global_eos_token_id = atoll(c+1);
                }
            }
            free(buf);
        }
        fclose(fc);
    }
    if (global_vocab_size <= 0) global_vocab_size = 50257;
    if (global_eos_token_id < 0) global_eos_token_id = 50256;
    engine_ready = 1;
    return 0;
}

static void free_kv_cache(void) {
    if (!kv_cache) return;
    for (size_t i = 0; i < kv_count; ++i) {
        if (kv_cache[i]) ort->ReleaseValue(kv_cache[i]);
        if (kv_input_names && kv_input_names[i]) free(kv_input_names[i]);
    }
    free(kv_cache); kv_cache = NULL;
    free(kv_input_names); kv_input_names = NULL;
    kv_count = 0;
}

int generate_text(const char *prompt, int max_tokens, char *out, size_t out_size) {
    if (!engine_ready) {
        if (init_engine("model/model.onnx") != 0) {
            snprintf(out, out_size, "(engine init failed)");
            return -1;
        }
    }
    const size_t MAX_SEQ = 2048;
    int64_t ids[MAX_SEQ];
    int n = tokenize(prompt ? prompt : "", ids, MAX_SEQ);
    if (n <= 0) { snprintf(out, out_size, ""); return 0; }

    const char* input_names[] = { session_input_name ? session_input_name : "input_ids", "attention_mask", "position_ids" };
    const char* output_names[] = { session_output_name ? session_output_name : "logits" };

    int cur_len = n;
    srand((unsigned)time(NULL));
    const int TOP_K = 40;
    const float TEMPERATURE = 1.0f;

    for (int step = 0; step < max_tokens; ++step) {
        int64_t dims[2] = {1, cur_len};
        OrtMemoryInfo* mem_info = NULL;
        ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info);

        OrtValue* input_tensor = NULL;
        ort->CreateTensorWithDataAsOrtValue(mem_info, ids, cur_len * sizeof(int64_t), dims, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &input_tensor);

        // attention_mask: all ones
        int64_t *att = malloc(sizeof(int64_t) * cur_len);
        for (int i = 0; i < cur_len; ++i) att[i] = 1;
        OrtValue* att_tensor = NULL;
        ort->CreateTensorWithDataAsOrtValue(mem_info, att, cur_len * sizeof(int64_t), dims, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &att_tensor);

        // position_ids: 0..cur_len-1
        int64_t *pos = malloc(sizeof(int64_t) * cur_len);
        for (int i = 0; i < cur_len; ++i) pos[i] = i;
        OrtValue* pos_tensor = NULL;
        ort->CreateTensorWithDataAsOrtValue(mem_info, pos, cur_len * sizeof(int64_t), dims, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &pos_tensor);

        ort->ReleaseMemoryInfo(mem_info);

        int base_inputs = 3;
        int input_count = base_inputs + (int)kv_count;
        OrtValue **inputs = malloc(sizeof(OrtValue*) * input_count);
        const char **run_input_names = malloc(sizeof(char*) * input_count);
        inputs[0] = input_tensor; run_input_names[0] = input_names[0];
        inputs[1] = att_tensor;  run_input_names[1] = input_names[1];
        inputs[2] = pos_tensor;  run_input_names[2] = input_names[2];
        for (size_t i = 0; i < kv_count; ++i) { inputs[base_inputs + i] = kv_cache[i]; run_input_names[base_inputs + i] = kv_input_names[i]; }

        size_t out_count = 1;
        char **run_output_names = NULL;
        if (kv_count == 0 && session_output_count > 0) {
            out_count = session_output_count;
            run_output_names = malloc(sizeof(char*) * out_count);
            for (size_t oi = 0; oi < out_count; ++oi) {
                char *nm = NULL;
                ort->SessionGetOutputName(session, oi, allocator, &nm);
                run_output_names[oi] = nm; // will free via allocator later
            }
        } else {
            out_count = 1;
            run_output_names = malloc(sizeof(char*) * 1);
            run_output_names[0] = (char*)output_names[0];
        }

        OrtValue **outputs_all = malloc(sizeof(OrtValue*) * out_count);
        ort->Run(session, NULL, run_input_names, (const OrtValue* const*)inputs, input_count, (const char* const*)run_output_names, out_count, outputs_all);

        // get logits pointer from outputs_all[0]
        float *logits = NULL;
        ort->GetTensorMutableData(outputs_all[0], (void**)&logits);

        int64_t vocab = global_vocab_size;
        int64_t last_offset = (int64_t)(cur_len - 1) * vocab;
        // sampling: take top-k then softmax with temperature
        int64_t k = (vocab < TOP_K) ? vocab : TOP_K;
        // find top-k ids
        int64_t *top_ids = malloc(sizeof(int64_t) * k);
        float *top_vals = malloc(sizeof(float) * k);
        for (int64_t i = 0; i < k; ++i) { top_ids[i] = -1; top_vals[i] = -INFINITY; }
        for (int64_t i = 0; i < vocab; ++i) {
            float v = logits[last_offset + i];
            // keep sorted min-heap-like simple insertion
            for (int j = 0; j < k; ++j) {
                if (v > top_vals[j]) {
                    // shift
                    for (int t = k-1; t > j; --t) { top_vals[t] = top_vals[t-1]; top_ids[t] = top_ids[t-1]; }
                    top_vals[j] = v; top_ids[j] = i; break;
                }
            }
        }

        // compute softmax over top-k with temperature
        double sum = 0.0;
        double *probs = malloc(sizeof(double) * k);
        for (int j = 0; j < k; ++j) {
            double x = (double)top_vals[j] / (double)TEMPERATURE;
            double ex = exp(x);
            probs[j] = ex; sum += ex;
        }
        if (sum <= 0.0) sum = 1.0;
        for (int j = 0; j < k; ++j) probs[j] /= sum;

        // sample
        double r = (double)rand() / (double)RAND_MAX;
        double accum = 0.0;
        int64_t best_id = top_ids[k-1]; // fallback if rounding
        for (int j = 0; j < k; ++j) {
            accum += probs[j];
            if (r <= accum) { best_id = top_ids[j]; break; }
        }

        free(top_ids); free(top_vals); free(probs);

        // append
        if (cur_len + 1 >= (int)MAX_SEQ) {
            // reached max buffer
            if (input_tensor) ort->ReleaseValue(input_tensor);
            break;
        }
        ids[cur_len++] = best_id;

        // manage outputs and kv-cache
        if (kv_count == 0 && out_count > 1) {
            // first time: capture presents from outputs_all[1..]
            kv_count = out_count - 1;
            kv_cache = malloc(sizeof(OrtValue*) * kv_count);
            kv_input_names = malloc(sizeof(char*) * kv_count);
            for (size_t i = 0; i < kv_count; ++i) {
                kv_cache[i] = outputs_all[1 + i]; // take ownership, do not release now
                // get corresponding output name to reuse as input name
                char *nm = NULL;
                ort->SessionGetOutputName(session, 1 + i, allocator, &nm);
                kv_input_names[i] = nm ? strdup(nm) : NULL;
                if (nm) allocator->Free(allocator, nm);
            }
        } else if (kv_count > 0 && out_count > 1) {
            // subsequent: outputs_all[1..] contain new presents; replace cache
            for (size_t i = 0; i < kv_count && (1 + i) < out_count; ++i) {
                if (kv_cache[i]) ort->ReleaseValue(kv_cache[i]);
                kv_cache[i] = outputs_all[1 + i]; // take ownership
            }
        }

        // release logits output
        if (outputs_all[0]) ort->ReleaseValue(outputs_all[0]);
        // free outputs_all array (but not kv_cache elements)
        free(outputs_all);

        // free run_output_names entries allocated via SessionGetOutputName
        if (kv_count == 0 && run_output_names) {
            for (size_t oi = 0; oi < out_count; ++oi) if (run_output_names[oi]) allocator->Free(allocator, run_output_names[oi]);
        }
        free(run_output_names);

        if (input_tensor) ort->ReleaseValue(input_tensor);
        if (att_tensor) ort->ReleaseValue(att_tensor);
        if (pos_tensor) ort->ReleaseValue(pos_tensor);
        free(att); free(pos);
        free(inputs);
        free(run_input_names);

        if (best_id == global_eos_token_id) break;
    }

    // decode
    decode_ids(ids, (size_t)cur_len, out, out_size);

    // do not free session_input_name/session_output_name here; free in free_tokenizer or on shutdown
    return 0;
}

int generate_text_from_ids(const int64_t *input_ids, int input_len, int max_tokens, char *out, size_t out_size) {
    if (!engine_ready) {
        if (init_engine("model/model.onnx") != 0) {
            snprintf(out, out_size, "(engine init failed)");
            return -1;
        }
    }
    const size_t MAX_SEQ = 2048;
    int64_t ids[MAX_SEQ];
    if (input_len <= 0) { snprintf(out, out_size, ""); return 0; }
    if ((size_t)input_len >= MAX_SEQ) input_len = (int)(MAX_SEQ - 1);
    for (int i = 0; i < input_len; ++i) ids[i] = input_ids[i];

    const char* input_names[] = { session_input_name ? session_input_name : "input_ids", "attention_mask", "position_ids" };
    const char* output_names[] = { session_output_name ? session_output_name : "logits" };

    int cur_len = input_len;
    srand((unsigned)time(NULL));
    const int TOP_K = 40;
    const float TEMPERATURE = 1.0f;

    for (int step = 0; step < max_tokens; ++step) {
        // prepare tensors: input_ids, attention_mask, position_ids
        int64_t dims[2] = {1, cur_len};
        OrtMemoryInfo* mem_info = NULL;
        ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info);

        OrtValue* input_tensor = NULL;
        ort->CreateTensorWithDataAsOrtValue(mem_info, ids, cur_len * sizeof(int64_t), dims, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &input_tensor);

        // attention_mask: all ones
        int64_t *att = malloc(sizeof(int64_t) * cur_len);
        for (int i = 0; i < cur_len; ++i) att[i] = 1;
        OrtValue* att_tensor = NULL;
        ort->CreateTensorWithDataAsOrtValue(mem_info, att, cur_len * sizeof(int64_t), dims, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &att_tensor);

        // position_ids: 0..cur_len-1
        int64_t *pos = malloc(sizeof(int64_t) * cur_len);
        for (int i = 0; i < cur_len; ++i) pos[i] = i;
        OrtValue* pos_tensor = NULL;
        ort->CreateTensorWithDataAsOrtValue(mem_info, pos, cur_len * sizeof(int64_t), dims, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &pos_tensor);

        ort->ReleaseMemoryInfo(mem_info);

        // Prepare to run: include kv-cache inputs if present
        OrtValue **inputs = NULL;
        const char **run_input_names = NULL;
        int base_inputs = 3; // input_ids, attention_mask, position_ids
        int input_count = base_inputs + (int)kv_count;
        inputs = malloc(sizeof(OrtValue*) * input_count);
        run_input_names = malloc(sizeof(char*) * input_count);
        inputs[0] = input_tensor; run_input_names[0] = input_names[0];
        inputs[1] = att_tensor;  run_input_names[1] = input_names[1];
        inputs[2] = pos_tensor;  run_input_names[2] = input_names[2];
        for (size_t i = 0; i < kv_count; ++i) { inputs[base_inputs + i] = kv_cache[i]; run_input_names[base_inputs + i] = kv_input_names[i]; }

        // Decide outputs to request
        size_t out_count = 1;
        char **run_output_names = NULL;
        if (kv_count == 0 && session_output_count > 0) {
            out_count = session_output_count;
            run_output_names = malloc(sizeof(char*) * out_count);
            for (size_t oi = 0; oi < out_count; ++oi) {
                char *nm = NULL;
                ort->SessionGetOutputName(session, oi, allocator, &nm);
                run_output_names[oi] = nm; // will free via allocator later
            }
        } else {
            out_count = 1;
            run_output_names = malloc(sizeof(char*) * 1);
            run_output_names[0] = (char*)output_names[0];
        }

        OrtValue **outputs_all = malloc(sizeof(OrtValue*) * out_count);
        ort->Run(session, NULL, run_input_names, (const OrtValue* const*)inputs, input_count, (const char* const*)run_output_names, out_count, outputs_all);

        float *logits = NULL;
        ort->GetTensorMutableData(outputs_all[0], (void**)&logits);

        int64_t vocab = global_vocab_size;
        int64_t last_offset = (int64_t)(cur_len - 1) * vocab;
        int64_t k = (vocab < TOP_K) ? vocab : TOP_K;
        int64_t *top_ids = malloc(sizeof(int64_t) * k);
        float *top_vals = malloc(sizeof(float) * k);
        for (int64_t i = 0; i < k; ++i) { top_ids[i] = -1; top_vals[i] = -INFINITY; }
        for (int64_t i = 0; i < vocab; ++i) {
            float v = logits[last_offset + i];
            for (int j = 0; j < k; ++j) {
                if (v > top_vals[j]) { for (int t = k-1; t > j; --t) { top_vals[t] = top_vals[t-1]; top_ids[t] = top_ids[t-1]; } top_vals[j] = v; top_ids[j] = i; break; }
            }
        }
        double sum = 0.0;
        double *probs = malloc(sizeof(double) * k);
        for (int j = 0; j < k; ++j) { double x = (double)top_vals[j] / (double)TEMPERATURE; double ex = exp(x); probs[j] = ex; sum += ex; }
        if (sum <= 0.0) sum = 1.0;
        for (int j = 0; j < k; ++j) probs[j] /= sum;
        double r = (double)rand() / (double)RAND_MAX; double accum = 0.0; int64_t best_id = top_ids[k-1];
        for (int j = 0; j < k; ++j) { accum += probs[j]; if (r <= accum) { best_id = top_ids[j]; break; } }
        free(top_ids); free(top_vals); free(probs);

        if (cur_len + 1 >= (int)MAX_SEQ) {
            if (input_tensor) ort->ReleaseValue(input_tensor);
            if (att_tensor) ort->ReleaseValue(att_tensor);
            if (pos_tensor) ort->ReleaseValue(pos_tensor);
            free(att); free(pos);
            free(inputs); free(run_input_names);
            break;
        }
        ids[cur_len++] = best_id;

        if (kv_count == 0 && out_count > 1) {
            kv_count = out_count - 1;
            kv_cache = malloc(sizeof(OrtValue*) * kv_count);
            kv_input_names = malloc(sizeof(char*) * kv_count);
            for (size_t i = 0; i < kv_count; ++i) {
                kv_cache[i] = outputs_all[1 + i];
                char *nm = NULL; ort->SessionGetOutputName(session, 1 + i, allocator, &nm);
                kv_input_names[i] = nm ? strdup(nm) : NULL; if (nm) allocator->Free(allocator, nm);
            }
        } else if (kv_count > 0 && out_count > 1) {
            for (size_t i = 0; i < kv_count && (1 + i) < out_count; ++i) {
                if (kv_cache[i]) ort->ReleaseValue(kv_cache[i]); kv_cache[i] = outputs_all[1 + i];
            }
        }

        if (outputs_all[0]) ort->ReleaseValue(outputs_all[0]);
        free(outputs_all);
        if (kv_count == 0 && run_output_names) { for (size_t oi = 0; oi < out_count; ++oi) if (run_output_names[oi]) allocator->Free(allocator, run_output_names[oi]); }
        free(run_output_names);

        if (input_tensor) ort->ReleaseValue(input_tensor);
        if (att_tensor) ort->ReleaseValue(att_tensor);
        if (pos_tensor) ort->ReleaseValue(pos_tensor);
        free(att); free(pos);
        free(inputs); free(run_input_names);

        if (best_id == global_eos_token_id) break;
    }

    decode_ids(ids, (size_t)cur_len, out, out_size);
    return 0;
}

#else

int init_engine(const char *model_path) {
    (void)model_path;
    engine_ready = 1;
    FILE *fv = fopen("model/vocab.json", "r");
    if (fv) { fclose(fv); init_tokenizer("model/vocab.json"); }
    else { init_tokenizer(NULL); }
    return 0;
}

int generate_text(const char *prompt, int max_tokens, char *out, size_t out_size) {
    if (!engine_ready) init_engine("model/model.onnx");
    (void)max_tokens;
    if (!prompt) prompt = "";
    snprintf(out, out_size, "%s", prompt);
    size_t len = strlen(out);
    if (len + 5 < out_size) strncat(out, "...", out_size - len - 1);
    return 0;
}

#endif

/* library-only: no standalone main here */
