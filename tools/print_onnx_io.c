#include <stdio.h>
#include <stdlib.h>
#include <onnxruntime_c_api.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s model.onnx\n", argv[0]);
        return 1;
    }
    const OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!ort) return 2;
    OrtEnv* env = NULL;
    if (ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "printio", &env) != NULL) return 3;
    OrtSessionOptions* opts = NULL;
    ort->CreateSessionOptions(&opts);

    OrtSession* session = NULL;
    OrtStatus* st = ort->CreateSession(env, argv[1], opts, &session);
    if (st != NULL) {
        const char* msg = ort->GetErrorMessage(st);
        fprintf(stderr, "CreateSession failed: %s\n", msg ? msg : "(no message)");
        return 4;
    }

    OrtAllocator* allocator = NULL;
    ort->GetAllocatorWithDefaultOptions(&allocator);

    size_t inp_count = 0, out_count = 0;
    ort->SessionGetInputCount(session, &inp_count);
    ort->SessionGetOutputCount(session, &out_count);
    printf("ONNX session inputs: %zu outputs: %zu\n", inp_count, out_count);

    for (size_t i = 0; i < inp_count; ++i) {
        char *name = NULL;
        ort->SessionGetInputName(session, i, allocator, &name);
        if (name) {
            printf(" input[%zu] = %s\n", i, name);
            allocator->Free(allocator, name);
        }
    }
    for (size_t i = 0; i < out_count; ++i) {
        char *name = NULL;
        ort->SessionGetOutputName(session, i, allocator, &name);
        if (name) {
            printf(" output[%zu] = %s\n", i, name);
            allocator->Free(allocator, name);
        }
    }

    ort->ReleaseSession(session);
    ort->ReleaseSessionOptions(opts);
    ort->ReleaseEnv(env);
    return 0;
}
