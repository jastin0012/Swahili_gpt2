#include "ort_loader.h"
#include <stdio.h>
#ifdef _WIN32
#include <windows.h>
#endif

static const OrtApi* g_ort = NULL;

int ort_init_from_dll(const char *dll_path) {
#ifdef _WIN32
    const char *path = dll_path ? dll_path : "onnxruntime.dll";
    HMODULE h = LoadLibraryA(path);
    if (!h) {
        return -1;
    }
    typedef const OrtApiBase* (ORT_API_CALL *OrtGetApiBase_t)(void);
    OrtGetApiBase_t get_api_base = (OrtGetApiBase_t)GetProcAddress(h, "OrtGetApiBase");
    if (!get_api_base) {
        FreeLibrary(h);
        return -1;
    }
    const OrtApiBase* base = get_api_base();
    if (!base) { FreeLibrary(h); return -1; }
    g_ort = base->GetApi(ORT_API_VERSION);
    return g_ort ? 0 : -1;
#else
    (void)dll_path; return -1;
#endif
}

const OrtApi* ort_get(void) { return g_ort; }
