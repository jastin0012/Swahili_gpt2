#ifndef ORT_LOADER_H
#define ORT_LOADER_H
#ifdef _WIN32
#include <windows.h>
#endif
#include <onnxruntime_c_api.h>

// Initialize the ONNX Runtime by loading the runtime DLL at runtime.
// If dll_path is NULL, the loader will try to load "onnxruntime.dll" from PATH.
// Returns 0 on success, non-zero on failure.
int ort_init_from_dll(const char *dll_path);

// Returns the initialized Ort API pointer (or NULL if not initialized).
const OrtApi* ort_get(void);

#endif