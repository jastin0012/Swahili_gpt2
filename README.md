## Swahili GPT-2 Submission

This project implements a Swahili GPT-2-style inference pipeline in C using ONNX Runtime, with a lightweight HTTP server stub built on Mongoose.

It demonstrates:
- training support via `train_swahili.py` and whole notebook in `train.ipynb`
- tokenizer export and model export workflows
- tokenization and inference plumbing in C
- a minimal OpenAI-style `/v1/chat/completions` server stub

---

## Repository Layout

- `src/` — core C sources and headers for the inference engine and server
- `model/` — model artifacts and tokenizer files
- `scripts/` — helper utilities for model file setup and build support
- `tools/` — testing and inference utilities
- `python_server/` — Python-based server/demo support

---

## IMPORTANT: Model Files (Download Required)

The large model files are not guaranteed to be included in the GitHub repository due to size limits.

Download them manually and place them in `model/`.

- **model.onnx**  
  https://drive.google.com/file/d/1Hv84oWi7Qr10P6eIJXx-ABBEDov9EeFy/view?usp=sharing

- **model.safetensors**  
  https://drive.google.com/file/d/1Uuw_u_NYc4f9v-P1VkXdq3hDIQadmrCV/view?usp=sharing

Place both files inside:

```powershell
C:\Users\YourUser\Desktop\Swahili_gpt2\model
```

If `model.onnx` or `model.safetensors` are missing, the C inference and server builds will not work correctly.

---

## Build Requirements

- Windows with GCC or MSYS2/Mingw-w64 installed
- `mongoose.c` and `mongoose.h` available in `externals/mongoose/` or `third_party/mongoose/`

Optional for real inference:
- ONNX Runtime C SDK for Windows
- `ORT_INCLUDE` and `ORT_LIB` environment variables set to the ONNX Runtime SDK paths

Example PowerShell settings:

```powershell
$env:ORT_INCLUDE = 'C:\externals\onnxruntime\include'
$env:ORT_LIB = 'C:\externals\onnxruntime\lib'
```

If `ORT_INCLUDE`/`ORT_LIB` are not set, `build.ps1` will still compile a stub C server and engine. The stub server will run and accept requests, but it will not perform real ONNX inference.

---

## Setup Model and Tokenizer Files

This repository includes bundled tokenizer assets under `swahili_onnx_bundle/` and `swahili_gpt2_4bit_ultra/`.

The `model/` folder should contain the runtime artifacts needed for inference:
- `config.json`
- `generation_config.json`
- `merges.txt`
- `vocab.json`
- `model.safetensors`

`model/onxx` is not committed here and must be downloaded separately from the links above and placed in `model/`.

To populate `model/` with the tokenizer metadata from the bundle directories:

```powershell
.\scripts\copy_model_files.ps1
```

This helper copies tokenizer files from the bundle directories into `model/`:
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`

The helper does not download `model.onnx` or `model.safetensors` for you.

---

## Build the C Server and Inference Engine

Run the PowerShell build helper:

```powershell
.\build.ps1
```

If ONNX Runtime is not detected, set `ORT_INCLUDE` and `ORT_LIB` first.

The build script will attempt to find ONNX Runtime inside `third_party/onnxruntime` if the environment variables are not set.

Expected outputs:
- `build/server.exe`
- `build/inference_engine.exe`

---

## Smoke Tests

Run the tokenizer test:

```powershell
.\bin\test_tokenizer.exe "Hujambo dunia"
```

Run the Python ONNX smoke test:

```powershell
python -m pip install -r requirements.txt
python tools/run_onnx_infer.py "Habari rafiki"
```

---

## MSYS2 / UCRT64 Build

If you use MSYS2 UCRT64, you can also build via `make`:

```sh
make all
./bin/test_tokenizer "Hujambo dunia"
```

If `make` is unavailable, use raw GCC:

```sh
gcc -O2 -I./src -std=c11 -o bin/test_tokenizer src/test_tokenizer.c src/tokenizer.c
./bin/test_tokenizer "Hujambo dunia"
```

---

## API Example

The repo includes a minimal stub server in `src/server.c`.

To test a Chat Completions request while the server is running:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"swahili-slm","messages":[{"role":"user","content":"Hello"}]}'
```

