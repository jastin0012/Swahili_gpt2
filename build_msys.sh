#!/usr/bin/env bash
set -eux

# Navigate to project (escape space in username)
cd /c/Users/Salvatory\ Jr/Desktop/Swahili_gpt2
mkdir -p build

# Paths
ORT_DIR=third_party/onnxruntime/onnxruntime-win-x64-1.27.0
MONGOOSE_INC=third_party/mongoose

# Compile server
gcc -I "$ORT_DIR/include" -I "$MONGOOSE_INC" -L "$ORT_DIR/lib" -o build/server.exe \
    third_party/mongoose/mongoose.c src/server.c -lonnxruntime -lws2_32

# Compile inference engine
gcc -I "$ORT_DIR/include" -I "$MONGOOSE_INC" -L "$ORT_DIR/lib" -DUSE_ORT -o build/inference_engine.exe \
    src/inference_engine.c src/json_simple.c src/tokenizer.c -lonnxruntime -lws2_32

echo "Build done"
