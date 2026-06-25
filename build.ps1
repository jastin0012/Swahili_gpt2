<#
Build script for the C server and inference engine.

Requirements:
- Install GCC (e.g., Mingw-w64) on PATH.
- Download ONNX Runtime C SDK for Windows and set the following environment variables:
  - `ORT_INCLUDE` => path to onnxruntime/include
  - `ORT_LIB` => path to onnxruntime/lib

Usage:
.
    .\build.ps1

#>
if (-not $env:ORT_INCLUDE -or -not $env:ORT_LIB) {
  Write-Host "Please set ORT_INCLUDE and ORT_LIB environment variables to the ONNX Runtime SDK paths." -ForegroundColor Yellow
  Write-Host "Example: $env:ORT_INCLUDE = 'C:\\externals\\onnxruntime\\include'" -ForegroundColor Yellow
  # Try to auto-detect a bundled ONNX Runtime under third_party/onnxruntime
  $onnx_root = "third_party/onnxruntime"
  if (Test-Path $onnx_root) {
    # If the root directly contains include/lib, use it
    if ((Test-Path (Join-Path $onnx_root "include")) -and (Test-Path (Join-Path $onnx_root "lib"))) {
      $env:ORT_INCLUDE = Join-Path $onnx_root "include"
      $env:ORT_LIB = Join-Path $onnx_root "lib"
      Write-Host "Using ONNX Runtime from $onnx_root" -ForegroundColor Green
    } else {
      # If a zip is present, try extracting it automatically
      $zipPath = Join-Path $onnx_root "onnxruntime.zip"
      if (Test-Path $zipPath) {
        Write-Host "Found $zipPath - extracting to $onnx_root" -ForegroundColor Yellow
        try {
          Expand-Archive -Path $zipPath -DestinationPath $onnx_root -Force
        } catch {
          Write-Host ("Warning: failed to extract {0}: {1}" -f $zipPath, $_) -ForegroundColor Yellow
        }
      }
      # Otherwise try to find a child folder that contains include/lib.
      # Prefer CPU builds (folder name contains 'cpu' or does NOT contain 'gpu'/'cuda').
      $children = Get-ChildItem -Path $onnx_root -Directory -ErrorAction SilentlyContinue
      $found = $null
      foreach ($c in $children) {
        if ((Test-Path (Join-Path $c.FullName "include")) -and (Test-Path (Join-Path $c.FullName "lib"))) {
          $name = $c.Name.ToLower()
          if ($name -match 'cpu' -or ($name -notmatch 'gpu' -and $name -notmatch 'cuda')) {
            $found = $c
            break
          }
          if (-not $found) { $found = $c } # keep fallback
        }
      }
      if ($found) {
        # If the only available runtime is a GPU build, warn the user and suggest CPU runtime.
        $name = $found.Name.ToLower()
        if ($name -match 'gpu' -or $name -match 'cuda') {
          Write-Host "Warning: only a GPU CUDA ONNX Runtime was found ($($found.FullName))." -ForegroundColor Yellow
          Write-Host "This machine appears to be Intel/CPU-only; please download the CPU ONNX Runtime and set ORT_INCLUDE and ORT_LIB accordingly." -ForegroundColor Yellow
          Write-Host "Download from: https://onnxruntime.ai/docs/build/eps.html#prebuilt-packages" -ForegroundColor Yellow
        }
        $env:ORT_INCLUDE = Join-Path $found.FullName "include"
        $env:ORT_LIB = Join-Path $found.FullName "lib"
        Write-Host "Using ONNX Runtime from $($found.FullName)" -ForegroundColor Green
      }
    }
  }
}

$gcc = "gcc"
# Prefer explicit MSYS2 gcc path when available to avoid PATH/quoting issues in PowerShell
$msys_gcc = 'C:\msys64\ucrt64\bin\gcc.exe'
$gccPath = $gcc
if (Test-Path $msys_gcc) {
  $gccPath = $msys_gcc
  Write-Host "Using MSYS2 gcc at $msys_gcc" -ForegroundColor Green
}
$src = "src"
$out = "build"
New-Item -ItemType Directory -Path $out -Force | Out-Null

$serverSrc = Join-Path $src "server.c"
$engineSrc = Join-Path $src "inference_engine.c"
 $jsonSrc = Join-Path $src "json_simple.c"
 $tokenizerSrc = Join-Path $src "tokenizer.c"

# If Mongoose exists under externals/mongoose, compile it into the server build
 $mongoose_c = "externals/mongoose/mongoose.c"
 $mongoose_inc = "externals/mongoose"
 if (-not (Test-Path $mongoose_c)) {
   $mongoose_c = "third_party/mongoose/mongoose.c"
   $mongoose_inc = "third_party/mongoose"
 }
if (Test-Path $mongoose_c) {
  Write-Host "Found mongoose at $mongoose_c; including in build." -ForegroundColor Green
  $serverFiles = @($mongoose_c, (Join-Path $src "server.c"))
  # On Windows, link Winsock if using mingw
  $extra_libs = "-lws2_32"
} else {
  Write-Host "Mongoose not found in externals/mongoose; server build will fail unless you provide it." -ForegroundColor Yellow
  $serverFiles = @((Join-Path $src "server.c"))
  $extra_libs = ""
}

$serverOut = Join-Path $out "server.exe"
$engineOut = Join-Path $out "inference_engine.exe"

Write-Host "Compiling server..."
$serverArgs = @('-I' + $mongoose_inc)
if ($use_ort) { $serverArgs += @('-I' + $env:ORT_INCLUDE, '-L' + $env:ORT_LIB) }
$serverArgs += @('-o', $serverOut)
$serverArgs += $serverFiles
if ($use_ort) { $serverArgs += @('-lonnxruntime') }
if ($extra_libs -ne '') { $serverArgs += $extra_libs -split ' ' }
Write-Host (& $gccPath @serverArgs) 2>&1

Write-Host "Compiling inference engine..."
$define_ort = ""
if ($use_ort) { $define_ort = "-DUSE_ORT" }
$engineArgs = @('-I' + $mongoose_inc)
if ($use_ort) { $engineArgs += @('-I' + $env:ORT_INCLUDE, '-L' + $env:ORT_LIB) }
if ($define_ort -ne '') { $engineArgs += $define_ort }
$engineArgs += @('-o', $engineOut, $engineSrc, $jsonSrc, $tokenizerSrc)
if ($use_ort) { $engineArgs += @('-lonnxruntime') }
if ($extra_libs -ne '') { $engineArgs += $extra_libs -split ' ' }
Write-Host (& $gccPath @engineArgs) 2>&1

Write-Host "Build finished. Executables are in the 'build' directory." -ForegroundColor Green
