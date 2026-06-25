$gcc='C:\msys64\ucrt64\bin\gcc.exe'
$args = @(
  '-I', 'third_party/onnxruntime/onnxruntime-win-x64-1.27.0/include',
  '-I', 'third_party/mongoose',
  '-L', 'third_party/onnxruntime/onnxruntime-win-x64-1.27.0/lib',
  '-o', 'build/server.exe',
  'third_party/mongoose/mongoose.c',
  'src/server.c',
  'third_party/onnxruntime/onnxruntime-win-x64-1.27.0/lib/onnxruntime.dll',
  '-lws2_32'
)
Write-Host "Running: $gcc $($args -join ' ')"
& $gcc @args 2>&1 | Tee-Object build/gcc_combined.txt
Write-Host "Exit: $LASTEXITCODE"
Get-Content build/gcc_combined.txt -ErrorAction SilentlyContinue | ForEach-Object { Write-Host $_ }
