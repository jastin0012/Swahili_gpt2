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
$proc = Start-Process -FilePath $gcc -ArgumentList $args -NoNewWindow -Wait -PassThru -RedirectStandardOutput 'build/gcc_out.txt' -RedirectStandardError 'build/gcc_err.txt'
Write-Host "ExitCode $($proc.ExitCode)"
Write-Host "----- STDOUT -----"
Get-Content build/gcc_out.txt -ErrorAction SilentlyContinue | ForEach-Object { Write-Host $_ }
Write-Host "----- STDERR -----"
Get-Content build/gcc_err.txt -ErrorAction SilentlyContinue | ForEach-Object { Write-Host $_ }
