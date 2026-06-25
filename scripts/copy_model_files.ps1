<# Copies tokenizer and vocab files from the bundled folders into `model/` #>
$src1 = "swahili_onnx_bundle"
$src2 = "swahili_gpt2_4bit_ultra"
$dst = "model"

Write-Host "Copying tokenizer and vocab files into $dst"
New-Item -ItemType Directory -Path $dst -Force | Out-Null

$files = @("tokenizer.json","vocab.json","special_tokens_map.json","tokenizer_config.json")
foreach ($f in $files) {
    if (Test-Path (Join-Path $src1 $f)) {
        Copy-Item (Join-Path $src1 $f) -Destination $dst -Force
        Write-Host "Copied $src1/$f"
    } elseif (Test-Path (Join-Path $src2 $f)) {
        Copy-Item (Join-Path $src2 $f) -Destination $dst -Force
        Write-Host "Copied $src2/$f"
    } else {
        Write-Host "Warning: $f not found in $src1 or $src2"
    }
}

Write-Host "Done. Check the $dst folder."
