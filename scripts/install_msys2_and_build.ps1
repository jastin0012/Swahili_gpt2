# Install MSYS2 + mingw-w64 and run build.ps1
# Usage: Run from repository root in an elevated PowerShell if required.

function Write-Log { param($m) Write-Host "[install] $m" }

# 1) Check for existing gcc
if (Get-Command gcc -ErrorAction SilentlyContinue) {
    Write-Log "gcc already installed: $(gcc --version | Select-Object -First 1)"
    Write-Log "Running build.ps1"
    powershell -NoProfile -ExecutionPolicy Bypass -File .\build.ps1
    exit 0
}

# 2) Try winget
$msysPath = "C:\msys64"
if (Get-Command winget -ErrorAction SilentlyContinue) {
    Write-Log 'winget found - installing MSYS2'
    winget install --id MSYS2.MSYS2 -e --accept-source-agreements --accept-package-agreements
    Start-Sleep -Seconds 3
} else {
    Write-Log 'winget not available. Please install MSYS2 manually from https://www.msys2.org/ and re-run this script.'
    exit 1
}

# 3) Wait for msys2 installation path
$tries = 0
while (-not (Test-Path $msysPath) -and $tries -lt 20) {
    Start-Sleep -Seconds 2
    $tries++
}
if (-not (Test-Path $msysPath)) {
    Write-Log "MSYS2 path $msysPath not found. Please verify installation and re-run."
    exit 1
}

# 4) Run pacman to update and install gcc
$bash = Join-Path $msysPath "usr/bin/bash.exe"
if (-not (Test-Path $bash)) {
    Write-Log "MSYS2 bash not found at $bash"
    exit 1
}

Write-Log "Updating MSYS2 package DB and core packages (this may take a few minutes)"
& $bash -lc "pacman -Syu --noconfirm" 
Write-Log "Installing mingw-w64-x86_64-gcc"
& $bash -lc "pacman -S --noconfirm mingw-w64-x86_64-gcc"

# 5) Add mingw64 bin to current PATH
$mingwBin = Join-Path $msysPath "mingw64/bin"
if (Test-Path $mingwBin) {
    $env:PATH = $mingwBin + ';' + $env:PATH
    Write-Log ('Added {0} to PATH for this session' -f $mingwBin)
} else {
    Write-Log ('mingw bin not found at {0}' -f $mingwBin)
}

# 6) Run build script
Write-Log 'Running build.ps1'
powershell -NoProfile -ExecutionPolicy Bypass -File .\build.ps1

Write-Log 'Done'
