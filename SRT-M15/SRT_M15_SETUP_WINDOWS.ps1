param(
    [switch]$SkipModelDownload,
    [switch]$SkipCudaTorch
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
Write-Host "=== SRT-M15 RTX 3080 local setup ===" -ForegroundColor Cyan
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher not found. Install Python 3.11 for Windows first."
}
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Cannot create Python 3.11 virtual environment." }
}
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
& $Python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Project Python dependencies failed." }
if (-not $SkipCudaTorch) {
    Write-Host "Installing PyTorch CUDA 12.8 wheel for RTX 3080..." -ForegroundColor Cyan
    & $Python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA PyTorch install failed. Check your NVIDIA driver and official PyTorch instructions."
    }
}
Write-Host "Checking PyTorch GPU:" -ForegroundColor Cyan
& $Python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not detected')"
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Warning "Ollama not installed. Install it from https://ollama.com/download/windows and rerun."
} elseif (-not $SkipModelDownload) {
    Write-Host "Downloading the local Qwen3 4B Instruct model (first run only)..." -ForegroundColor Cyan
    & ollama pull qwen3:4b-instruct
    if ($LASTEXITCODE -ne 0) { Write-Warning "Qwen download failed; retry after starting Ollama." }
}
Write-Host "Self-check:" -ForegroundColor Cyan
& $Python srt_full_pipeline.py doctor
Write-Host ""
Write-Host "Setup finished. Run: SRT_M15_FULL_SUITE_WINDOWS.bat" -ForegroundColor Green
Write-Host "IMPORTANT: Teacher responses still need human review before strict training."
