# ============================
# run_all_fusions.ps1
# ============================

# Change to project root
Set-Location $PSScriptRoot\..

Write-Host "======================================" -ForegroundColor Cyan
Write-Host " Running ALL Multimodal Fusion Models" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# --- Activate environment ---
Write-Host "Activating conda environment: multimodal_posture" -ForegroundColor Yellow
conda activate multimodal_posture

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Could not activate conda environment!" -ForegroundColor Red
    exit 1
}

# Paths
$early_yaml      = "config/fusion_early.yaml"
$attention_yaml  = "config/fusion_attention.yaml"
$core_yaml       = "config/fusion_core.yaml"

$early_script     = "src/train_fusion_early.py"
$attention_script = "src/train_fusion_attention.py"
$core_script      = "src/train_fusion_core.py"

# --- 1. Early Fusion ---
Write-Host ""
Write-Host "[1/3] Running Early Fusion..." -ForegroundColor Cyan
python $early_script $early_yaml

if ($LASTEXITCODE -ne 0) {
    Write-Host "Early fusion FAILED!" -ForegroundColor Red
    exit 1
}
Write-Host "Early Fusion DONE." -ForegroundColor Green

# --- 2. Attention Fusion ---
Write-Host ""
Write-Host "[2/3] Running Attention Fusion..." -ForegroundColor Cyan
python $attention_script $attention_yaml

if ($LASTEXITCODE -ne 0) {
    Write-Host "Attention fusion FAILED!" -ForegroundColor Red
    exit 1
}
Write-Host "Attention Fusion DONE." -ForegroundColor Green

# # --- 3. Core Fusion ---
# Write-Host ""
# Write-Host "[3/3] Running CORE Fusion Model..." -ForegroundColor Cyan
# python $core_script $core_yaml

# if ($LASTEXITCODE -ne 0) {
#     Write-Host "Core fusion FAILED!" -ForegroundColor Red
#     exit 1
# }
# Write-Host "CORE Fusion Model DONE." -ForegroundColor Green

# --- All done ---
Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host " ALL TRAINING COMPLETE 🎉" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
