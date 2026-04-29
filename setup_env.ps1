# AutoQSAR Environment Setup Script (PowerShell)

Write-Host "========================================"
Write-Host "AutoQSAR Conda Environment Setup"
Write-Host "========================================"
Write-Host ""
Write-Host "Choose your setup option:"
Write-Host "1. CUDA GPU (recommended if you have NVIDIA GPU with CUDA 12.1+)"
Write-Host "2. CPU-only (for laptops or non-GPU systems)"
Write-Host "3. Custom (manual PyTorch installation)"
Write-Host ""

$choice = Read-Host "Enter your choice (1, 2, or 3)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "Creating CUDA environment..."
        & conda env create -f environment-cuda.yml -n autoqsar-py311
        if (-not $?) { Write-Host "Conda env creation failed."; Pause; exit 1 }
        Write-Host ""
        Write-Host "Installing pip packages via uv (this may take several minutes)..."
        & conda run -n autoqsar-py311 pip install uv
        & conda run -n autoqsar-py311 uv pip install --system --index-strategy unsafe-best-match --prerelease=allow -r requirements-cuda.txt
        if (-not $?) { Write-Host "Package installation failed."; Pause; exit 1 }
        Write-Host ""
        Write-Host "Setup complete! Activate with:"
        Write-Host "  conda activate autoqsar-py311"
    }
    "2" {
        Write-Host ""
        Write-Host "Creating CPU-only environment..."
        & conda env create -f environment-cpu.yml -n autoqsar-py311
        if (-not $?) { Write-Host "Conda env creation failed."; Pause; exit 1 }
        Write-Host ""
        Write-Host "Installing pip packages via uv (this may take several minutes)..."
        & conda run -n autoqsar-py311 pip install uv
        & conda run -n autoqsar-py311 uv pip install --system --index-strategy unsafe-best-match --prerelease=allow -r requirements-cpu.txt
        if (-not $?) { Write-Host "Package installation failed."; Pause; exit 1 }
        Write-Host ""
        Write-Host "Setup complete! Activate with:"
        Write-Host "  conda activate autoqsar-py311"
    }
    "3" {
        Write-Host ""
        Write-Host "Creating base environment (without PyTorch)..."
        & conda env create -f environment-base.yml -n autoqsar-py311
        Write-Host ""
        Write-Host "Next steps:"
        Write-Host "  1. Activate: conda activate autoqsar-py311"
        Write-Host "  2. Install PyTorch manually from https://pytorch.org/get-started/locally/"
        Write-Host "  3. Then run: pip install uv; uv pip install --system -r requirements-cpu.txt"
    }
    default {
        Write-Host "Invalid choice. Please run this script again."
    }
}

Write-Host ""
Pause
