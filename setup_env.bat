@echo off
REM AutoQSAR Environment Setup Script

echo ========================================
echo AutoQSAR Conda Environment Setup
echo ========================================
echo.
echo Choose your setup option:
echo 1. CUDA GPU (recommended if you have NVIDIA GPU with CUDA 12.1+)
echo 2. CPU-only (for laptops or non-GPU systems)
echo 3. Custom (manual PyTorch installation)
echo.

set /p choice="Enter your choice (1, 2, or 3): "

if "%choice%"=="1" (
    echo.
    echo Creating CUDA environment...
    conda env create -f environment-cuda.yml -n autoqsar-py311
    if errorlevel 1 goto :error
    echo.
    echo Installing pip packages via uv (this may take several minutes)...
    conda run -n autoqsar-py311 pip install uv
    conda run -n autoqsar-py311 uv pip install --system --index-strategy unsafe-best-match --prerelease=allow -r requirements-cuda.txt
    if errorlevel 1 goto :error
    echo.
    echo Setup complete! Activate with:
    echo   conda activate autoqsar-py311
) else if "%choice%"=="2" (
    echo.
    echo Creating CPU-only environment...
    conda env create -f environment-cpu.yml -n autoqsar-py311
    if errorlevel 1 goto :error
    echo.
    echo Installing pip packages via uv (this may take several minutes)...
    conda run -n autoqsar-py311 pip install uv
    conda run -n autoqsar-py311 uv pip install --system --index-strategy unsafe-best-match --prerelease=allow -r requirements-cpu.txt
    if errorlevel 1 goto :error
    echo.
    echo Setup complete! Activate with:
    echo   conda activate autoqsar-py311
) else if "%choice%"=="3" (
    echo.
    echo Creating base environment (without PyTorch)...
    conda env create -f environment-base.yml -n autoqsar-py311
    echo.
    echo Next steps:
    echo   1. Activate: conda activate autoqsar-py311
    echo   2. Install PyTorch manually from https://pytorch.org/get-started/locally/
    echo   3. Then run: pip install uv ^&^& uv pip install --system --index-strategy unsafe-best-match --prerelease=allow -r requirements-cpu.txt
) else (
    echo Invalid choice. Please run this script again.
)

goto :end

:error
echo.
echo Setup failed. Check the error messages above.

:end
pause
