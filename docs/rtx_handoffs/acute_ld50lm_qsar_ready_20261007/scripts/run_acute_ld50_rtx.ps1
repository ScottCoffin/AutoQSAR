param(
    [string]$Python = "C:\Users\Scott.Coffin\AppData\Local\miniconda3\envs\autoqsar-py311\python.exe",
    [string]$OutputDir = "benchmark_results\acute_ld50lm_qsar_ready_rtx_full_20261007",
    [switch]$Detached
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")
$InputCsv = Join-Path $RepoRoot "docs\rtx_handoffs\acute_ld50lm_qsar_ready_20261007\input\full_dataset_qsar_ready_ld50lm.csv"
$LogPath = Join-Path $RepoRoot "logs\acute_ld50lm_qsar_ready_rtx_full_20261007.log"
$CmdPath = Join-Path $RepoRoot "logs\run_acute_ld50lm_qsar_ready_rtx_full_20261007.cmd"

$RunnerArgs = @(
    "-m", "portable_colab_qsar_bundle.run_qsarena_benchmarks",
    "--dataset", $InputCsv,
    "--smiles-col", "QSAR_READY_SMILES",
    "--target-col", "LD50_LM",
    "--task", "regression",
    "--target-transform", "raw",
    "--benchmark-profile", "full",
    "--output-dir", $OutputDir,
    "--use-gpu", "true",
    "--run-chemprop-mpnn",
    "--run-chemprop-dmpnn",
    "--run-chemprop-cmpnn",
    "--run-chemprop-attentivefp",
    "--run-chemprop-rdkit2d",
    "--run-chemprop-selected-features",
    "--chemprop-epochs", "40",
    "--chemprop-ensemble-size", "3",
    "--chemprop-random-seed", "42",
    "--run-unimol-v1",
    "--run-unimol-v2",
    "--unimol-batch-size", "32",
    "--unimol-max-atoms", "64",
    "--run-tabpfn",
    "--tabpfn-max-train-rows", "11000",
    "--no-run-tdc22-multiseed-best"
)

if (!(Test-Path -LiteralPath $Python)) {
    throw "Python interpreter not found: $Python"
}
if (!(Test-Path -LiteralPath $InputCsv)) {
    throw "Input CSV not found: $InputCsv"
}

if ($Detached) {
    $quotedArgs = ($RunnerArgs | ForEach-Object { '"' + ($_ -replace '"', '\"') + '"' }) -join " "
    $cmd = @(
        "@echo off",
        "cd /d `"$RepoRoot`"",
        "`"$Python`" $quotedArgs > `"$LogPath`" 2>&1",
        "echo LAUNCHER_DONE %DATE% %TIME% >> `"$LogPath`""
    ) -join "`r`n"
    Set-Content -LiteralPath $CmdPath -Value $cmd -Encoding ASCII
    $result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = "cmd.exe /c `"$CmdPath`""}
    $result
    Write-Host "Detached RTX run launched. Log: $LogPath"
} else {
    Set-Location -LiteralPath $RepoRoot
    & $Python @RunnerArgs 2>&1 | Tee-Object -FilePath $LogPath
}
