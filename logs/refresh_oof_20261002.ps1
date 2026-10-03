Set-Location "C:\Users\scott\AutoQSAR"
& .\tools\run_oof_ensemble_rtx.ps1 -Python "C:\Users\scott\.conda\envs\autoqsar-py311\python.exe" *>&1 | Out-File -Encoding utf8 logs/oof_ensemble_rtx_refresh_wrapper.log
