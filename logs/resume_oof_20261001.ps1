Set-Location "C:\Users\scott\AutoQSAR"
$ds = @("tdc_cyp2c9_veith","tdc_cyp1a2_veith","tdc_cyp2c19_veith","tdc_cyp2d6_veith","tdc_cyp3a4_veith","tdc_herg_karim")
& .\tools\run_oof_ensemble_rtx.ps1 -Python "C:\Users\scott\.conda\envs\autoqsar-py311\python.exe" -Datasets $ds *>&1 | Out-File -Encoding utf8 logs/oof_ensemble_rtx_resume2_wrapper.log
