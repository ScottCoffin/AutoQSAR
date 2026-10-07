# 2026-10-06: after the Tox21 ensemble rebuild finishes, give tdc_herg and tdc_bioavailability_ma nested CV metrics for
# the selected-feature models that have no saved predictions (metrics-only path; they were never ensemble members).
Set-Location C:\Users\scott\AutoQSAR
while (-not (Select-String -Path logs\nested_selection_tox21_20261005.log -Pattern "LAUNCHER_DONE" -Quiet)) { Start-Sleep -Seconds 60 }
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File logs\run_nested_selection.ps1 -Datasets "tdc_herg,tdc_bioavailability_ma" -Tag followup
Add-Content logs\nested_selection_followup_20261005.log "FOLLOWUPS_DONE"
