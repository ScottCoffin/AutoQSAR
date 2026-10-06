@echo off
cd /d C:\Users\scott\AutoQSAR
set OMP_NUM_THREADS=6
"C:\Users\scott\.conda\envs\autoqsar-py311\python.exe" -W ignore portable_colab_qsar_bundle\render_manuscript_assets.py --run-dir benchmark_results/qsarena_benchmark_oof_ensemble > logs\render_20261006.log 2>&1
echo RENDER_EXIT=%ERRORLEVEL% >> logs\render_20261006.log
