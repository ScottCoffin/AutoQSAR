@echo off
cd /d C:\Users\scott\AutoQSAR
set PYTHONPATH=.
set PYTHONIOENCODING=utf-8
python -m qsarena.feature_expansion.selection_leak >> logs\selection_leak_20261003.log 2>&1
echo EXIT=%ERRORLEVEL% >> logs\selection_leak_20261003.log
