@echo off
cd /d C:\Users\scott\AutoQSAR
set PYTHONPATH=.
set PYTHONIOENCODING=utf-8
"C:\Users\scott\.conda\envs\autoqsar-py311\python.exe" -W ignore -m qsarena.feature_expansion.featurize --families unimol_repr --workers 1 > logs\feature_expansion_featurize_unimol_repr.log 2>&1
echo DONE >> logs\feature_expansion_featurize_unimol_repr.log
