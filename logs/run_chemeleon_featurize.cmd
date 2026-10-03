@echo off
cd /d C:\Users\scott\AutoQSAR
set PYTHONPATH=.
set OMP_NUM_THREADS=2
"C:\Users\scott\.conda\envs\autoqsar-py311\python.exe" -W ignore -m qsarena.feature_expansion.featurize --families chemeleon --workers 1 > logs\feature_expansion_chemeleon.log 2>&1
echo DONE >> logs\feature_expansion_chemeleon.log
