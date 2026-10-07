@echo off
cd /d C:\Users\scott\AutoQSAR
set PYTHONPATH=.
set OMP_NUM_THREADS=2
python -W ignore -m qsarena.feature_expansion.train --feature-set admetboost --cv-strategy scaffold --cv-only --n-jobs 2 >> logs\feature_expansion_scaffoldcv.log 2>&1
echo DONE >> logs\feature_expansion_scaffoldcv.log
