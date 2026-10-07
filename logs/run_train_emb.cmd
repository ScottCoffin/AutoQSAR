@echo off
cd /d C:\Users\scott\AutoQSAR
set PYTHONPATH=.
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=2
"C:\Users\scott\.conda\envs\autoqsar-py311\python.exe" -W ignore -m qsarena.feature_expansion.train --feature-set admetboost+emb --models xgboost --device cuda --n-jobs 2 > logs\feature_expansion_train_admetboost_emb.log 2>&1
"C:\Users\scott\.conda\envs\autoqsar-py311\python.exe" -W ignore -m qsarena.feature_expansion.train --feature-set emb --models xgboost --device cuda --n-jobs 2 > logs\feature_expansion_train_emb.log 2>&1
echo DONE >> logs\feature_expansion_train_emb.log
