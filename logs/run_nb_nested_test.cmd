@echo off
set PATH=C:\Users\scott\.conda\envs\autoqsar-py311;C:\Users\scott\.conda\envs\autoqsar-py311\Scripts;C:\Users\scott\.conda\envs\autoqsar-py311\Library\bin;%PATH%
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=6
cd /d C:\Users\scott\AppData\Local\Temp\claude\c--Users-scott-AutoQSAR\38c1a67d-2ccf-492f-9c39-630f7a300f36\scratchpad\nbtest
python run_nb_nested.py freesolv_sampl.csv freesolv freesolv > freesolv_driver.log 2>&1
python run_nb_nested.py tdc_caco2_wang.csv caco2 caco2 > caco2_driver.log 2>&1
echo NBTEST_DONE > nbtest_done.txt
