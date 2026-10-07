@echo off
set PATH=C:\Users\scott\.conda\envs\autoqsar-py311;C:\Users\scott\.conda\envs\autoqsar-py311\Scripts;C:\Users\scott\.conda\envs\autoqsar-py311\Library\bin;%PATH%
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=6
cd /d C:\Users\scott\AppData\Local\Temp\claude\c--Users-scott-AutoQSAR\38c1a67d-2ccf-492f-9c39-630f7a300f36\scratchpad\nbtest
if exist cls_done.txt del cls_done.txt
python C:\Users\scott\AutoQSAR\tools\notebook_nested_check.py tdc_hia_hou.csv hia hia > hia_driver.log 2>&1
python C:\Users\scott\AutoQSAR\tools\notebook_nested_check.py tdc_bbb_martins.csv bbb bbb > bbb_driver.log 2>&1
python C:\Users\scott\AutoQSAR\tools\notebook_nested_check.py freesolv_sampl.csv freesolv freesolv2 > freesolv2_driver.log 2>&1
echo DONE > cls_done.txt
