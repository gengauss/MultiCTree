@echo off

set result=0426
set train=0
set end=99

for /L %%i in (1,1,3) do (
    start %~dp0\make_evalgraph.bat %%i %result% 1
    start %~dp0\make_evalgraph.bat %%i %result% 2
    
)

pause