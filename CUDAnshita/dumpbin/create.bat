@echo off
set dir="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin"
for %%i in (%dir%\*.dll) do (
echo %%~ni%%~xi
dumpbin /exports "%%i" > %%~ni.txt
)
