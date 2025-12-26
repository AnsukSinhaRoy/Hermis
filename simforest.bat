@echo off
REM simforest - Hermis experiment launcher (Windows wrapper)
REM Usage:
REM   simforest --newconfig.yaml
REM   simforest configs\newconfig.yaml

setlocal

REM Run the Python launcher sitting next to this .bat file.
set SCRIPT_DIR=%~dp0

python "%SCRIPT_DIR%simforest" %*

endlocal