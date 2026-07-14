@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "PYTHON=.venv314\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo Error: project Python environment was not found at %PYTHON%.
  echo Recreate it with: py -3.14 -m venv .venv314
  exit /b 1
)

"%PYTHON%" -u -m src.main daily-signal --engine v3 --debug %*
