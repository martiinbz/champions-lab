@echo off
cd /d "%~dp0"
set PY=python
if exist .venv\Scripts\python.exe set PY=.venv\Scripts\python.exe
%PY% -m champions.cli export-web
if errorlevel 1 (
  pause
  exit /b 1
)
start "" "%~dp0web\index.html"
