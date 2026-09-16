@echo off
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe -m streamlit run app\streamlit_app.py --server.address localhost
) else (
  python -m streamlit run app\streamlit_app.py --server.address localhost
)
pause
