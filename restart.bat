@echo off
powershell -Command "Get-Process -Id (Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue).OwningProcess | Stop-Process -Force -ErrorAction SilentlyContinue"
streamlit cache clear
streamlit run main.py