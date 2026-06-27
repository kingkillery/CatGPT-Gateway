@echo off
REM Single end-to-end verification pass for the CatGPT gateway.
REM Invoked by the "CatGPT-E2E-Watch" scheduled task every 5 minutes.
cd /d C:\dev\desktop-projects\CatGPT-Gateway
"C:\Python314\python.exe" scripts\e2e_watch.py --once --timeout 600
