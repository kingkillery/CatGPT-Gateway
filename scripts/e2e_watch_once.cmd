@echo off
REM Single end-to-end verification pass for the CatGPT gateway.
REM Run manually, or from a disabled/on-demand "CatGPT-E2E-Watch" scheduled task.
cd /d C:\dev\desktop-projects\CatGPT-Gateway
"C:\Python314\python.exe" scripts\e2e_watch.py --once --timeout 600
