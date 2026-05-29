@echo off
setlocal enableextensions

REM ============================================================
REM   FaceMatching - service launcher + keep-alive loop
REM ------------------------------------------------------------
REM   - Launches FaceMatching.exe and keeps it running.
REM   - If the exe ever stops/crashes, it restarts it.
REM   - Pair this with Task Scheduler ("At startup") so the
REM     service also comes back automatically after a reboot.
REM
REM   DEPLOYMENT LAYOUT this script expects:
REM     C:\releasebuilds\AIPhotoMatch\AIPhtoMatch\
REM         start.bat              <-- this file
REM         .env                  <-- DB creds, thresholds
REM         AIPhotoMatch2026.exe
REM         _internal\
REM         logs\                 <-- auto-created
REM         log_archive\          <-- auto-created
REM ============================================================

REM --- Always run from this script's own folder, so .env and ---
REM --- the logs\ folder resolve relative to HERE, not C:\Windows\System32 ---
cd /d "%~dp0"

set "EXE=AIPhotoMatch2026.exe"
set "LAUNCH_LOG=logs\launcher.log"

if not exist "logs" mkdir "logs"

if not exist "%EXE%" (
    echo [%date% %time%] ERROR: %EXE% not found. Check the deployment layout. >> "%LAUNCH_LOG%"
    echo ERROR: %EXE% not found. Check the deployment layout.
    pause
    exit /b 1
)

:loop
echo [%date% %time%] Starting FaceMatching service... >> "%LAUNCH_LOG%"
"%EXE%"
echo [%date% %time%] FaceMatching stopped (exit code %errorlevel%). Restarting in 15s... >> "%LAUNCH_LOG%"
timeout /t 15 /nobreak >nul
goto loop
