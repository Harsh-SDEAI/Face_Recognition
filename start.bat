@echo off
setlocal enableextensions

REM ============================================================
REM   FaceMatching - service launcher + keep-alive loop
REM ------------------------------------------------------------
REM   - Launches FaceMatching.exe and keeps it running.
REM   - If the exe ever stops/crashes, it restarts it.
REM
REM   NOTE: Production scheduling (the 02:00-06:00 window) is now
REM   handled NATIVELY by Task Scheduler, which runs the EXE
REM   directly with a 4-hour execution-time-limit and
REM   restart-on-failure. This script is NOT used by the
REM   scheduled task - it is kept only as a convenient way to
REM   launch the exe by hand for testing.
REM
REM   DEPLOYMENT LAYOUT this script expects:
REM     C:\releasebuilds\AIPhotoMatch\
REM         AIPhtoMatch\
REM             start.bat              <-- this file
REM             .env                  <-- DB creds, thresholds
REM             AIPhotoMatch2026.exe
REM             _internal\
REM         logs\                      <-- one level up, already exists
REM             archive\
REM             error.log
REM             info.log
REM ============================================================

REM --- Always run from this script's own folder, so .env and ---
REM --- exe resolve relative to HERE, not C:\Windows\System32 ---
cd /d "%~dp0"

set "EXE=AIPhotoMatch2026.exe"
set "LAUNCH_LOG=..\logs\launcher.log"

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
