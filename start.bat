@echo off
setlocal enableextensions

REM ============================================================
REM   FaceMatching - service launcher + keep-alive loop
REM ------------------------------------------------------------
REM   - Launches FaceMatching.exe and keeps it running.
REM   - If the exe ever stops/crashes, it restarts it.
REM   - RUNS ONLY DURING THE 02:00-06:00 WINDOW. Outside that
REM     window the loop exits instead of relaunching the exe.
REM   - Pair this with two Task Scheduler tasks:
REM       * "AIPhotoMatch-Start" -> Daily at 02:00 (runs this file)
REM       * "AIPhotoMatch-Stop"  -> Daily at 06:00 (runs stop.bat)
REM     The Stop task is what actually ends a long-running batch
REM     at 06:00; the time check below is only a safety net so the
REM     loop never relaunches the exe outside the window.
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
REM --- Only run inside the 02:00-06:00 window. ----------------
REM %time% can be " 9:05:..." (leading space) -> turn space into
REM a zero, take the hour, then force base-10 (1HH-100) so hours
REM like 08/09 are not mis-read as invalid octal.
for /f "tokens=1 delims=:" %%h in ("%time: =0%") do set "HHRAW=%%h"
set /a HH=1%HHRAW% - 100
if %HH% LSS 2 goto outside
if %HH% GEQ 6 goto outside

echo [%date% %time%] Starting FaceMatching service... >> "%LAUNCH_LOG%"
"%EXE%"
echo [%date% %time%] FaceMatching stopped (exit code %errorlevel%). Restarting in 15s... >> "%LAUNCH_LOG%"
timeout /t 15 /nobreak >nul
goto loop

:outside
echo [%date% %time%] Outside 02:00-06:00 window (hour=%HH%). Launcher exiting. >> "%LAUNCH_LOG%"
exit /b 0
