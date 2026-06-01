@echo off
setlocal enableextensions

REM ============================================================
REM   FaceMatching - service teardown (runs at 06:00 daily)
REM ------------------------------------------------------------
REM   Killing the exe alone is NOT enough: start.bat is a
REM   keep-alive loop and would relaunch it. So we:
REM     1) End the Start task   -> stops the start.bat loop
REM     2) Kill the exe + tree  -> stops any in-flight batch
REM
REM   Schedule this with Task Scheduler:
REM     "AIPhotoMatch-Stop" -> Daily at 06:00
REM ============================================================

cd /d "%~dp0"
set "LAUNCH_LOG=..\logs\launcher.log"

echo [%date% %time%] STOP requested. Ending Start task and killing exe... >> "%LAUNCH_LOG%"

REM 1) Stop the keep-alive loop (ignore error if not running)
schtasks /End /TN "AIPhotoMatch-Start" >nul 2>&1

REM 2) Kill the exe and any child processes it spawned
taskkill /F /IM AIPhotoMatch2026.exe /T >nul 2>&1

echo [%date% %time%] STOP complete. >> "%LAUNCH_LOG%"
exit /b 0
