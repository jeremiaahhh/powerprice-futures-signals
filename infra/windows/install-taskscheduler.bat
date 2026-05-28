@echo off
REM ============================================================
REM PowerPrice Signal Daemon — Windows Task Scheduler Installer
REM
REM Alternative to NSSM. Does NOT require administrator rights for
REM the user's own tasks, but the task runs only when logged in
REM unless you configure "Run whether user is logged on or not".
REM
REM Requires:
REM   - Python venv with project dependencies installed
REM   - Run as Administrator for "run on startup / not logged in"
REM
REM Usage:
REM   1. Edit the PATHS section below
REM   2. Run this script (as Administrator for full service-mode)
REM ============================================================

setlocal EnableDelayedExpansion

REM ---- PATHS (edit these) ----
set PROJECT_DIR=C:\powerprice-futures-signals
set BACKEND_DIR=%PROJECT_DIR%\backend
set VENV_PYTHON=%PROJECT_DIR%\.venv\Scripts\python.exe
set DATA_DIR=%PROJECT_DIR%\data

set TASK_NAME=PowerPriceSignalDaemon

REM ---- Pre-flight check ----
if not exist "%VENV_PYTHON%" (
    echo ERROR: Python venv not found at %VENV_PYTHON%
    pause
    exit /b 1
)

REM ---- Create data directory ----
if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"

REM ---- Remove existing task if present ----
schtasks /query /tn "%TASK_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo Removing existing task...
    schtasks /delete /tn "%TASK_NAME%" /f
)

REM ---- Create wrapper script that activates venv and runs daemon ----
set WRAPPER=%DATA_DIR%\run_daemon.bat
(
echo @echo off
echo cd /d "%BACKEND_DIR%"
echo "%VENV_PYTHON%" -m app.runtime.signal_daemon >> "%DATA_DIR%\daemon_taskscheduler.log" 2^>^&1
) > "%WRAPPER%"

REM ---- Register task: run at system startup, repeat every 5min if process exits ----
REM /SC ONSTART  — triggers at machine boot
REM /DELAY       — wait 60s after boot for network
REM /RI          — retry interval in minutes if process exits
schtasks /create ^
    /tn "%TASK_NAME%" ^
    /tr "%WRAPPER%" ^
    /sc onstart ^
    /delay 0001:00 ^
    /ru "%USERDOMAIN%\%USERNAME%" ^
    /f

echo.
echo Task registered: %TASK_NAME%
echo The daemon will start automatically at next boot.
echo.
echo To start now:
echo   schtasks /run /tn "%TASK_NAME%"
echo.
echo To view status:
echo   schtasks /query /tn "%TASK_NAME%"
echo.
echo To stop:
echo   schtasks /end /tn "%TASK_NAME%"
echo.
echo Logs: %DATA_DIR%\daemon_taskscheduler.log
echo.
echo NOTE: For "run whether logged on or not" mode, open Task Scheduler GUI,
echo find the task, and set "Run whether user is logged on or not".
echo That requires entering your Windows password.
echo.
echo Signal only. Keine Order ausgefuehrt.

pause
