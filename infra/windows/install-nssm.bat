@echo off
REM ============================================================
REM PowerPrice Signal Daemon — Windows Service Installer (NSSM)
REM
REM Requires:
REM   - NSSM (Non-Sucking Service Manager): https://nssm.cc/download
REM   - Python venv with project dependencies installed
REM   - Run this script as Administrator
REM
REM Usage:
REM   1. Download nssm.exe and place it in C:\nssm\ or add to PATH
REM   2. Edit the PATHS section below to match your installation
REM   3. Right-click this file → "Run as administrator"
REM ============================================================

setlocal EnableDelayedExpansion

REM ---- PATHS (edit these) ----
set PROJECT_DIR=C:\powerprice-futures-signals
set BACKEND_DIR=%PROJECT_DIR%\backend
set VENV_PYTHON=%PROJECT_DIR%\.venv\Scripts\python.exe
set DATA_DIR=%PROJECT_DIR%\data
set NSSM=nssm
REM If nssm.exe is not in PATH, set the full path:
REM set NSSM=C:\nssm\nssm.exe

set SERVICE_NAME=PowerPriceSignalDaemon
set DISPLAY_NAME=PowerPrice Signal Daemon
set DESCRIPTION=PowerPrice Futures Signal generation daemon. SIGNAL ONLY — no live orders.

REM ---- Pre-flight checks ----
if not exist "%VENV_PYTHON%" (
    echo ERROR: Python venv not found at %VENV_PYTHON%
    echo Create one: python -m venv %PROJECT_DIR%\.venv
    echo Install deps: %PROJECT_DIR%\.venv\Scripts\pip install -r %BACKEND_DIR%\requirements.txt
    pause
    exit /b 1
)

where %NSSM% >nul 2>&1
if errorlevel 1 (
    echo ERROR: nssm.exe not found in PATH.
    echo Download from https://nssm.cc/download and add to PATH or set NSSM= to full path.
    pause
    exit /b 1
)

REM ---- Stop and remove existing service if present ----
%NSSM% status %SERVICE_NAME% >nul 2>&1
if not errorlevel 1 (
    echo Stopping existing service...
    %NSSM% stop %SERVICE_NAME% confirm
    %NSSM% remove %SERVICE_NAME% confirm
)

REM ---- Create data directory ----
if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"

REM ---- Install service ----
echo Installing service: %SERVICE_NAME%

%NSSM% install %SERVICE_NAME% "%VENV_PYTHON%" "-m" "app.runtime.signal_daemon"
%NSSM% set %SERVICE_NAME% DisplayName "%DISPLAY_NAME%"
%NSSM% set %SERVICE_NAME% Description "%DESCRIPTION%"
%NSSM% set %SERVICE_NAME% AppDirectory "%BACKEND_DIR%"
%NSSM% set %SERVICE_NAME% AppEnvironmentExtra "PYTHONUNBUFFERED=1"

REM Stdout/Stderr logs
%NSSM% set %SERVICE_NAME% AppStdout "%DATA_DIR%\daemon_nssm.out.log"
%NSSM% set %SERVICE_NAME% AppStderr "%DATA_DIR%\daemon_nssm.err.log"
%NSSM% set %SERVICE_NAME% AppRotateFiles 1
%NSSM% set %SERVICE_NAME% AppRotateBytes 10485760

REM Restart policy: restart 30s after failure
%NSSM% set %SERVICE_NAME% AppRestartDelay 30000

REM Start type: Automatic (start on boot)
%NSSM% set %SERVICE_NAME% Start SERVICE_AUTO_START

REM ---- Start service ----
echo Starting service...
%NSSM% start %SERVICE_NAME%

echo.
echo SUCCESS. Service installed and started.
echo.
echo Commands:
echo   Status:   sc query %SERVICE_NAME%
echo   Logs:     type "%DATA_DIR%\daemon_nssm.err.log"
echo   Stop:     sc stop %SERVICE_NAME%
echo   Uninstall: infra\windows\uninstall.bat
echo.
echo Signal only. Keine Order ausgefuehrt.

pause
