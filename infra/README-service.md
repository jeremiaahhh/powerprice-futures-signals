# PowerPrice Signal Daemon — Background Service Setup

The daemon runs as a **permanent background service** that:
- Starts automatically at boot
- Survives OS sleep / hibernate (wake-from-sleep detection built in)
- Uses an extended polling interval when running on battery power
- Restarts itself after crashes

**SIGNAL ONLY. Keine Live-Orders. Keine Broker-API. Keine automatische Handelsausführung.**

---

## How sleep-resilience works

The daemon's sleep loop compares wall-clock time against monotonic time after each 10-second chunk. If the wall clock jumped by more than expected (configurable via `WAKE_CLOCK_JUMP_THRESHOLD_SECONDS`, default 30 s), it knows the OS was sleeping. It then:

1. Waits a short grace period (`POST_WAKE_GRACE_SECONDS`, default 15 s) for network/database to reconnect
2. Immediately runs the next signal cycle instead of waiting for the remaining interval

The asyncpg engine uses `pool_pre_ping=True`, so stale DB connections are silently replaced after wake.

## Battery / energy-saving mode

When `psutil` detects the system is on battery (not plugged in), the daemon automatically extends its polling interval to `SIGNAL_LOOP_INTERVAL_BATTERY_MINUTES` (default 30 min instead of 15 min). Add to `backend/.env`:

```
SIGNAL_LOOP_INTERVAL_BATTERY_MINUTES=30
WAKE_DETECTION_ENABLED=true
POST_WAKE_GRACE_SECONDS=15
WAKE_CLOCK_JUMP_THRESHOLD_SECONDS=30.0
```

---

## macOS — LaunchAgent

LaunchAgents run under your user account, survive sleep, and restart automatically after crash.

### Install

```bash
cd /path/to/powerprice-futures-signals

# Create venv if needed
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt

# Install and start the agent
bash infra/macos/install.sh
```

The installer auto-detects `.venv/` in the project root. To specify a custom venv:

```bash
bash infra/macos/install.sh --venv /path/to/custom/venv
```

### Status and logs

```bash
# Check if running
launchctl list com.powerprice.signal-daemon

# Daemon application log (structured)
tail -f data/daemon_run.log

# LaunchAgent stdout/stderr
tail -f data/daemon_launchd.err.log
```

### Stop / Uninstall

```bash
# Temporary stop (will restart at next boot)
launchctl unload ~/Library/LaunchAgents/com.powerprice.signal-daemon.plist

# Permanent uninstall
bash infra/macos/uninstall.sh
```

---

## Windows — NSSM (recommended)

NSSM (Non-Sucking Service Manager) installs the daemon as a proper Windows Service that starts at boot, runs without being logged in, and restarts on failure.

### Install NSSM

Download `nssm.exe` from https://nssm.cc/download and place it in `C:\nssm\` or add it to `PATH`.

### Install the service

1. Edit `infra/windows/install-nssm.bat` — set `PROJECT_DIR` and `VENV_PYTHON` to match your installation
2. Right-click → **Run as administrator**

```
infra\windows\install-nssm.bat
```

### Status and logs

```
# Check service status
sc query PowerPriceSignalDaemon

# Application log (structured)
type data\daemon_run.log

# NSSM stdout/stderr
type data\daemon_nssm.err.log
```

### Stop / Uninstall

```
# Stop without removing
sc stop PowerPriceSignalDaemon

# Full uninstall
infra\windows\uninstall.bat
```

---

## Windows — Task Scheduler (no NSSM)

Simpler alternative — does not require NSSM but has fewer restart guarantees.

1. Edit `infra/windows/install-taskscheduler.bat` — set `PROJECT_DIR`
2. Run as Administrator
3. Open **Task Scheduler GUI** → find `PowerPriceSignalDaemon` → Properties → check **"Run whether user is logged on or not"**

```
infra\windows\install-taskscheduler.bat
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `daemon_health.json` not found | Check `DAEMON_STATE_FILE` in `backend/.env` points to a writable path |
| Database connection failed after wake | Wait for grace period (15 s default) — `pool_pre_ping` handles reconnection |
| Daemon exits immediately on macOS | Check `data/daemon_launchd.err.log` — usually a missing dependency or wrong venv path |
| High CPU on battery | Increase `SIGNAL_LOOP_INTERVAL_BATTERY_MINUTES` in `backend/.env` |
| Wake not detected | Reduce `WAKE_CLOCK_JUMP_THRESHOLD_SECONDS` (default 30 s) |

---

## Security reminder

All outputs are **signals only**. The system never places real orders, never connects to a broker API, and never executes trades automatically.

`SIGNAL_ONLY=true` must remain set in all environment files at all times.
