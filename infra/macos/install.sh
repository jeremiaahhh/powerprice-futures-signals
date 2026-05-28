#!/usr/bin/env bash
# Install PowerPrice Signal Daemon as a macOS LaunchAgent.
# The agent survives OS sleep — launchd restarts it automatically after wake.
#
# Usage:
#   cd /path/to/powerprice-futures-signals
#   bash infra/macos/install.sh [--venv /path/to/venv]
#
# Requirements: Python venv with project dependencies installed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKEND_DIR="$PROJECT_DIR/backend"
DATA_DIR="$PROJECT_DIR/data"
PLIST_LABEL="com.powerprice.signal-daemon"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
PLIST_SRC="$SCRIPT_DIR/$PLIST_LABEL.plist"
PLIST_DST="$LAUNCH_AGENTS_DIR/$PLIST_LABEL.plist"

# Parse optional --venv argument
VENV_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv) VENV_DIR="$2"; shift 2;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

# Auto-detect venv
if [[ -z "$VENV_DIR" ]]; then
    for candidate in "$PROJECT_DIR/.venv" "$PROJECT_DIR/venv" "$BACKEND_DIR/.venv" "$BACKEND_DIR/venv"; do
        if [[ -f "$candidate/bin/python" ]]; then
            VENV_DIR="$candidate"
            break
        fi
    done
fi

if [[ -z "$VENV_DIR" || ! -f "$VENV_DIR/bin/python" ]]; then
    echo "ERROR: Python venv not found. Create one first:"
    echo "  python3 -m venv $PROJECT_DIR/.venv && $PROJECT_DIR/.venv/bin/pip install -r $BACKEND_DIR/requirements.txt"
    exit 1
fi

PYTHON_BIN="$VENV_DIR/bin/python"
VENV_BIN_DIR="$VENV_DIR/bin"

echo "=== PowerPrice Signal Daemon — macOS LaunchAgent Installer ==="
echo "Project:  $PROJECT_DIR"
echo "Backend:  $BACKEND_DIR"
echo "Python:   $PYTHON_BIN"
echo "Data dir: $DATA_DIR"

# Create data directory if missing
mkdir -p "$DATA_DIR"

# Stop existing agent if loaded
if launchctl list | grep -q "$PLIST_LABEL" 2>/dev/null; then
    echo "Stopping existing agent..."
    launchctl unload "$PLIST_DST" 2>/dev/null || true
fi

# Create LaunchAgents dir if missing
mkdir -p "$LAUNCH_AGENTS_DIR"

# Substitute placeholders into plist
sed \
    -e "s|PLACEHOLDER_BACKEND_DIR|$BACKEND_DIR|g" \
    -e "s|PLACEHOLDER_PYTHON|$PYTHON_BIN|g" \
    -e "s|PLACEHOLDER_VENV_BIN|$VENV_BIN_DIR|g" \
    -e "s|PLACEHOLDER_DATA_DIR|$DATA_DIR|g" \
    "$PLIST_SRC" > "$PLIST_DST"

echo "Plist written to: $PLIST_DST"

# Load the agent
launchctl load -w "$PLIST_DST"

echo ""
echo "SUCCESS. The daemon is now running as a LaunchAgent."
echo ""
echo "Commands:"
echo "  Status:   launchctl list $PLIST_LABEL"
echo "  Logs:     tail -f $DATA_DIR/daemon_launchd.err.log"
echo "  Stop:     launchctl unload $PLIST_DST"
echo "  Uninstall: bash $SCRIPT_DIR/uninstall.sh"
echo ""
echo "Signal only. Keine Order ausgeführt."
