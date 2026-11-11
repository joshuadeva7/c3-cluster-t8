# Sourced by the mac/*.sh scripts: activate the demo's Python venv (system python3 3.11).
VENV="${VENV:-$HOME/swarm-live/.venv}"
if [ ! -f "$VENV/bin/activate" ]; then echo "[venv] no venv at $VENV: run  bash mac/setup_mac.sh  (or set VENV=...)"; return 1 2>/dev/null || exit 1; fi
source "$VENV/bin/activate"
