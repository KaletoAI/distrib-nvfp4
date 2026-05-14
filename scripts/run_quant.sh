#!/bin/bash
# Wrapper that runs distrib_quant.py with logging.
# All CLI args are passed through to the driver — e.g.:
#   tmux new -d -s quant '/path/to/run_quant.sh \
#       --source-model /path/to/BF16 \
#       --output-dir /path/to/NVFP4'
#
# Env vars (with defaults):
#   PYTHON    — python interpreter to use (default: python3, must have modelopt + ray + transformers)
#   DRIVER    — path to distrib_quant.py (default: alongside this wrapper)
#   LOG       — log file path (default: /tmp/distrib_quant.log)
set -o pipefail

SCRIPT_DIR=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
PYTHON=${PYTHON:-python3}
DRIVER=${DRIVER:-"$SCRIPT_DIR/distrib_quant.py"}
LOG=${LOG:-/tmp/distrib_quant.log}

echo "=== run started: $(date -Iseconds) on $(hostname) ===" > "$LOG"
echo "python: $PYTHON" | tee -a "$LOG"
echo "driver: $DRIVER" | tee -a "$LOG"
echo "args:   $*"      | tee -a "$LOG"

"$PYTHON" -u "$DRIVER" "$@" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}

echo "=== run finished: $(date -Iseconds) RC=$RC ===" | tee -a "$LOG"
exit $RC
