#!/usr/bin/env bash
# Make sure we use the project virtual environment python if present
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

if [ -f "$DIR/.venv/bin/python" ]; then
    PYTHON_EXEC="$DIR/.venv/bin/python"
elif [ -f "$DIR/../.venv/bin/python" ]; then
    PYTHON_EXEC="$DIR/../.venv/bin/python"
else
    PYTHON_EXEC="python3"
fi

echo "========================================================================="
echo "   Launching Stock Market Prediction Academic Showcase Dashboard...      "
echo "========================================================================="
exec "$PYTHON_EXEC" "$DIR/run_dashboard.py" "$@"
