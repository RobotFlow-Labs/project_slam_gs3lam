#!/usr/bin/env bash
# release_validate.sh — GS3LAM release validation entrypoint
#
# Usage:
#   bash scripts/release_validate.sh [REPORT_JSON]
#
# Steps:
#   1. Activate project venv
#   2. Run release gate unit tests
#   3. Run release checks against eval report (if provided)
#   4. Exit with appropriate code
#
# Exit codes:
#   0 — all gates passed
#   1 — test or gate failure
#   2 — environment / setup error

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPORT_JSON="${1:-}"

echo "=============================================="
echo "  GS3LAM Release Validation"
echo "=============================================="
echo "  Project: $PROJECT_DIR"
echo "  Report:  ${REPORT_JSON:-'(none — tests only)'}"
echo "=============================================="
echo ""

# -------------------------------------------------------
# Step 0: Activate venv
# -------------------------------------------------------

VENV="$PROJECT_DIR/.venv"
if [ ! -d "$VENV" ]; then
    echo "ERROR: .venv not found at $VENV"
    echo "Run: cd $PROJECT_DIR && uv venv .venv && uv sync"
    exit 2
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

PYTHON_PATH="$(which python)"
if [[ "$PYTHON_PATH" != *".venv"* ]]; then
    echo "ERROR: Python not from venv: $PYTHON_PATH"
    exit 2
fi

echo "[OK] venv activated: $PYTHON_PATH"
echo ""

# -------------------------------------------------------
# Step 1: Run release gate unit tests
# -------------------------------------------------------

echo "--- Running release gate tests ---"
cd "$PROJECT_DIR"

if ! uv run pytest tests/test_release_checks.py -v --tb=short; then
    echo ""
    echo "FAIL: Release gate tests failed."
    exit 1
fi

echo ""
echo "[OK] All release gate tests passed."
echo ""

# -------------------------------------------------------
# Step 2: Run release checks against eval report
# -------------------------------------------------------

if [ -n "$REPORT_JSON" ]; then
    if [ ! -f "$REPORT_JSON" ]; then
        echo "ERROR: Report file not found: $REPORT_JSON"
        exit 2
    fi

    echo "--- Running release checks against $REPORT_JSON ---"

    GATE_OUTPUT="/mnt/artifacts-datai/reports/project_slam_gs3lam/gate_summary.json"
    mkdir -p "$(dirname "$GATE_OUTPUT")"

    if ! uv run python -m anima_slam_gs3lam.release_checks \
        "$REPORT_JSON" \
        --output "$GATE_OUTPUT" \
        --strict; then
        echo ""
        echo "FAIL: Release gates did not pass."
        echo "See: $GATE_OUTPUT"
        exit 1
    fi

    echo ""
    echo "[OK] All release gates passed."
    echo "Gate summary: $GATE_OUTPUT"
else
    echo "[SKIP] No report JSON provided — skipping gate evaluation."
    echo "  To run full validation:"
    echo "    bash scripts/release_validate.sh /path/to/eval_report.json"
fi

echo ""
echo "=============================================="
echo "  Release validation complete."
echo "=============================================="
exit 0
