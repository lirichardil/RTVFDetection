#!/usr/bin/env bash
# =============================================================================
# run_tests.sh — Activate the rtvf-detection conda environment and run pytest
# =============================================================================
# Usage:
#   chmod +x run_tests.sh
#   ./run_tests.sh [pytest-options]
#
# Examples:
#   ./run_tests.sh                        # run all Phase 1 tests
#   ./run_tests.sh -v                     # verbose output
#   ./run_tests.sh -k test_tensorflow_gpu # run a single test by name
# =============================================================================

# -e  exit on error in our own commands
# -o pipefail  catch errors in pipelines
# NOTE: -u (unbound variable as error) is intentionally omitted.
#       conda activation scripts (e.g. libblas_mkl_activate.sh) reference
#       variables such as $MKL_INTERFACE_LAYER without defaults; -u would
#       abort the script when those scripts are sourced.
set -eo pipefail

ENV_NAME="rtvf-detection"
TEST_PATH="tests/test_phase1_environment.py"

# ---------------------------------------------------------------------------
# Resolve project root (directory containing this script)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  RTVF Detection — Phase 1 Environment Tests"
echo "  Project root : $SCRIPT_DIR"
echo "  Conda env    : $ENV_NAME"
echo "============================================================"

# ---------------------------------------------------------------------------
# Activate conda
# ---------------------------------------------------------------------------
# Support both conda installed system-wide and via miniforge/mambaforge
CONDA_BASE=""
for candidate in \
    "$HOME/miniforge3" \
    "$HOME/mambaforge" \
    "$HOME/miniconda3" \
    "$HOME/anaconda3" \
    "/opt/conda" \
    "/usr/local/anaconda3"; do
    if [ -f "$candidate/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="$candidate"
        break
    fi
done

if [ -z "$CONDA_BASE" ]; then
    echo "[ERROR] Could not locate a conda installation."
    echo "        Please activate '$ENV_NAME' manually and re-run:"
    echo "          conda activate $ENV_NAME && pytest $TEST_PATH -v"
    exit 1
fi

# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo ""
echo "  Python   : $(python --version)"
echo "  Conda env: $CONDA_DEFAULT_ENV"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Expose conda env CUDA libraries to TensorFlow's dynamic loader
# ---------------------------------------------------------------------------
# TF 2.10 opens libcudart.so.11.0, libcublas.so.11, libcudnn.so.8, etc. via
# dlopen() at runtime.  When CUDA is installed through conda (cudatoolkit +
# cudnn packages) the .so files live in $CONDA_PREFIX/lib but are NOT on
# LD_LIBRARY_PATH by default, so TF fails to find them even though the libs
# are physically present.  Prepending $CONDA_PREFIX/lib fixes this.
if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    echo "  LD_LIBRARY_PATH set to include: ${CONDA_PREFIX}/lib"
fi

# ---------------------------------------------------------------------------
# Run pytest
# Extra flags:
#   -v          verbose per-test output
#   -s          show print() output (for PASS/FAIL inline messages)
#   --tb=short  concise tracebacks on failure
# ---------------------------------------------------------------------------
pytest "$TEST_PATH" \
    -v \
    -s \
    --tb=short \
    "$@"          # forward any extra args the user supplies

EXIT_CODE=$?

echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  [PASS]  All Phase 1 tests passed."
else
    echo "  [FAIL]  One or more Phase 1 tests failed (exit code $EXIT_CODE)."
    echo "          Review the output above, fix the issues, and re-run."
fi
echo "============================================================"

exit $EXIT_CODE
