#!/usr/bin/env bash
# Run the automated test suite (unit + integration + end-to-end).
#   ./run_tests.sh          full suite, verbose
#   ./run_tests.sh --fast   skip tests marked "slow" (ablation, elbow, live server)
# Extra arguments are passed straight to pytest, e.g. ./run_tests.sh -k tfidf
set -uo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

ARGS=()
for a in "$@"; do
  if [ "$a" = "--fast" ]; then ARGS+=(-m "not slow"); else ARGS+=("$a"); fi
done

if ! "$PY" -c "import pytest" 2>/dev/null; then
  echo "pytest missing - installing from requirements-dev.txt"
  "$PY" -m pip install -q -r requirements-dev.txt || exit 1
fi

echo "== $($PY --version) | streamlit $($PY -c 'import streamlit; print(streamlit.__version__)') =="
START=$(date +%s)
"$PY" -m pytest -v -p no:cacheprovider ${ARGS[@]+"${ARGS[@]}"}
STATUS=$?
echo
echo "== test suite finished in $(( $(date +%s) - START ))s with exit code $STATUS ($([ $STATUS -eq 0 ] && echo PASS || echo FAIL)) =="
exit $STATUS
