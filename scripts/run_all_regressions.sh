#!/usr/bin/env bash
#
# Unified regression suite — runs every eval/regression set in one command and
# writes all outputs under data/regression_reports/<timestamp>/.
#
# Suites:
#   1. Identifier probes            (12)  -> /ask, gold-citation
#   2. Tester regression / Raul     (30)  -> /ask, answered + norm-cite
#   3. Retrieval eval               (90)  -> in-process recall@k (hybrid/rerank)
#   4. eval_v2 citation+abstention  (100) -> /ask, gold-cited / abstained
#
# Suites 1/2/4 hit the live API (default prod :8088); suite 3 evaluates the
# retrieval pipeline in-process (the same shipped code). Run sequentially so the
# suites never contend for the single chat GPU.
#
# Usage: scripts/run_all_regressions.sh [API_BASE_URL]
#        scripts/run_all_regressions.sh http://127.0.0.1:8088
set -uo pipefail

API="${1:-http://127.0.0.1:8088}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Resolve the project venv. Prefer $ROOT/.venv; when run from a git worktree
# (which has no .venv of its own), fall back to the main checkout's .venv found
# via git-common-dir. Override with DOGV_PY=/path/to/python.
PY="${DOGV_PY:-$ROOT/.venv/bin/python}"
if [ ! -x "$PY" ]; then
    MAIN="$(dirname "$(git -C "$ROOT" rev-parse --git-common-dir 2>/dev/null || echo /nonexistent)")"
    if [ -x "$MAIN/.venv/bin/python" ]; then
        PY="$MAIN/.venv/bin/python"
    fi
fi
if [ ! -x "$PY" ]; then
    echo "ERROR: could not find the project venv python (set DOGV_PY=/path/to/python)" >&2
    exit 1
fi
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/data/regression_reports/$STAMP"
mkdir -p "$OUT"
cd "$ROOT"

echo "=== DOGV regression suite | api=$API | $(date) ==="
echo "reports -> $OUT"

echo; echo "### [1/4] Identifier probes (12)"
"$PY" scripts/oneoff/run_identifier_probes.py --api "$API" 2>&1 | tee "$OUT/1_identifier_probes.txt"

echo; echo "### [2/4] Tester regression / Raul (30)"
"$PY" scripts/run_tester_regression.py --api "$API" 2>&1 | tee "$OUT/2_tester_regression.txt"

echo; echo "### [3/4] Retrieval eval (90, recall@k, in-process)"
"$PY" scripts/run_eval.py --input data/eval_v2/retrieval_input.json \
    --output-dir "$OUT/retrieval" 2>&1 | tee "$OUT/3_retrieval_eval.txt"

echo; echo "### [4/4] eval_v2 citation + abstention (100)"
"$PY" scripts/eval_v2_citation_check.py --api "$API" 2>&1 | tee "$OUT/4_eval_v2_citation.txt"

echo; echo "=== done $(date) | reports in $OUT ==="
