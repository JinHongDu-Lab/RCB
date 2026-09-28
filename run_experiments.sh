#!/usr/bin/env bash
#
# One entry point for all four studies.
#
#   ./run_experiments.sh                  every study, production settings
#   ./run_experiments.sh --smoke          every study, reduced; proves the chain runs
#   ./run_experiments.sh simulation       one study (simulation|evaluation|lalonde|replogle)
#   ./run_experiments.sh --notebooks      execute all four notebooks in place
#   ./run_experiments.sh --jobs 8         cap parallelism
#   ./run_experiments.sh --tests          run the test suite
#
# Every stage is timed and the totals are printed at the end.  That is not a
# nicety: the runtime table in README.md goes stale the moment anything
# changes, and a script that reports its own timings is what keeps the
# documentation honest.  Paste the summary into the commit message.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if [[ ! -d rcb || ! -d experiments ]]; then
    echo "run_experiments.sh must live in code/ beside rcb/ and experiments/" >&2
    exit 2
fi

PYTHON="${PYTHON:-python}"
SMOKE=0
JOBS=""
NOTEBOOKS=0
TESTS=0
STUDIES=()

while (($#)); do
    case "$1" in
        --smoke)     SMOKE=1 ;;
        --notebooks) NOTEBOOKS=1 ;;
        --tests)     TESTS=1 ;;
        --jobs)      JOBS="${2:?--jobs needs a count}"; shift ;;
        --jobs=*)    JOBS="${1#*=}" ;;
        -h|--help)   sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        simulation|evaluation|lalonde|replogle) STUDIES+=("$1") ;;
        *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
    esac
    shift
done

# No study named and no other action requested means "run everything".
if ((${#STUDIES[@]} == 0)) && ((NOTEBOOKS == 0)) && ((TESTS == 0)); then
    STUDIES=(simulation evaluation lalonde replogle)
fi

export CAUSAL_EXTRAPOLATION_SMOKE=$SMOKE
[[ -n "$JOBS" ]] && export CAUSAL_EXTRAPOLATION_JOBS="$JOBS"

# A smoke run uses eight replications where production uses five hundred.
# Left to write into results/ it would overwrite the committed numbers with
# numbers that mean nothing, and the files would not say which they were.  So
# smoke output goes to a scratch tree unless the caller has already chosen one.
if ((SMOKE)) && [[ -z "${RCB_OUTPUT_ROOT:-}" ]]; then
    RCB_OUTPUT_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/rcb-smoke-XXXXXX")"
    export RCB_OUTPUT_ROOT
    echo "smoke output -> $RCB_OUTPUT_ROOT (results/ untouched)"
fi

# Section 5 is sequential by design; its RNG is consumed in one stream and
# parallelizing it would change every number it reports.  Left unthreaded so
# BLAS does not oversubscribe the cores the other two studies are using.
if [[ -n "$JOBS" ]]; then
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
fi

SMOKE_FLAG=()
((SMOKE)) && SMOKE_FLAG=(--smoke)

TIMINGS=()
FAILED=0

run_stage() {
    local label="$1"; shift
    printf '\n\033[1m=== %s ===\033[0m\n' "$label"
    local started
    started=$(date +%s)
    if "$@"; then
        local elapsed=$(( $(date +%s) - started ))
        TIMINGS+=("$(printf '%-28s %5dm %02ds  ok' "$label" $((elapsed / 60)) $((elapsed % 60)))")
    else
        local status=$? elapsed=$(( $(date +%s) - started ))
        TIMINGS+=("$(printf '%-28s %5dm %02ds  FAILED (exit %d)' "$label" $((elapsed / 60)) $((elapsed % 60)) "$status")")
        FAILED=1
    fi
}

TOTAL_STARTED=$(date +%s)

for study in ${STUDIES[@]+"${STUDIES[@]}"}; do
    run_stage "$study" "$PYTHON" -m "experiments.${study}.run" \
        ${SMOKE_FLAG[@]+"${SMOKE_FLAG[@]}"}
done

if ((NOTEBOOKS)); then
    if ((SMOKE)); then
        # Several figure cells iterate the production dimension and eta grid
        # (imported as ``CFG = DEFAULT_CONFIG``) to decide which columns to
        # look for, regardless of what actually produced the CSV they read.
        # Smoke output uses a reduced grid, so those cells look up a (p, eta)
        # cell that does not exist and crash on an empty array.  That is a
        # property of the notebooks, not of this script, and the fix belongs
        # there; until then, --smoke --notebooks is refused rather than left
        # to fail with a stack trace that does not explain itself.
        echo
        echo "skipping --notebooks under --smoke: several figure cells key" \
             "off the production"
        echo "dimension/eta grid regardless of what generated results/, and" \
             "will raise on a"
        echo "(p, eta) cell smoke output does not have.  Run notebooks" \
             "against production output."
    else
        for notebook in 1_simulation 2_evaluation 3_lalonde 4_replogle; do
            run_stage "notebook ${notebook}" \
                "$PYTHON" -m jupyter nbconvert --to notebook --execute --inplace \
                "${notebook}.ipynb"
        done
    fi
fi

((TESTS)) && run_stage "tests" "$PYTHON" -m pytest tests -q

TOTAL=$(( $(date +%s) - TOTAL_STARTED ))

printf '\n\033[1m=== timings ===\033[0m\n'
printf '%s\n' ${TIMINGS[@]+"${TIMINGS[@]}"}
printf '%-28s %5dm %02ds\n' "TOTAL" $((TOTAL / 60)) $((TOTAL % 60))
((SMOKE)) && printf '\n(--smoke: every stage ran at reduced counts.  This proves the\n chain executes and nothing whatever about the numbers.)\n'

exit $FAILED
