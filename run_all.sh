#!/usr/bin/env bash
# Runs every task in order. Results -> $OUT, logs -> logs/.
# Smoke test first:   EXTRA="--quick" OUT=results_quick ./run_all.sh
# Full run:           ./run_all.sh
# Each task resumes: finished runs are skipped if you restart after a crash.
set -euo pipefail
cd "$(dirname "$0")"
OUT=${OUT:-results}
DATA=${DATA:-data}
EXTRA=${EXTRA:-}
mkdir -p logs "$OUT"
COMMON="--out $OUT --data $DATA $EXTRA"

run () {  # run <name> <script> [args...]
  local name=$1; shift
  echo "=== $(date '+%F %T')  START $name ==="
  python -u "$@" $COMMON 2>&1 | tee "logs/${name}.log"
  echo "=== $(date '+%F %T')  END   $name ==="
}

nvidia-smi || echo "WARNING: no GPU visible"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

run task1  task1_backprop.py
run task2  task2_optimizers.py
run task3a task3a_regularization.py
run task3b task3b_residual.py
# Task 4 is optional. Remove --evaluate-test while you are still tuning;
# the test set is then evaluated exactly once at the end.
run task4  task4_best_cnn.py --evaluate-test

python build_report.py --out "$OUT"
echo "All done. Open report.ipynb, write the interpretations, export to PDF."
