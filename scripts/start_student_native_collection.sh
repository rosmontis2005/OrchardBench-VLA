#!/usr/bin/env bash
set -euo pipefail
cd /home/rosmontis/Projects/orchardbench
.pixi/envs/default/bin/python - <<'PY'
import json,hashlib
from pathlib import Path
p=Path('artifacts/student_native_1008/gate_status.json');g=json.loads(p.read_text())
assert g['gate_a_pass'] and g['gate_b_pass'] and g['gate_c_total']==30 and g['gate_c_success']>=24, 'Gates have not passed; collection will not start'
assert g['expert_sha256']==hashlib.sha256(Path('treesim/student_native_expert.py').read_bytes()).hexdigest(), 'Expert differs from frozen gate version'
PY
if tmux has-session -t orchard-native-v2 2>/dev/null; then
    echo 'tmux session orchard-native-v2 already exists'
    exit 0
fi
mkdir -p data/orchard_requested_v2_2000
tmux new-session -d -s orchard-native-v2 'cd /home/rosmontis/Projects/orchardbench && exec env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .pixi/envs/default/bin/python -u scripts/run_student_native_batch.py --output data/orchard_requested_v2_2000 --gate artifacts/student_native_1008/gate_status.json --workers 24 --active-workers 12 --target 2000 --start 8200000 >> data/orchard_requested_v2_2000/manager.log 2>&1'
echo 'Started tmux session orchard-native-v2; inspect actual progress and episode artifacts before declaring it healthy.'
