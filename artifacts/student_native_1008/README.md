# Student-native expert round — 2026-10-08/09

This directory versions compact experiment evidence. The running collection and
full simulation artifacts stay local; organizing these files does not restart
workers or alter the frozen expert.

| Tracked file | Purpose |
|---|---|
| [final_report.md](final_report.md) | Design, actual gate outcomes, limitations and collection commands |
| [gate_status.json](gate_status.json) | Gate verdict and frozen expert digest required by the launcher |
| [gate_c_seeds.json](gate_c_seeds.json) | The same 30 scenes used in both fixed-cohort acceptance runs |
| [gate_results.csv](gate_results.csv) | Per-scene phase times, outcome, controller diagnostics and data checks |
| [source_manifest.json](source_manifest.json) | Base repository versions, source entry points and expert configuration |
| [worker_scaling.md](worker_scaling.md), [worker_scaling.json](worker_scaling.json) | Measured 3/12/24-worker throughput and selected concurrency |

Code is organized under `treesim/` (expert and V2 action/loader contract) and
`scripts/` (collection, scheduling, validation, monitoring and reports). The
contract is documented in [docs/student_native_contract.md](../../docs/student_native_contract.md).

The collector imports the existing strict observer from the sibling XR-0
checkout at `../dualsys/Xiaomi-Robotics-0/xr0/test_grasp_1004/evaluation/strict_metrics.py`.
The existing environment and fixed-base planner are required. Use the project's
Pixi Python environment; the current launcher targets the configured local
workspace and requires tmux and ffmpeg.

From the repository root:

```bash
# CPU contract tests (does not launch a simulator)
.pixi/envs/default/bin/python scripts/test_orchard_command.py

# Reproduce the frozen-cohort evaluation in a NEW output directory
.pixi/envs/default/bin/python scripts/run_student_native_batch.py \
  --cohort artifacts/student_native_1008/gate_c_seeds.json \
  --output artifacts/student_native_1008/recheck --workers 3

# Start or resume the formal collection with the existing gate guard
bash scripts/start_student_native_collection.sh

# Current progress; this is authoritative over historical report snapshots
cat data/orchard_requested_v2_2000/progress.json
```

The launcher reserves capacity for 24 workers and initially dispatches at most
12 concurrent episodes, selected by the throughput measurements. Reports and
PIDs describe their observation time; use `progress.json` and `manager.pid` for
current state. Raw `gate_c_cohort.json` includes approximately 3.8 MB of planner
candidate diagnostics; the tracked seed-only file is sufficient to reproduce
the scenario list without copying that diagnostic dump.

Ignored local material includes `dev_*`, `h1_*`, `gate_c_v*`, per-step JSONL,
RGB videos, telemetry samples, `source_snapshot/`, intermediate expert copies,
and the entire `data/orchard_requested_v2_2000/` collection. Do not use `git add -f`
on those paths. Post-collection statistics are generated separately with
`scripts/summarize_student_native.py`; this commit does not claim the batch is
complete or fully revalidated after completion.

CSV phase boundaries are real control-step indices; divide by 30 for simulation
seconds. Success-chain fields are populated only for strict successes; an empty
field in a failed episode does not imply that no partial grasp/detach occurred.
Full partial-event histories remain in the ignored per-episode result files.
