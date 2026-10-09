# MPS production restart — 2026-10-09

The previous manager received SIGTERM and stopped dispatching new seeds. All
12 in-flight episodes finished naturally: 1371 attempts, 1201 accepted, zero
in-flight at shutdown. No episode was interrupted or rerun for this transition.

The resumed tmux pipeline uses a private CUDA MPS service at
`/tmp/orchard_requested_v2_mps`, one CUDA device, and OMP/OpenBLAS/MKL thread
limits of 1. This follows the earlier 2650-batch setup without changing the GPU
compute mode. `get_client_list` confirmed 12, then 24, then 32 actual MPS clients.
Expert parameters, the controller, H5/30 Hz timing, seed order, success criteria
and requested-command labels remained unchanged.

| Active workers | Measured interval after startup (s) | Aggregate control steps / wall second |
|---:|---:|---:|
| 12 | 166.75 | 139.55 |
| 24 | 197.72 | 241.67 |
| 32 | 306.80 | 249.25 |

The earlier non-MPS 12-worker observation was 34.71 steps/s. MPS with 24 workers
was approximately 6.96 times faster in these windows. MPS with 32 workers added
only 3.13% over 24, with higher memory use, so the sustained choice is **24 active
episodes, 32 process-pool capacity**. Lowering the limit lets excess episodes
finish; idle pool processes can retain CUDA memory until the manager exits.

These are sequential production windows on different seeds, including normal
encoding/validation stalls. They establish an engineering choice, not a matched
experiment or a precise statistical optimum. New accepted episodes ran the same
full per-trajectory validator. No CUDA/worker infrastructure errors appeared in
the measured windows; reset-infeasible scene exceptions are retained separately
in the attempt results, including those from before MPS.

The first 100 accepted episodes after the switch had step-weighted IK failure
0.976% and clipping 0.802%; the 1201 preceding accepted episodes had 0.914% and
0.775%. These are different scene sets and are not a test of MPS determinism.

Raw samples, endpoints and resource readings remain local under
`data/orchard_requested_v2_2000/mps_{12,24,32}_samples.jsonl` and
`mps_scaling_results.json`. The service environment and logs are in
`mps_runtime.json` and `mps/`.

`scripts/start_student_native_collection.sh` resumes the same dataset through
`scripts/run_student_native_mps.py`. After 2000 accepted trajectories, the
pipeline performs full validation and computes new V2 statistics with 8 CPU
workers. Only then does it write `pipeline_complete.json` and stop its private
MPS service. See `final_report.md` for the actual completion status.

## Completed batch

The resumed MPS session completed the remaining **799 accepted trajectories**
from 909 attempts in 80.31 minutes, including ramp-up,
the three concurrency windows, and the final drain. During the sustained
24-active-worker observation before the final drain, aggregate throughput was
233.92 control steps/s over 59.61 minutes.

Final batch: **2280 attempts, 2000 accepted, 483284 full H5-anchored windows**.
All 2266 generated trajectories passed full revalidation; the other 14 attempts
had no feasible reset. All 20 independent step-log/real-video loader samples
passed. There were no infrastructure failures and no interrupted episode
directories. The strict event predicate passed on 2035 attempts; 35 of those
were excluded by the existing controller-quality filter. All 280 rejected
attempts remain recorded.

Full validation, sampling and V2 candidate statistics finished approximately
9.13 minutes after collection. The pipeline,
collection workers and private MPS service exited normally; MPS `quit` returned 0.
The final manifests and statistics remain under the ignored dataset directory.
