# Worker scaling measurement

| Concurrent episodes | Control steps / wall second | Measurement |
|---:|---:|---|
| 3 | 30.37 | Whole initial session, including startup/drain |
| 12 | 34.71 | 165 s after startup |
| 24 | 31.32 | 301 s after startup |

Selected active episode limit: 12. Process-pool capacity: 24. The 24-worker trial was stable and produced accepted, validated episodes, but did not improve observed throughput. Existing episodes finish naturally before the limit is reached; no seed is discarded, no episode is interrupted, and no data-quality or expert parameter is changed.

These short intervals use different production scenes and are not matched-seed statistical estimates. The nominal simulation/control clock remains 60/30 Hz; these values measure aggregate offline collection throughput.

Resume uses the existing tmux launcher, now configured with --workers 24 --active-workers 12. Live limit is recorded in data/orchard_requested_v2_2000/worker_limit.json; progress.json reports the limit and the actual in_flight count separately. The manager PID is 33571.
