#!/usr/bin/env python3
"""Resume the frozen native collection under private MPS, then validate it.

This only configures process scheduling. Expert, control and dataset semantics
remain those checked by the collection manager's gate.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--active-workers', type=int, default=24)
    args = parser.parse_args()
    assert 1 <= args.active_workers <= args.workers
    output = ROOT / 'data/orchard_requested_v2_2000'
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Do not start MPS while a previous collection session is still draining.
    with (output / 'manager.lock').open('a') as manager_lock:
        fcntl.flock(manager_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pipe = Path('/tmp/orchard_requested_v2_mps')
    pipe.mkdir(mode=0o700, exist_ok=True)
    logs = output / 'mps'
    logs.mkdir(exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='0',
               CUDA_MPS_PIPE_DIRECTORY=str(pipe), CUDA_MPS_LOG_DIRECTORY=str(logs))

    def control(command, check=True):
        result = subprocess.run(['nvidia-cuda-mps-control'], input=command+'\n',
                                text=True, capture_output=True, env=env,
                                timeout=15, check=check)
        with (logs / 'commands.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=time.time(), command=command,
                         stdout=result.stdout, stderr=result.stderr,
                         returncode=result.returncode))+'\n')
        return result.stdout.strip()

    if not (pipe / 'nvidia-cuda-mps-control.pid').exists():
        subprocess.run(['nvidia-cuda-mps-control', '-d'], env=env, check=True, timeout=15)
        control(f'start_server -uid {os.getuid()}')
    servers = control('get_server_list').split()
    assert servers, 'Private MPS server unavailable'
    (output / 'mps_runtime.json').write_text(json.dumps(dict(
        time=time.time(), pipeline_pid=os.getpid(), server_pids=servers,
        workers=args.workers, initial_active_workers=args.active_workers,
        environment={k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'CUDA_MPS_PIPE_DIRECTORY',
                     'CUDA_MPS_LOG_DIRECTORY', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
        gpu_compute_mode_changed=False), indent=2))
    child = None
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    (output / 'pipeline.pid').write_text(str(os.getpid()))
    command = [sys.executable, '-u', str(ROOT/'scripts/run_student_native_batch.py'),
               '--output', str(output), '--gate', str(ROOT/'artifacts/student_native_1008/gate_status.json'),
               '--workers', str(args.workers), '--active-workers', str(args.active_workers),
               '--target', '2000', '--start', '8200000']
    print(json.dumps(dict(event='mps_collection_start', command=command, servers=servers)), flush=True)
    try:
        with (output/'manager.log').open('a') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            rc = child.wait()
        if rc:
            raise RuntimeError(f'Collection manager exited {rc}')
        progress = json.loads((output/'progress.json').read_text())
        if stopping or progress['status'] != 'complete':
            print('Collection stopped; completed episodes preserved. Finalization deferred.', flush=True)
            return
        assert progress['accepted'] == progress['target'] == 2000
        print('Collection complete; revalidating all trajectories and computing V2 statistics.', flush=True)
        child = subprocess.Popen([sys.executable, '-u', str(ROOT/'scripts/summarize_student_native.py'),
                                  str(output), '--revalidate', '--stats', '--workers', '8'], cwd=ROOT, env=env)
        if child.wait():
            raise RuntimeError('Final validation/statistics failed; inspect pipeline.log')
        summary=json.loads((output/'summary.json').read_text())
        stats=json.loads((output/'action_stats_requested_v2.json').read_text())
        assert summary['final'] and summary['accepted']==2000 and summary['full_revalidation_requested']
        assert stats['complete_batch'] and stats['windows']==summary['usable_windows']
        (output/'pipeline_complete.json').write_text(json.dumps(dict(
            completed=time.time(), accepted=2000, full_revalidation=True,
            statistics='action_stats_requested_v2.json', source_split='unsplit_candidates'), indent=2))
        print('Collection and full validation complete.', flush=True)
    finally:
        # Only this dataset's private service; never stop another CUDA workload.
        if child is None or child.poll() is not None:
            control('quit', check=False)


if __name__ == '__main__':
    main()
