#!/usr/bin/env python3
"""Run the user-requested separate extension only after the first batch is healthy."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_v1_dataset_pipeline import ROOT, read, rows, sha, write, append


def main():
    first = ROOT / 'log/v1_450_collection_replay_filter'
    out = ROOT / 'log/v1_2650_collection_replay_filter'
    with (out / '.extension_queue.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(out / 'queue_status.json', dict(stage='waiting_for_450_completion', no_extra_attempts_started=True))
        while not ((first / 'report.md').exists() and (first / 'dataset_summary.json').exists()):
            for name in ('collection_workers.json', 'replay_workers.json'):
                p = first / name
                if p.exists() and read(p).get('failure'):
                    raise RuntimeError(f'450 batch requires repair: {p}')
            console = first / 'continuation.log'
            if console.exists() and 'Traceback (most recent call last)' in console.read_text():
                raise RuntimeError('450 continuation requires repair; extension not launched')
            time.sleep(5)
        summary = read(first / 'dataset_summary.json')
        protocol = read(first / 'protocol.json')
        checks = dict(
            status=summary['status'] == 'COMPLETE',
            raw_accepted=summary['collection']['expert_accepted'] == 450,
            replay_count=summary['oracle_replay']['total_replayed'] == 450,
            unique_replay_seeds=len({r['seed'] for r in rows(first / 'replay_results.jsonl')}) == 450,
            integrity=summary['dataset']['integrity_invalid_total'] == 0,
            loader=summary['dataset']['loader_verification']['status'] == 'PASS',
            train_only_stats=not summary['dataset']['stats']['validation_used'],
            no_training=summary['no_training_performed'],
            collection_completed=read(first / 'collection_workers.json')['failure'] is None,
            replay_completed=read(first / 'replay_workers.json')['failure'] is None,
            sources_unchanged=all(sha(p) == h for p,h in protocol['source_hashes'].items()))
        write(out / 'preceding_450_health_check.json', dict(checks=checks, passed=all(checks.values()),
              strict_FAIL_is_not_a_blocker=True, summary_sha256=sha(first / 'dataset_summary.json')))
        assert all(checks.values()), '450 batch not healthy; do not start extension'
        prior_attempts = rows(first / 'all_attempts.jsonl')
        maximum = max(r['seed'] for r in prior_attempts)
        start = (max(maximum, protocol['formal_seed_range'][1]) // 10000 + 1) * 10000
        if not (out / 'seed_scan.json').exists():
            write(out / 'seed_scan.json', dict(method='Reuse initial manifest scan plus completed 450 attempts; choose above previous reserved range',
                original_scan=str(first / 'seed_scan.json'), original_scan_sha256=sha(first / 'seed_scan.json'),
                preceding_attempts_sha256=sha(first / 'all_attempts.jsonl'), max_existing_seed=maximum,
                previous_reserved_seed_range=protocol['formal_seed_range'],
                formal_seed_start=start, formal_seed_range=[start, start + 999999],
                no_overlap=True, pilot_reused=True))
            for src, dest in [('baseline_verification.json', 'baseline_verification.json'),
                              ('pilot_raw/pilot_gate.json', 'pilot_raw/pilot_gate.json'),
                              ('sources/frozen_gate1_oracle.py', 'sources/frozen_gate1_oracle.py')]:
                path = out / dest
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(first / src, path)
            write(out / 'pilot_reuse.json', dict(source=str(first / 'pilot_raw/pilot_gate.json'),
                sha256=sha(first / 'pilot_raw/pilot_gate.json'), identical_scientific_config=True,
                new_pilot_attempts=0, reused_accepted=3, included_in_cohort=False))
            (out / 'git_state.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)
                + '\ncommit ' + subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True))
            (out / 'worktree_before.patch').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=ROOT))
            (out / 'infrastructure_retries.jsonl').touch()
        # Per-batch CUDA scheduling only. Each client still owns its Newton world,
        # trajectory transaction and replay result; no simulation/controller changes.
        pipe = Path('/tmp/orchard_v1_2650_mps')
        mps_logs = out / 'mps'
        pipe.mkdir(mode=0o700, exist_ok=True)
        mps_logs.mkdir(exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   CUDA_MPS_PIPE_DIRECTORY=str(pipe), CUDA_MPS_LOG_DIRECTORY=str(mps_logs),
                   CUDA_VISIBLE_DEVICES='0')
        def mps_control(command):
            p = subprocess.run(['nvidia-cuda-mps-control'], input=command + '\n',
                               text=True, capture_output=True, env=env, check=True, timeout=15)
            append(out / 'mps_commands.jsonl', dict(command=command, stdout=p.stdout,
                                                  stderr=p.stderr, time=time.time()))
            return p.stdout
        if not (pipe / 'nvidia-cuda-mps-control.pid').exists():
            subprocess.run(['nvidia-cuda-mps-control', '-d'], env=env, check=True, timeout=15)
            mps_control(f'start_server -uid {os.getuid()}')
            time.sleep(1)
        servers = mps_control('get_server_list').split()
        assert servers, 'Private MPS server unavailable; no extension attempts launched'
        write(out / 'runtime_environment.json', dict(
            CUDA_MPS_PIPE_DIRECTORY=str(pipe), CUDA_MPS_LOG_DIRECTORY=str(mps_logs),
            CUDA_VISIBLE_DEVICES='0', server_pids=servers, independent_environments=True,
            gpu_compute_mode_changed=False, active_thread_percentage=100,
            startup_check='first four actual collection/replay jobs, no repeated semantic attempts',
            documentation='https://docs.nvidia.com/deploy/mps/appendix-tools-and-interface-reference.html'))
        import run_v1_dataset_2650 as extension
        extension.initialize(out, 24)
        p = read(out / 'protocol.json')
        if 'runtime_environment' not in p:
            assert not (out / 'shards').exists(), 'Never amend a running scientific protocol'
            p['runtime_environment'] = read(out / 'runtime_environment.json')
            write(out / 'protocol.json', p)
        for source, digest in p['source_hashes'].items():
            source = Path(source)
            assert sha(source) == digest
            if source.parent == out / 'sources':
                continue
            dest = out / 'sources' / source.parent.name / source.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
        finalizer = ROOT / 'scripts/finalize_v1_dataset_2650.py'
        write(out / 'preparation_source.json', dict(path=str(finalizer), sha256=sha(finalizer)))
        jobs = [([sys.executable, str(ROOT / 'scripts/run_v1_dataset_2650.py'), 'collect', '--workers', '24'], 'collection'),
                ([sys.executable, str(ROOT / 'scripts/run_v1_dataset_2650.py'), 'replay', '--workers', '24'], 'replay'),
                ([sys.executable, str(finalizer)], 'finalize')]
        for cmd, stage in jobs:
            write(out / 'queue_status.json', dict(stage=stage, preceding_450_health_verified=True, time=time.time()))
            append(out / 'stage_commands.jsonl', dict(command=cmd, stage=stage, time=time.time()))
            with (out / f'{stage}_console.log').open('a') as log:
                subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        mps_control('quit')
        write(out / 'queue_status.json', dict(stage='complete', no_training=True, time=time.time()))


if __name__ == '__main__':
    main()
