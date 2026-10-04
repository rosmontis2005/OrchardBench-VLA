#!/usr/bin/env python3
"""Central, resumable scheduling and merge for the frozen 2650-expert cohort."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import collect_autopicker_dataset as collector

WORKER = ROOT / 'scripts/v1_dataset_worker.py'
TARGET = 2650


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    collector.write_json(path, data)


def append(path, data):
    with path.open('a') as f:
        f.write(json.dumps(data, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def manifest(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    collector.manifest_write(path, data)


def initialize(out, workers):
    if (out / 'protocol.json').exists():
        from v1_dataset_worker import check_sources
        check_sources(out)
        return
    scan = read(out / 'seed_scan.json')
    gate = read(out / 'pilot_raw/pilot_gate.json')
    assert gate['status'] == 'PASS' and gate['accepted'] == 3
    baseline = read(out / 'baseline_verification.json')
    config = dict(gate['config'])
    config.update(seed_start=scan['formal_seed_start'], record_rgb=True,
                  requested_accepted=TARGET, pilot_in_formal_cohort=False,
                  split_rule='every10th accepted val, centrally assigned in seed order')
    write(out / 'collection_config.json', config)
    replay_config = dict(mode='reach-conditioned', env_config=baseline['env_config'],
        maximum_dwell=30, position_tolerance_m=.01, rotation_tolerance_rad=.08,
        max_control_steps=900, max_episode_seconds=30, control_hz=30,
        advancement='reach OR maximum-dwell fallback; advance exactly one target',
        terminal_rule='strict evaluator success OR budget OR target sequence exhausted',
        additional_terminal_hold=False, passed_waypoint=False, lookahead=False,
        interpolation=False, recovery=False, replanning=False,
        success_source='OrchardVLAEnv.step info.success (current strict bucket evaluator)')
    write(out / 'replay_config.json', replay_config)
    sources = [ROOT / 'scripts/collect_autopicker_dataset.py', WORKER,
               out / 'sources/frozen_gate1_oracle.py', *sorted((ROOT / 'treesim').glob('*.py'))]
    protocol = dict(name='OrchardBench V1 2650 expert collection and replay filter',
        created_unix=time.time(), git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        baseline=baseline, source_hashes={str(p): sha(p) for p in sources},
        collector_source_sha256=sha(sources[0]), replay_source_sha256=sha(sources[2]),
        collection_config_sha256=sha(out / 'collection_config.json'),
        replay_config_sha256=sha(out / 'replay_config.json'),
        formal_seed_range=scan['formal_seed_range'], target_expert_accepted=TARGET,
        cohort_rule='sort all completed accepted attempts by seed, first 2650; preserve overflow',
        stopping_rule='stop new launches as soon as 2650 completed accepted; drain in-flight jobs',
        split_rule='every 10th raw accepted is val; filtering preserves split and episode ID',
        filter_rule='expert accepted AND strict replay success AND basic integrity valid',
        retry_rule='infrastructure only; never retry semantic task rejection or strict FAIL',
        requested_workers=workers, pilot_accepted=3, pilot_in_cohort=False,
        raw_root=str(ROOT / 'data/orchard_v1_2650/raw'),
        filtered_root=str(ROOT / 'data/orchard_v1_2650/filtered'),
        no_training=True)
    assert not Path(protocol['raw_root']).exists()
    assert not Path(protocol['filtered_root']).exists()
    write(out / 'protocol.json', protocol)
    write(out / 'worker_limit.json', {'collection': workers, 'replay': workers})


def resource_sample(out, stage, active):
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True).strip()
    mem = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
           if line.startswith(('MemAvailable:', 'MemTotal:'))}
    append(out / 'resources.jsonl', dict(time=time.time(), stage=stage, active=active, gpu=gpu, **mem))


def schedule(out, stage, workers):
    protocol = read(out / 'protocol.json')
    base = out / ('shards' if stage == 'collection' else 'replay_jobs')
    base.mkdir(exist_ok=True)
    completed = {}
    if stage == 'collection':
        for path in sorted(base.glob('seed_*/try_*/result.json')):
            r = read(path)
            assert r['seed'] not in completed, 'More than one semantic result for a seed'
            completed[r['seed']] = r
        first_seed = protocol['formal_seed_range'][0]
        launched = {int(p.name.removeprefix('seed_')) for p in base.glob('seed_*')}
        queue = sorted(launched - set(completed))
        next_seed = max([first_seed - 1, *launched]) + 1
        def accepted():
            return sum(r['accepted'] for r in completed.values())
    else:
        assert read(out / 'raw_cohort_frozen.json')['accepted'] == TARGET
        cohort = rows(out / 'raw_2650_manifest.jsonl')
        by_seed = {r['seed']: r for r in cohort}
        for p in base.glob('seed_*/result.json'):
            r = read(p)
            assert r['seed'] in by_seed and r['source_sha256'] == by_seed[r['seed']]['annotation_sha256']
            completed[r['seed']] = r
        queue = [r['seed'] for r in cohort if r['seed'] not in completed]
    active = {}
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               CUDA_VISIBLE_DEVICES='0')
    logpath = out / 'launch_commands.jsonl'
    retries = out / 'infrastructure_retries.jsonl'
    started = time.time()
    max_active = 0
    last_report = last_resources = 0.
    # A single small startup wave, then use the whole machine. Not a benchmark.
    startup = len(completed) == 0
    startup_launched = 0
    failure = None
    try:
        while True:
            for seed, job in list(active.items()):
                process, log, dest, attempt = job
                rc = process.poll()
                if rc is None:
                    continue
                log.close()
                del active[seed]
                result_path = dest / 'result.json' if stage == 'collection' else base / f'seed_{seed}/result.json'
                if result_path.exists():
                    r = read(result_path)
                    assert r['seed'] == seed and r['protocol_sha256'] == sha(out / 'protocol.json')
                    completed[seed] = r
                else:
                    output = (dest / 'stdout.log').read_text(errors='replace')
                    outcome = dest / 'expert_outcome.json'
                    if stage == 'collection' and outcome.exists() and not read(outcome)['accepted']:
                        r = read(outcome)
                        r.update(status='complete', protocol_sha256=sha(out / 'protocol.json'),
                                 recovered_committed_task_rejection=True)
                        write(result_path, r)
                        completed[seed] = r
                        append(retries, dict(stage=stage, seed=seed, kind='commit_recovery', returncode=rc,
                                             semantic_retry=False, previous_attempt=attempt))
                        continue
                    infrastructure = rc in (-9, -11, -6) or any(x in output.lower() for x in
                        ('out of memory', 'cuda_error_out_of_memory', 'no space left', 'input/output error',
                         'resource temporarily unavailable', 'too many open files', 'broken pipe'))
                    if not infrastructure or attempt >= 3:
                        raise RuntimeError(f'{stage} seed {seed} incomplete rc={rc}; inspect {dest / "stdout.log"}')
                    append(retries, dict(stage=stage, seed=seed, kind='process_or_filesystem_failure',
                        returncode=rc, previous_attempt=attempt, semantic_retry=False,
                        reason=output[-3000:], time=time.time()))
                    queue.insert(0, seed)
                    if 'out of memory' in output.lower():
                        limits = read(out / 'worker_limit.json')
                        limits[stage] = max(4, min(limits[stage] - 2, len(active)))
                        write(out / 'worker_limit.json', limits)
            if startup and startup_launched == 4 and not active:
                startup = False
                print(f'{stage}: startup wave complete; ramping to {workers} workers', flush=True)
            limit = 4 if startup else read(out / 'worker_limit.json')[stage]
            while len(active) < limit:
                if startup and startup_launched >= 4:
                    break
                if stage == 'collection' and accepted() >= TARGET and not queue:
                    break
                if queue:
                    seed = queue.pop(0)
                elif stage == 'collection':
                    seed = next_seed
                    next_seed += 1
                    assert seed <= protocol['formal_seed_range'][1]
                else:
                    break
                jobroot = base / f'seed_{seed}'
                jobroot.mkdir(exist_ok=True)
                previous = sorted(jobroot.glob('try_*'))
                attempt = max([0, *[int(p.name.removeprefix('try_')) for p in previous]]) + 1
                dest = jobroot / f'try_{attempt:02d}'
                dest.mkdir()
                cmd = [sys.executable, str(WORKER), 'collect' if stage == 'collection' else 'replay',
                       '--run', str(out)]
                if stage == 'collection':
                    cmd += ['--seed', str(seed), '--output', str(dest)]
                    launched.add(seed)
                else:
                    cmd += ['--trajectory', str(Path(protocol['raw_root']) / by_seed[seed]['annotation']),
                            '--output', str(jobroot / 'result.json')]
                log = (dest / 'stdout.log').open('w')
                p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                active[seed] = (p, log, dest, attempt)
                append(logpath, dict(stage=stage, seed=seed, attempt=attempt, pid=p.pid,
                    command=cmd, shell_command=shlex.join(cmd), time=time.time(),
                    environment={k: env[k] for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                        'CUDA_VISIBLE_DEVICES', 'CUDA_MPS_PIPE_DIRECTORY', 'CUDA_MPS_LOG_DIRECTORY') if k in env}))
                startup_launched += 1
                max_active = max(max_active, len(active))
            now = time.time()
            if now - last_report >= 15 or not active:
                status = dict(stage=stage, completed=len(completed), active=len(active),
                              worker_limit=limit, elapsed_s=now - started)
                if stage == 'collection':
                    status.update(accepted=accepted(), rejected=len(completed) - accepted(), next_seed=next_seed)
                    manifest(out / 'all_attempts.jsonl', sorted(completed.values(), key=lambda r: r['seed']))
                else:
                    status.update(strict_pass=sum(r['success'] for r in completed.values()))
                    manifest(out / 'replay_results.jsonl', sorted(completed.values(), key=lambda r: r['seed']))
                write(out / 'progress.json', status)
                print(json.dumps(status), flush=True)
                last_report = now
            if now - last_resources >= 30:
                resource_sample(out, stage, len(active))
                last_resources = now
            if not active and not queue and (stage != 'collection' or accepted() >= TARGET):
                break
            time.sleep(1)
    except BaseException as exc:
        failure = str(exc)
        raise
    finally:
        # Let already-launched semantic attempts finish even if the parent needs repair.
        for p, log, _, _ in active.values():
            p.wait()
            log.close()
        write(out / f'{stage}_workers.json', dict(requested_workers=workers,
            startup_workers=4, maximum_active_workers=max_active,
            final_worker_limit=read(out / 'worker_limit.json')[stage],
            independent_process_per_job=True, device_mapping={'all': 'CUDA_VISIBLE_DEVICES=0'},
            cuda_mps_pipe_directory=env.get('CUDA_MPS_PIPE_DIRECTORY'),
            elapsed_s=time.time() - started, failure=failure,
            scheduler_source_sha256=sha(__file__), worker_source_sha256=sha(WORKER)))
    return sorted(completed.values(), key=lambda r: r['seed'])


def merge(out):
    protocol = read(out / 'protocol.json')
    raw = Path(protocol['raw_root'])
    attempts = rows(out / 'all_attempts.jsonl')
    assert len({r['seed'] for r in attempts}) == len(attempts)
    accepted = [r for r in attempts if r['accepted']]
    assert len(accepted) >= TARGET
    selected, overflow = accepted[:TARGET], accepted[TARGET:]
    manifest(out / 'overflow_manifest.jsonl', overflow)
    manifest(out / 'expert_rejected_attempts.jsonl', [r for r in attempts if not r['accepted']])
    for sub in ('json/train', 'json/val', 'videos'):
        (raw / sub).mkdir(parents=True, exist_ok=True)
    final = []
    for index, attempt in enumerate(selected, 1):
        source = Path(attempt['trajectory'])
        assert sha(source) == attempt['trajectory_sha256']
        t = read(source)
        assert t['seed'] == attempt['seed']
        eid = f'episode_{index:06d}'
        split = 'val' if index % 10 == 0 else 'train'
        t.update(episode_id=eid, split=split)
        for view, entries in t['observations'].items():
            src = Path(entries[0]['path'])
            assert sha(src) == attempt['video_sha256'][view]
            dest = raw / 'videos' / f'{eid}_{view}.mp4'
            if not dest.exists():
                os.link(src, dest)
            assert os.path.samefile(src, dest)
            entries[0]['path'] = str(dest)
        relative = f'json/{split}/{eid}.json'
        dest = raw / relative
        collector.validate_arrays(t)
        if dest.exists():
            assert read(dest) == t
        else:
            write(dest, t)
        final.append(dict(attempt, episode_id=eid, split=split, annotation=relative,
                          dataset_root=str(raw), annotation_sha256=sha(dest),
                          integrity_valid=True, cohort='raw_2650'))
    manifest(raw / 'manifest.jsonl', final)
    manifest(out / 'raw_2650_manifest.jsonl', final)
    write(raw / 'collection_config.json', read(out / 'collection_config.json'))
    # Every video was fully decoded in its worker; merge verifies exact byte identity.
    assert len(list((raw / 'json').glob('*/*.json'))) == TARGET
    assert len(list((raw / 'videos').glob('*.mp4'))) == 2 * TARGET
    assert Counter(r['split'] for r in final) == {'train': 2385, 'val': 265}
    write(out / 'raw_cohort_frozen.json', dict(accepted=TARGET, splits={'train':2385, 'val':265},
        manifest_sha256=sha(out / 'raw_2650_manifest.jsonl'),
        raw_manifest_sha256=sha(raw / 'manifest.jsonl'), accepted_overflow=len(overflow),
        total_attempts=len(attempts), rejection_reasons=dict(Counter(r['reject_reason'] for r in attempts if not r['accepted'])),
        seed_min=attempts[0]['seed'], seed_max=attempts[-1]['seed'],
        cohort_seed_max=selected[-1]['seed'], integrity_invalid=0,
        all_videos_fully_decoded=True, centrally_verified_video_hashes=True,
        frozen_before_any_replay=True, timestamp=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('collect', 'replay'))
    parser.add_argument('--run', type=Path, default=ROOT / 'log/v1_2650_collection_replay_filter')
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    out = args.run.resolve()
    with (out / '.pipeline.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        initialize(out, args.workers)
        if args.stage == 'collect':
            if not (out / 'raw_cohort_frozen.json').exists():
                schedule(out, 'collection', args.workers)
                merge(out)
        else:
            schedule(out, 'replay', args.workers)


if __name__ == '__main__':
    main()
