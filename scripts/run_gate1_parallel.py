#!/usr/bin/env python3
"""Scheduling only: resume frozen Gate 1 revalidation in independent seed jobs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import revalidate_gate1_transport as experiment


def run_episode(out, protocol_hash, e):
    eid = e['episode_id']
    dest = out / 'expert' / eid
    dest.mkdir(parents=True, exist_ok=True)
    worker = str(Path(experiment.__file__).resolve())
    if not (dest / 'collection.json').exists():
        print(f"COLLECT {eid} seed={e['seed']}", flush=True)
        with (dest / 'stdout.log').open('w') as log:
            subprocess.run([sys.executable, worker, '--output', str(out), '--collect', eid],
                           stdout=log, stderr=subprocess.STDOUT, cwd=experiment.ROOT, check=True)
    c = json.loads((dest / 'collection.json').read_text())
    print(f"EXPERT {eid} accepted={c['result']['accepted']} reason={c['result'].get('reject_reason')}", flush=True)
    if not c['trajectory_available']:
        return
    trajectory = dest / 'trajectory.json'
    assert experiment.gate1.sha(trajectory) == c['trajectory_sha256']
    if json.loads(trajectory.read_text())['num_frames'] < experiment.gate1.HORIZON:
        return
    for mode in experiment.MODES:
        target = out / 'episodes' / f'{eid}_{mode}.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            print(f'REPLAY {eid} {mode}', flush=True)
            with target.with_suffix('.log').open('w') as log:
                subprocess.run([sys.executable, worker, '--replay', str(trajectory), '--mode', mode,
                                '--output', str(target), '--protocol-hash', protocol_hash],
                               stdout=log, stderr=subprocess.STDOUT, cwd=experiment.ROOT, check=True)
        r = json.loads(target.read_text())
        assert r['protocol_sha256'] == protocol_hash and r['source_sha256'] == c['trajectory_sha256']
        print(f"DONE {eid} {mode} success={r['success']} phase={r['final_phase']} budget={r['timeout']} dwell={r['dwell_timeout_count']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--jobs', type=int, default=3)
    args = ap.parse_args()
    assert 1 <= args.jobs <= 3
    out = args.output.resolve()
    path = out / 'frozen_protocol.json'
    protocol = json.loads(path.read_text())
    for p, h in {**protocol['protected_hashes'], **protocol['source_hashes']}.items():
        assert experiment.gate1.sha(Path(p)) == h
    amendment = dict(reason='User requested concurrent independent environments to accelerate the same fixed experiment.',
                     concurrent_seed_jobs=args.jobs, no_worker_or_parameter_changes=True,
                     seed_order_within_job='collect once -> time-indexed -> reach-conditioned',
                     scheduling_source=str(Path(__file__).resolve()),
                     scheduling_source_sha256=experiment.gate1.sha(Path(__file__)),
                     preexisting_results=sorted(p.name for p in (out / 'episodes').glob('*.json')))
    amendment_path = out / 'parallel_scheduling.json'
    assert not amendment_path.exists(), 'Do not overwrite the scheduling record'
    experiment.gate1.write(amendment_path, amendment)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_episode, out, experiment.gate1.sha(path), e) for e in protocol['episodes']]
        for future in futures:
            future.result()
    for p, h in {**protocol['protected_hashes'], **protocol['source_hashes']}.items():
        assert experiment.gate1.sha(Path(p)) == h
    assert experiment.gate1.sha(Path(__file__)) == amendment['scheduling_source_sha256']
    experiment.summarize(out, protocol)


if __name__ == '__main__':
    main()
