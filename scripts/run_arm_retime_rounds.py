#!/usr/bin/env python3
"""Fresh paired retiming rounds; reuses pilot workers without canonical gating.

A round must be explicitly selected with an evidence-based reason. No sweep and
no automatic expansion. Each candidate is preceded by a new legacy subprocess.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import run_arm_retime_pilot as pilot

ROOT = pilot.ROOT
OUT = ROOT / 'log/arm_retime_rounds'
# Scope-only repair keeps the previous velocity/acceleration values.
for name in ['v2.0', 'v1.5', 'va1.5_a20']:
    cfg = asdict(pilot.PROFILES[name]); cfg['active_phases'] = ('TRANSPORT',)
    pilot.PROFILES['transport_'+name] = pilot.ArmMotionProfile(**cfg)

SOURCES = ['treesim/arm_motion.py', 'treesim/picker.py', 'treesim/fixed_base_picker.py',
           'treesim/vla_env.py', 'treesim/orchard_action.py', 'treesim/xr0_adapter.py',
           'scripts/collect_autopicker_dataset.py', 'scripts/run_arm_retime_pilot.py',
           'scripts/analyze_arm_retime_pilot.py']


def hashes():
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}


def prepare():
    path = OUT/'protocol.json'
    if path.exists():
        protocol = json.loads(path.read_text())
        revision = OUT/'source_revision.json'
        expected = json.loads(revision.read_text())['source_hashes'] if revision.exists() else protocol['source_hashes']
        assert expected == hashes(), 'Underlying experiment sources changed without documented revision'
        overrides = OUT/'case_overrides.json'
        if overrides.exists():
            updates=json.loads(overrides.read_text())
            for row in protocol['smoke']:
                if str(row['seed']) in updates:
                    row['initial_case_type']=row['case_type']
                    row['case_type']=updates[str(row['seed'])]['case_type']
                    row['case_reason']=updates[str(row['seed'])]['reason']
        return protocol
    old = json.loads((pilot.OUT/'pilot_seed_selection.json').read_text())
    seeds = [dict(r, case_type='stress' if r['seed'] == 1000074 else 'normal',
                  case_reason='Historical contact/IK trajectory sensitivity and extreme transport speed' if r['seed'] == 1000074
                  else 'Canonical accepted smoke anchor; provisional normal, checked against fresh legacy') for r in old['smoke']]
    protocol = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        smoke=seeds, source_hashes=hashes(), historical_selection=str(pilot.OUT/'pilot_seed_selection.json'),
        profiles={k: asdict(v) for k,v in pilot.PROFILES.items()},
        pairing='Round 0 fresh legacy; each later round runs a new legacy immediately before its same-seed candidate in separate subprocesses.',
        historical_trajectory_gate=False, simulation_hz=60, substeps=3, record_hz=30,
        detach_force_scale=1.5, record_rgb=False, max_candidate_rounds=3,
        stress_policy='Retain all seeds; no numerical trajectory or all-seeds-success veto. Class changes require recorded baseline/geometry evidence; candidate failure alone does not reclassify normal cases.',
        metric_policy='Reuse pilot raw measured 30Hz finite differences and SO(3); interval-start phase labels. Command traces at 60Hz after update. Report pooled frames plus paired seed ratios and cohort counts.')
    pilot.write(path, protocol)
    return protocol


def worker(round_index, profile, seed):
    pilot.OUT = OUT/f'round{round_index}'
    return pilot.one(profile, seed)


def run(round_index, profile, reason, selected_seeds=None, label=None):
    protocol = prepare()
    start_hashes = hashes()
    assert 0 <= round_index <= 3
    assert (round_index == 0 and profile == 'legacy') or (round_index > 0 and profile != 'legacy')
    directory = OUT/(label or f'round{round_index}')
    # Main rounds are immutable/resumable. Targeted repeat uses a separate label.
    spec = dict(round=round_index, profile=profile, reason=reason, seeds=selected_seeds or [r['seed'] for r in protocol['smoke']])
    if (directory/'spec.json').exists():
        assert json.loads((directory/'spec.json').read_text()) == spec
    else:
        assert reason
        pilot.write(directory/'spec.json', spec)
    snapshot=directory/'source_snapshot.json'
    if snapshot.exists():
        assert json.loads(snapshot.read_text()) == start_hashes, 'Cannot resume a round under different sources'
    else:
        pilot.write(snapshot,start_hashes)
    pilot.write(directory/'pilot_seed_selection.json', dict(all_candidates=protocol['smoke']))
    for seed in spec['seeds']:
        for name in (['legacy'] if round_index == 0 else ['legacy', profile]):
            dest = directory/'runs'/name/str(seed)
            if (dest/'run.json').exists():
                prior=json.loads((dest/'run.json').read_text())
                assert not prior.get('error'), f'Previous worker error: {dest}'
                continue
            dest.mkdir(parents=True, exist_ok=True)
            with (dest/'stdout.log').open('w') as log:
                proc=subprocess.run([sys.executable, __file__, 'worker', '--round',str(round_index),
                    '--profile', name, '--seed',str(seed), '--label', directory.name], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            assert proc.returncode == 0 and (dest/'run.json').exists(), f'Worker infrastructure failure: {dest}/stdout.log'
            payload=json.loads((dest/'run.json').read_text()); r=payload['result']
            print(f"{directory.name} {name} seed={seed} success={r['accepted']} placed={r.get('placed')} reason={r.get('reject_reason')} sim={r['sim_duration_s']:.3f}s wall={payload['wall_time_s']:.1f}s",flush=True)
    assert start_hashes == hashes()
    from analyze_arm_retime_rounds import summarize
    summary=summarize(directory)
    print(json.dumps(summary['cohorts'],indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['run','worker','prepare'])
    ap.add_argument('--round',type=int,default=0);ap.add_argument('--profile',choices=pilot.PROFILES,default='legacy')
    ap.add_argument('--reason');ap.add_argument('--seed',type=int);ap.add_argument('--seeds',type=int,nargs='+');ap.add_argument('--label')
    a=ap.parse_args()
    if a.stage=='worker':
        pilot.OUT=OUT/(a.label or f'round{a.round}')
        raise SystemExit(pilot.one(a.profile,a.seed))
    if a.stage=='prepare':prepare();return
    run(a.round,a.profile,a.reason,a.seeds,a.label)

if __name__=='__main__':main()
