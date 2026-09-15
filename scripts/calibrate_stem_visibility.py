#!/usr/bin/env python
"""Paired strength sweep or reset-only visibility audit; no training/data collection.

Outputs immutable per-attempt JSONL and raw stdout. All planned seeds are kept,
including planner/GRASP failures; no success-conditioned resampling.
"""
import argparse
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
import gc
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collect_autopicker_dataset import collect_episode, write_json


def aggregate(rows):
    output = []
    for scale in sorted(set(r['detach_force_multiplier'] for r in rows)):
        rr = [r for r in rows if r['detach_force_multiplier'] == scale]
        dd = [r['detach_diagnostics'] for r in rr if r.get('detach_diagnostics', {}).get('detach_frame') is not None]
        distances = [d['retract_distance_at_detach_m'] for d in dd]
        forces = [r['max_pull_N'] for r in rr if r.get('grasped')]
        output.append(dict(multiplier=scale, attempts=len(rr),
            planner_feasible=sum(r['reject_reason'] != 'no_feasible_fixed_base_setup' for r in rr),
            grasped=sum(r['grasped'] for r in rr), detached=len(dd), placed=sum(r['placed'] for r in rr),
            accepted=sum(r['accepted'] for r in rr),
            premature_detach=sum(d['premature_detach'] for d in dd),
            detached_before_pull_entry=sum(d['pull_enter_frame'] is None for d in dd),
            pull_settle_detach=sum(d['premature_detach'] and d['pull_enter_frame'] is not None for d in dd),
            accepted_premature_detach=sum(r.get('detach_diagnostics', {}).get('premature_detach') is True
                for r in rr if r['accepted']),
            detach_after_active_pull=sum(not d['premature_detach'] for d in dd),
            detach_after_3cm=sum(d['meaningful_pull'] for d in dd),
            detach_after_5cm=sum(d['strong_meaningful_pull'] for d in dd),
            grasp_failure=sum(not r['grasped'] and any(t['state']=='GRASP' for t in r.get('state_trace', [])) for r in rr),
            branch_break=sum(r['branch_break_count'] > 0 for r in rr),
            mean_retract_at_detach=float(np.mean(distances)) if distances else None,
            median_retract_at_detach=float(np.median(distances)) if distances else None,
            mean_max_pull=float(np.mean(forces)) if forces else None,
            max_pull=max(forces, default=None),
            reject_reasons=dict(Counter(r['reject_reason'] for r in rr if not r['accepted']))))
    return output


def visibility_summary(rows):
    feasible = [r for r in rows if r.get('target_visibility')]
    views = [r['target_visibility']['initial'] for r in feasible]
    counts = Counter(('both' if v['static_visible'] and v['wrist_visible'] else
        'static_only' if v['static_visible'] else 'wrist_only' if v['wrist_visible'] else 'neither') for v in views)
    def dist(name):
        a = [v[name+'_visible_pixels'] for v in views]
        return dict(values=a, min=min(a), median=float(np.median(a)), max=max(a),
                    percentiles=np.percentile(a,[0,10,25,50,75,90,100]).tolist()) if a else None
    return dict(attempts=len(rows), planner_feasible=len(feasible), target_visible=sum(v['any_policy_view_visible'] for v in views),
        fully_invisible=counts['neither'], view_counts=dict(counts), static=dist('static'), wrist=dist('wrist'),
        rgb_auxiliary_output_sanity='PASS' if all(v.get('static_rgb_unchanged_with_shape_output') and
           v.get('wrist_rgb_unchanged_with_shape_output') for v in views) and views else 'NOT RUN')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['stem','visibility'])
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--seed-start', type=int, default=920000)
    ap.add_argument('--seeds', type=int, default=12, help='stem: fixed attempts per multiplier')
    ap.add_argument('--multipliers', type=float, nargs='+', default=[1.,1.25,1.5,2.])
    ap.add_argument('--feasible-target', type=int, default=40)
    ap.add_argument('--max-attempts', type=int, default=150)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'results.jsonl'
    if path.exists(): raise FileExistsError(path)
    write_json(args.output/'command.json', dict(argv=sys.argv, seeds=list(range(args.seed_start,args.seed_start+args.seeds)),
        pairing='same fixed seeds for every multiplier', thresholds_cm=[3,5]))
    rows = []
    scales = args.multipliers if args.mode == 'stem' else [1.]
    with path.open('x') as f:
        for scale in scales:
            count = args.seeds if args.mode == 'stem' else args.max_attempts
            for seed in range(args.seed_start,args.seed_start+count):
                gallery = args.output/'gallery'/str(seed) if args.mode=='visibility' else None
                with (args.output/f'{scale:g}_{seed}.txt').open('w') as capture:
                    with redirect_stdout(capture), redirect_stderr(capture):
                        row, traj, images = collect_episode(seed,args.output,scale,record_rgb=False,
                            initial_only=args.mode=='visibility',gallery_dir=gallery)
                rows.append(row); f.write(json.dumps(row,allow_nan=False)+'\n'); f.flush()
                print(json.dumps(dict(seed=seed,multiplier=scale,accepted=row['accepted'],reason=row['reject_reason'],
                    detach=row.get('detach_diagnostics'),visibility=row.get('target_visibility'))),flush=True)
                del traj,images; gc.collect()
                summary = aggregate(rows) if args.mode=='stem' else visibility_summary(rows)
                write_json(args.output/'summary.json',summary)
                if args.mode=='visibility' and summary['planner_feasible']>=args.feasible_target: break
    print(json.dumps(summary,indent=2))

if __name__ == '__main__': main()
