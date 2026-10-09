#!/usr/bin/env python3
"""Scene-paired statistics. All attempts and common-valid subsets stay separate."""
import csv
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/initialization_prior_1009'
MODES=['C0','Cz','Cxyz0','Cxyz'];INITS=['G0','G-','G+']


def load_rows():
    rows=[]
    for name in ['round0_reference_initial_audit','round0_reference_diagnostic','round0_reference','round0_smoke','round1','round2','round3']:
        for f in sorted((OUT/name).glob('*/result.json')):
            row=json.loads(f.read_text());row['round']=name;rows.append(row)
    return rows


def wilson(k,n):
    if not n:return [None,None]
    z=1.959963984540054;p=k/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,mid-half),min(1.,mid+half)]


def paired(values):
    values=np.asarray(values,float);n=len(values)
    if not n:return dict(n=0)
    rng=np.random.default_rng(1009)
    samples=values[rng.integers(0,n,(20000,n))].mean(axis=1)
    positive=int((values>0).sum());negative=int((values<0).sum());discordant=positive+negative
    # Exact paired binary test supplements percentile bootstrap for sparse pairs.
    exact=(min(1.,2*sum(math.comb(discordant,k) for k in range(min(positive,negative)+1))/2**discordant)
           if discordant else 1.) if np.isin(values,[-1.,0.,1.]).all() else None
    return dict(n=n,exact_mcnemar_p=exact,difference=float(values.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),
                positive=int((values>0).sum()),negative=int((values<0).sum()),ties=int((values==0).sum()))


def summarize(rows):
    if not rows:return {}
    seeds=sorted({r['scene_seed'] for r in rows});lookup={(r['scene_seed'],r['controller_mode'],r['initialization_mode']):r for r in rows}
    assert len(lookup)==len(rows), 'Duplicate scene/mode/initialization rows'
    groups={}
    for mode in MODES:
        for g in INITS:
            rr=[r for r in rows if r['controller_mode']==mode and r['initialization_mode']==g]
            if not rr:continue
            k=sum(r['strict_success_any_fruit'] for r in rr);n=len(rr)
            groups[mode+'/'+g]=dict(n=n,success=k,success_rate=k/n,ci95=wilson(k,n),
                planned_success=sum(r['strict_success_planned_fruit'] for r in rr),
                invalid=sum(not r['initialization_valid'] for r in rr),
                funnel={key:sum(bool(r[key]) for r in rr) for key in ['held15','detach','release','valid_release','stable_bucket']},
                chain_funnel=dict(grasp_phase=sum(any(e['phase']=='GRASP' for e in r['phases']) for r in rr),
                    held15=sum(any(v['held15_step'] is not None for v in r['strict']['fruit_chains'].values()) for r in rr),
                    detach=sum(any(v['held15_step'] is not None and v['first_detach_after_grasp_step'] is not None for v in r['strict']['fruit_chains'].values()) for r in rr),
                    valid_release=sum(any(e['valid_chain'] for e in r['strict']['release_events']) for r in rr),stable_bucket=k),
                release_diagnostics={key:dict(count=len(values),median=float(np.median(values)),p95=float(np.percentile(values,95)),maximum=max(values)) if values else None
                    for key in ['fruit_bucket_distance','fruit_speed'] if (values:=[e[key] for r in rr for e in r['release_diagnostics']]) is not None},
                steps_median=float(np.median([r['controller_steps'] for r in rr])),
                steps_mean=float(np.mean([r['controller_steps'] for r in rr])),
                success_steps_median=float(np.median([r['controller_steps'] for r in rr if r['strict_success_any_fruit']])) if k else None,
                ik_failure_rate=float(np.average([r['ik_failure_rate'] for r in rr],weights=[r['controller_steps'] for r in rr])),
                clipping_rate=float(np.average([r['clipping_rate'] for r in rr],weights=[r['controller_steps'] for r in rr])),
                tracking_position_p95_median=float(np.median([r['tracking_position_p95'] for r in rr])),
                failures=dict(Counter(r['termination_reason'] for r in rr if not r['strict_success_any_fruit'])))
    contrasts={}
    def success(s,m,g):return int(lookup[s,m,g]['strict_success_any_fruit'])
    for g in INITS:
        for high,low in [('Cz','C0'),('Cxyz0','Cz'),('Cxyz','Cxyz0'),('Cxyz','C0')]:
            both=[s for s in seeds if (s,high,g) in lookup and (s,low,g) in lookup]
            contrasts[f'{high}-{low}/{g}']=paired([success(s,high,g)-success(s,low,g) for s in both])
    for g in ['G-','G+']:
        for m in MODES:
            both=[s for s in seeds if (s,m,g) in lookup and (s,m,'G0') in lookup]
            contrasts[f'{m}/{g}-G0']=paired([success(s,m,g)-success(s,m,'G0') for s in both])
        both=[s for s in seeds if all((s,m,i) in lookup for m in ['C0','Cxyz'] for i in ['G0',g])]
        contrasts[f'information_gain/{g}-G0']=paired([(success(s,'Cxyz',g)-success(s,'C0',g))-(success(s,'Cxyz','G0')-success(s,'C0','G0')) for s in both])
    both=[s for s in seeds if all((s,m,g) in lookup for m in ['C0','Cxyz'] for g in INITS)]
    contrasts['Cxyz-C0/averaged_G_clustered']=paired([np.mean([success(s,'Cxyz',g)-success(s,'C0',g) for g in INITS]) for s in both])
    for key,value in contrasts.items():
        if key.startswith('information_gain/') or 'averaged_G' in key:value.pop('exact_mcnemar_p',None)
    return dict(base_scenes=len(seeds),rollouts=len(rows),groups=groups,paired_contrasts=contrasts)


def geometry(rows):
    unique={}
    for r in rows:
        key=(r['scene_seed'],r['initialization_mode'])
        if key in unique:
            np.testing.assert_allclose(r['actual_initial_target_base_xyz'],unique[key]['actual_initial_target_base_xyz'],atol=1e-6,rtol=0)
            np.testing.assert_allclose(r['base_pose'],unique[key]['base_pose'],atol=1e-6,rtol=0)
        unique[key]=r
    paired_checks=0
    for (seed,g),r in unique.items():
        if g=='G0' or (seed,'G0') not in unique:continue
        base=unique[seed,'G0'];dy=-.08 if g=='G-' else .08
        np.testing.assert_allclose(np.array(r['actual_initial_target_base_xyz'])-base['actual_initial_target_base_xyz'],[0.,-dy,0.],atol=1e-6,rtol=0)
        np.testing.assert_allclose(r['base_pose'][3:],base['base_pose'][3:],atol=1e-6,rtol=0)
        paired_checks+=1
    result={}
    for g in INITS:
        rr=[r for (seed,init),r in unique.items() if init==g]
        if not rr:continue
        xyz=np.array([r['actual_initial_target_base_xyz'] for r in rr])
        distance=np.array([r['target_tcp_distance'] for r in rr])
        result[g]=dict(n=len(rr),xyz_mean=xyz.mean(0).tolist(),xyz_std=xyz.std(0,ddof=1).tolist(),
            xyz_min=xyz.min(0).tolist(),xyz_max=xyz.max(0).tolist(),
            target_tcp_distance_mean=float(distance.mean()),target_tcp_distance_range=[float(distance.min()),float(distance.max())],
            invalid=sum(not r['initialization_valid'] for r in rr),
            endpoint_ik_reachable={phase:sum(r['initial_pose_ik'][phase]['reachable'] for r in rr) for phase in ['pregrasp','grasp']},
            home_twig_contact_scenes=sum(r['initial_geometry']['home_twig_contacts']>0 for r in rr),
            min_chassis_clearance=min(r['initial_geometry']['chassis_clearance'] for r in rr),
            min_home_arm_clearance=min(r['initial_geometry']['home_arm_clearance'] for r in rr))
    return dict(initializations=result,paired_geometry_checks=paired_checks)


def main():
    rows=load_rows();assert rows
    fields=['round','scene_seed','controller_mode','initialization_mode','actual_initial_target_base_xyz',
            'estimated_initial_target_base_xyz','base_pose','initialization_valid','held15','detach','release',
            'valid_release','stable_bucket','strict_success_any_fruit','strict_success_planned_fruit','first_grasp_step',
            'release_step','success_step','termination_reason','ik_failure_rate','clipping_rate','tracking_position_p95',
            'controller_steps','target_tcp_distance','initial_geometry','initial_pose_ik','phases','release_diagnostics',
            'planned_fruit_id','strict','branch_breaks','incidental_detached_ids','wall_seconds','planning_calls','live_position_updates','shadow_max_command_error']
    with (OUT/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");writer.writeheader()
        for r in rows:writer.writerow({k:json.dumps(r[k],ensure_ascii=False) if isinstance(r.get(k),(dict,list)) else r.get(k) for k in fields})
    stats={}
    for label,names in [('development',['round1','round2']),('formal',['round3'])]:
        rr=[r for r in rows if r['round'] in names]
        if not rr:continue
        valid={s for s in {r['scene_seed'] for r in rr} if all(r['initialization_valid'] for r in rr if r['scene_seed']==s)}
        stats[label]=dict(geometry=geometry(rr),all_attempts=summarize(rr),common_valid=summarize([r for r in rr if r['scene_seed'] in valid]),common_valid_seeds=sorted(valid))
    manifest={name:json.loads((OUT/(name+'_scenes.json')).read_text()) for name in ['calibration','round0','development','test'] if (OUT/(name+'_scenes.json')).exists()}
    (OUT/'scene_manifest.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'statistics.json').write_text(json.dumps(stats,indent=2))
    print(json.dumps({label:data['all_attempts']['groups'] for label,data in stats.items()},indent=2))
if __name__=='__main__':main()
