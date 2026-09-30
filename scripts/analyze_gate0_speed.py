"""Small acceptance report adapter around the existing retime analysis."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from run_arm_retime_pilot import ROOT,write,PROFILES,ArmMotionProfile
from analyze_arm_retime_rounds import load,per_run,scoped_diagnostics,cohort
from analyze_arm_retime_pilot import aggregate,transition_aggregate,csvout
OUT=ROOT/'log/gate0_speed_30seed'

def vla_success(run):
    samples=[c for c in run['command_trace'] if 'vla_success_equivalent' in c and c['frame']%2==0]
    return any(c['vla_success_equivalent'] for c in samples) if samples else None

def analyze(directory,profile):
    runs=[]
    for p in sorted((directory/'runs'/profile).glob('*/run.json')):
        r=load(p);cfg=r['profile_config'].copy()
        if cfg.get('active_phases'):cfg['active_phases']=tuple(cfg['active_phases'])
        PROFILES[profile]=ArmMotionProfile(**cfg);runs.append(r)
    if not runs:return None
    selection=json.loads((OUT/'pilot_seed_selection.json').read_text())
    smoke={r['seed'] for r in selection['smoke']};stress={1000074,1000268}
    rows=[];env={}
    for r in runs:
        x=per_run(r);x.update(accepted=r['result']['accepted'],case_type='stress' if r['seed'] in stress else 'normal' if r['seed'] in smoke else 'unclassified',
          designation_source='existing smoke classification' if r['seed'] in smoke else 'expanded selection has no normal/stress designation; retained unclassified',
          vla_success_equivalent=vla_success(r),
          incidental_detach_count=r['result'].get('incidental_detach_count'),run_path=r['path'])
        rows.append(x)
    for ph in ['ALL','REACH','GRASP','PULL','TRANSPORT','DROP']:
        vals=[]
        for r in runs:
            if r['motion'] is None:continue
            p=r['trajectory']['proprios'];rot=np.array(p['ee_rotm']).reshape(-1,3,3)
            dp=np.diff(p['ee_pos'],axis=0);dr=Rotation.from_matrix(rot[1:]@rot[:-1].transpose(0,2,1)).as_euler('xyz')
            a=np.any(abs(dp)>.02+1e-9,axis=1);b=np.any(abs(dr)>.05+1e-9,axis=1)
            mask=np.ones(len(a),bool) if ph=='ALL' else r['motion']['labels']==ph
            vals.extend(np.stack([a,b,a|b],axis=1)[mask].tolist())
        env[ph]=dict(intervals=len(vals),**dict(zip(['translation','rotation','either'],np.mean(vals,axis=0).tolist() if vals else [None]*3)))
    counts={}
    for kind in ['normal','stress','unclassified','all']:
        rr=[r for r,x in zip(runs,rows) if kind=='all' or x['case_type']==kind]
        counts[kind]=cohort(rr)
        observed=[vla_success(r) for r in rr if vla_success(r) is not None]
        counts[kind]['vla_success_equivalent']=sum(observed) if observed else None
        counts[kind]['vla_success_equivalent_observed_N']=len(observed)
    doc=dict(profile=profile,profile_config=runs[0]['profile_config'],cohorts=counts,runs=rows,
             phase_metrics=aggregate(runs),student_envelope_metrics=env,transition_metrics=transition_aggregate(runs),
             diagnostics=scoped_diagnostics(runs))
    write(OUT/(profile+'_summary.json'),doc);csvout(OUT/(profile+'_seeds.csv'),rows)
    print(json.dumps(dict(profile=profile,counts=counts,envelope=env),indent=2))
    return doc
if __name__=='__main__':analyze(Path(sys.argv[1]),sys.argv[2])
