"""Independent artifact checks for completed rounds (no simulator execution)."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
# This script lives under ROOT/log/arm_retime_rounds.
sys.path.insert(0,str(ROOT/'scripts'))
from run_arm_retime_rounds import OUT,hashes,prepare
from run_arm_retime_pilot import PROFILES,write
protocol=prepare();expected={r['seed'] for r in protocol['smoke']}
summary_checks=[];command_checks=[]
for path in sorted(OUT.glob('round[123]_summary.json'))+[OUT/'round0_legacy_summary.json']:
 d=json.loads(path.read_text());assert set(r['seed'] for r in d['runs'])==expected
 assert all(r['valid'] for r in d['pair_identity'])
 for kind in ['normal','stress','all']:
  for tag in ['legacy','candidate']:
   c=d['cohorts'][kind][tag];assert c['success_count']<=c['N']
 summary_checks.append(dict(path=str(path),seed_count=8,all_pairs_same_setup=True))
for path in sorted(OUT.glob('round*/runs/*/*/run.json')):
 r=json.loads(path.read_text());p=PROFILES[r['profile']];trace=r['command_trace']
 assert not r.get('error') and r['trajectory'] is not None
 q=np.array([x['q_cmd'] for x in trace]);g=np.array([x['q_goal'] for x in trace]);qd=np.array([x['qd_cmd'] for x in trace]);ts=np.array([x['sim_time'] for x in trace])
 max_error=0.;active_max_speed=0.;active_internal_acc=0.;landing_exceptions=0;unexplained=0
 for i in range(1,len(trace)):
  if trace[i-1]['phase'] in ['DONE','FAILED']:continue
  active=p.active_phases is None or trace[i]['phase'] in p.active_phases
  if p.mode in ['legacy','velocity'] or not active:
   rate=.045 if p.mode=='legacy' or not active else p.vmax_rad_s/60
   error=float(np.abs(q[i]-(q[i-1]+np.clip(g[i]-q[i-1],-rate,rate))).max());max_error=max(max_error,error)
   assert error<1e-12,(path,i,error)
  if active:
   active_max_speed=max(active_max_speed,float(np.abs((q[i]-q[i-1])/(ts[i]-ts[i-1])).max()))
   if p.amax_rad_s2:
    acc=np.abs((qd[i]-qd[i-1])/(ts[i]-ts[i-1]));active_internal_acc=max(active_internal_acc,float(acc.max()))
    exceed=acc>p.amax_rad_s2+1e-7;landing=np.array(trace[i]['landing'])
    landing_exceptions+=int(np.sum(exceed&landing));unexplained+=int(np.sum(exceed&~landing))
 assert unexplained==0,(path,unexplained)
 command_checks.append(dict(path=str(path),max_legacy_or_velocity_command_arithmetic_error=max_error,
    active_actual_command_velocity_max=active_max_speed,active_internal_acceleration_max=active_internal_acc,
    acceleration_landing_exceptions=landing_exceptions,unexplained_acceleration_exceptions=unexplained))
initial=json.loads((OUT/'protocol.json').read_text())['source_hashes'];current=hashes()
changed=[k for k in initial if initial[k]!=current[k]]
assert set(changed)=={'treesim/picker.py','treesim/arm_motion.py'}
write(OUT/'source_integrity.json',dict(initial=initial,final=current,changed_underlying_sources=changed,
    allowed_scope_fix_only=True,new_script_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'scripts/run_arm_retime_rounds.py',ROOT/'scripts/analyze_arm_retime_rounds.py',ROOT/'scripts/report_arm_retime_rounds.py',ROOT/'scripts/test_arm_motion.py']}))
write(OUT/'validation.json',dict(status='PASS',summary_checks=summary_checks,command_checks=command_checks,total_runs=len(command_checks)))
print('PASS',len(summary_checks),'round summaries;',len(command_checks),'run traces; permitted scope sources only')
