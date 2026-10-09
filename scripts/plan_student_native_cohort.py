#!/usr/bin/env python3
"""Select reset-feasible scenes before any native-expert outcome is observed."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collect_autopicker_dataset import episode_config
from treesim.fixed_base_picker import plan_fixed_base_stance
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--start',type=int,default=8100000);p.add_argument('--count',type=int,default=30);a=p.parse_args()
assert not a.output.exists()
rows=[];accepted=[]
for seed in range(a.start,a.start+500):
 plan=plan_fixed_base_stance(episode_config(seed));rows.append(dict(seed=seed,feasible=plan.feasible,report=plan.report))
 if plan.feasible:accepted.append(seed)
 print(seed,plan.feasible,len(accepted),flush=True)
 if len(accepted)==a.count:break
assert len(accepted)==a.count
a.output.write_text(json.dumps(dict(seeds=accepted,selection='first reset-feasible seeds under unchanged existing planner; no expert rollouts inspected',attempts=rows),indent=2))
