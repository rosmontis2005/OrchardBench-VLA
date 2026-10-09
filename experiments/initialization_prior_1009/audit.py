"""Replay the original strict observer from saved boundaries; check H5 clock."""
import argparse
import gzip
import json
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[2]/'scripts')]
from run import strict_mod,OUT
import numpy as np


def check(folder):
    result=json.loads((folder/'result.json').read_text());strict=strict_mod.StrictPlacement()
    n=0
    with gzip.open(folder/'steps.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line);n+=1
            assert r['step']==n and abs(r['sim_time']-n/30)<1e-7
            assert r['command']['planning_step']==(n-1)//5*5
            assert np.isfinite(r['native']['action']).all()
            strict.update(n,r['held'],r['detached'],r['in_bucket'],bool(r['in_bucket']))
    assert n==result['controller_steps']
    if result['controller_mode']=='Cxyz':assert result['live_position_updates']==result['planning_calls']
    assert strict.summary()==result['strict']
    if result['controller_mode'] in ['C0','Cz','Cxyz0']:assert result['live_position_updates']==0
    return dict(folder=str(folder.relative_to(OUT)),steps=n,passed=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('batch');p.add_argument('--sample-per-cell',type=int,default=0);a=p.parse_args()
    files=sorted((OUT/a.batch).glob('*/result.json'))
    if a.sample_per_cell:
        cells={}
        for f in files:
            r=json.loads(f.read_text());cells.setdefault((r['controller_mode'],r['initialization_mode']),[]).append(f)
        files=[ff[int(i)] for ff in cells.values() for i in np.linspace(0,len(ff)-1,min(a.sample_per_cell,len(ff)),dtype=int)]
    rows=[check(f.parent) for f in files]
    assert rows
    (OUT/(a.batch+'_audit.json')).write_text(json.dumps(rows,indent=2))
    print('Audited',len(rows),'rollouts;',sum(r['steps'] for r in rows),'consecutive boundaries')
if __name__=='__main__':main()
