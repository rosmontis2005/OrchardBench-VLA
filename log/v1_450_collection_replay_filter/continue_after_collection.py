import json,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]);out=root/'log/v1_450_collection_replay_filter'
while not (out/'raw_cohort_frozen.json').exists():
    marker=out/'collection_workers.json'
    if marker.exists() and json.loads(marker.read_text()).get('failure'):
        raise RuntimeError('Collection needs infrastructure repair; replay not launched')
    time.sleep(5)
for cmd,name in [([sys.executable,str(root/'scripts/run_v1_dataset_pipeline.py'),'replay','--workers','24'],'replay_console.log'),([sys.executable,str(root/'scripts/finalize_v1_dataset.py')],'finalize_console.log')]:
    # The scheduler exits immediately after writing the freeze marker; allow its lock to close.
    time.sleep(2)
    with (out/name).open('a') as log:
        subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
