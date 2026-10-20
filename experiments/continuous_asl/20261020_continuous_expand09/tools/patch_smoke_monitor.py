"""Preserve failed pre-optimization smoke; fix its empty monitor intersection."""
import ast,json
from core import RUN,read,write,sha

def main():
 destination=RUN/'tools/train_pilot.v2.py'
 if destination.exists():return
 source_path=RUN/'tools/train_pilot.py';source=source_path.read_text(encoding='utf-8')
 replacements=[("out=RUN/'models'/name","out=RUN/'models'/('smoke_v2' if name=='smoke' else name)"),
   ("RUN/'models/smoke/training_status.json'","RUN/'models/smoke_v2/training_status.json'"),
   ("monitor=[r for r in fit if r['sample_id'] in config['train_monitor_ids']]","monitor=list(fit) if name=='smoke' else [r for r in fit if r['sample_id'] in config['train_monitor_ids']]\n    if not monitor:raise ValueError('EMPTY_TRAIN_MONITOR')"),
   ("status=RUN/'models'/args.name/'training_status.json'","status=RUN/'models'/('smoke_v2' if args.name=='smoke' else args.name)/'training_status.json'"),
   ("pointer=read(RUN/'models'/args.name/'checkpoints/best.json')","pointer=read(RUN/'models'/('smoke_v2' if args.name=='smoke' else args.name)/'checkpoints/best.json')"),
   ("sha(RUN/'models'/args.name/pointer['path'])","sha(RUN/'models'/('smoke_v2' if args.name=='smoke' else args.name)/pointer['path'])")]
 for before,after in replacements:
  if before not in source:raise ValueError('SMOKE_FIX_ANCHOR:'+before)
  source=source.replace(before,after)
 ast.parse(source)
 with destination.open('x',encoding='utf-8') as f:f.write(source)
 rows=read(RUN/'data/processed/train_dev/manifest.json');fit=[r for r in rows if r['split']=='train'];plan=read(RUN/'training_plan.v2.json')
 write(RUN/'pilot_patch.v2.json',{'reason':'The four-row smoke fit had no overlap with the global stratified64 monitor; evaluation on an empty list raised ZeroDivisionError before epoch1.',
       'failed_command':'Run-Pilot.ps1 → python -B tools/train_pilot.py --name smoke',
       'failed_output_preserved':'models/smoke and original logs; no checkpoint or epoch produced',
       'source_sha256':sha(source_path),'fixed_sha256':sha(destination),'manifest_sha256_unchanged':sha(RUN/'data/processed/train_dev/manifest.json'),
       'smoke_monitor_count':4,'pilot_monitor_count':sum(r['sample_id'] in plan['train_monitor_ids'] for r in fit),
       'changes':'Smoke monitor uses its actual four TRAIN rows; successful-smoke output gets new directory smoke_v2. Full pilot monitor/data/model/optimizer unchanged.'})
 print('PASS_SMOKE_MONITOR4_MAIN_MONITOR64_CODE_PARSED',flush=True)

if __name__=='__main__':main()
