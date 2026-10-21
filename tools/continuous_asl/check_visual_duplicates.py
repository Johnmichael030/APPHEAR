"""Decoded-pixel duplicates and conservative review candidates, no sample filtering."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from prepare_features import checked_run
p=argparse.ArgumentParser();p.add_argument('--run-root',required=True,type=Path);args=p.parse_args();run=checked_run(args.run_root)
rows=json.loads((run/'data/processed/full/manifest.json').read_text());records=[];groups=defaultdict(list)
cv2.setNumThreads(2)
for row in rows:
    cap=cv2.VideoCapture(str(run/'data/raw/ncslgr/videos'/f"{row['sample_id']}.mov"));frames=[];h=hashlib.sha256()
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            h.update(np.asarray(frame.shape,np.int32).tobytes());h.update(frame.tobytes())
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);small=cv2.resize(gray,(9,8),interpolation=cv2.INTER_AREA)
            frames.append(np.packbits(small[:,1:]>small[:,:-1]).tobytes())
    finally:cap.release()
    if not frames:raise RuntimeError('NO_DECODED_FRAMES')
    sig=[frames[int(round((len(frames)-1)*q))].hex() for q in (0,.25,.5,.75,1)]
    rec={'sample_id':row['sample_id'],'split':row['split'],'decoded_sha256':h.hexdigest(),'frames':len(frames),'dhash5':sig}
    records.append(rec);groups[h.hexdigest()].append(rec)
exact=[g for g in groups.values() if len(g)>1];candidates=[]
for i,a in enumerate(records):
    for b in records[i+1:]:
        ratio=min(a['frames'],b['frames'])/max(a['frames'],b['frames'])
        distance=[(int(x,16)^int(y,16)).bit_count() for x,y in zip(a['dhash5'],b['dhash5'])]
        if ratio>=.95 and max(distance)<=2:
            candidates.append({'a':a['sample_id'],'b':b['sample_id'],'cross_split':a['split']!=b['split'],'hamming5':distance})
report={'decoded_pixel_exact_duplicates':exact,'near_duplicate_review_candidates':candidates,'clips_checked':len(records),
    'method':'all-frame BGR SHA256; five chronological 64-bit dHashes, max Hamming<=2, frame-count ratio>=0.95',
    'heuristic_warning':'dHash candidates require human review; absence does not prove no near-duplicates or signer leakage',
    'filtering_performed':False,'records':records}
with (run/'logs/visual_duplicates.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps({k:report[k] for k in ('clips_checked','decoded_pixel_exact_duplicates','near_duplicate_review_candidates')}))
