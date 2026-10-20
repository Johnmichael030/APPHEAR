"""Additional post-selection final-epoch TRAIN inference; no refit or DEV score."""
import json,time
import numpy as np
import torch
from core import RUN,read,write,sha
from ctc_experiment import CausalCTC,evaluate,seeds

def main():
 if (RUN/'last_checkpoint_train_diagnostic.json').exists():print('FINAL_TRAIN_DIAGNOSTIC_ALREADY_COMPLETE');return
 seeds(20261020);base=RUN/'models/reference48';pointer=read(base/'checkpoints/latest.json');path=base/pointer['path']
 if sha(path)!=pointer['sha256']:raise ValueError('LATEST_CHECKPOINT_HASH')
 freeze=read(RUN/'model_selection_freeze.json')
 if sha(RUN/'data/processed/train_dev/manifest.json')!=freeze['manifest_sha256']:raise ValueError('FROZEN_MANIFEST_HASH')
 state=torch.load(path,map_location='cpu',weights_only=False)
 model=CausalCTC(len(state['vocabulary']),48,state['model']['mean'].numpy(),state['model']['std'].numpy());model.load_state_dict(state['model'])
 rows=[r for r in read(RUN/'data/processed/train_dev/manifest.json') if r['split']=='train'];data={}
 for row in rows:
  with np.load(row['feature_path'],allow_pickle=False) as f:data[row['sample_id']]=f['X'].copy()
 start=time.perf_counter();result=evaluate(model,rows,data,state['vocabulary'],'cpu')
 write(RUN/'last_checkpoint_train_diagnostic.json',{'scope':'Additional post-selection final-epoch TRAIN inference only; no candidate reselection, no DEV or TEST scoring.',
       'checkpoint_path':str(path),'checkpoint_sha256':pointer['sha256'],'epoch':pointer['epoch'],'seconds':time.perf_counter()-start,
       'selection_freeze_sha256':sha(RUN/'model_selection_freeze.json'),'script_sha256':sha(__file__),'result':result})
 print(json.dumps({'last_epoch':pointer['epoch'],'full_train_wer':result['gloss_wer'],'loss':result['loss'],'training_performed':False}),flush=True)

if __name__=='__main__':main()
