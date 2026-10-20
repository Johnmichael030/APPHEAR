"""One-variable TRAIN-only learnability test: unused output classes, not labels."""
from collections import defaultdict
import hashlib,json,sys,time
import numpy as np
import torch
from core import RUN,ROOT,read,write,sha,SCHEMA
sys.path.insert(0,str(ROOT/'tools/continuous_asl'))
from ctc_experiment import CausalCTC,batch,evaluate,seeds

def main():
 out=RUN/'vocab_cardinality_probe'
 if (out/'results.json').exists():print('PROBE_COMPLETE_NO_REPEAT');return
 if out.exists():raise ValueError('INCOMPLETE_PROBE_RETAINED')
 manifest=[r for r in read(RUN/'data/processed/train_dev/manifest.json') if r['split']=='train'];groups=defaultdict(list)
 for row in manifest:groups[row['signer_id']].append(row)
 for rows in groups.values():rows.sort(key=lambda r:hashlib.sha256(r['sample_id'].encode()).hexdigest())
 selected=[];used=set()
 while len(selected)<8:
  for signer in sorted(groups):
   if len(selected)==8:break
   remaining=[r for r in groups[signer] if r['source_group'] not in used]
   row=remaining[0] if remaining else groups[signer][0];groups[signer].remove(row);selected.append(row);used.add(row['source_group'])
 data={}
 for row in selected:
  if sha(row['feature_path'])!=row['feature_sha256']:raise ValueError('FEATURE_HASH')
  with np.load(row['feature_path'],allow_pickle=False) as f:
   if str(f['schema_version'].item())!=SCHEMA:raise ValueError('SCHEMA')
   data[row['sample_id']]=f['X'].copy()
 full_vocab=read(RUN/'pilot_fresh48_v2/vocabulary.json')['tokens'];small_vocab=['<BLANK>','<UNK>']+sorted({t for r in selected for t in r['glosses']})
 # Only TRAIN-fitted normalization buffers are read, never trained recognition weights.
 pointer=read(RUN/'pilot_fresh48_v2/checkpoints/best.json');state=torch.load(RUN/'pilot_fresh48_v2'/pointer['path'],map_location='cpu',weights_only=False)
 mean=state['model']['mean'].numpy();std=state['model']['std'].numpy();del state
 seeds(20261020);full=CausalCTC(len(full_vocab),48,mean,std).to('cuda:0');small=CausalCTC(len(small_vocab),48,mean,std).to('cuda:0')
 full_state=full.state_dict();small.load_state_dict({k:v for k,v in full_state.items() if not k.startswith('output.')},strict=False)
 ids=[full_vocab.index(t) for t in small_vocab]
 with torch.no_grad():small.output.weight.copy_(full.output.weight[ids]);small.output.bias.copy_(full.output.bias[ids])
 xb,_,_,_=batch(selected,data,full_vocab,'cuda:0')
 with torch.inference_mode():common_error=float((full(xb)[:,:,ids]-small(xb)).abs().max())
 if common_error>1e-5:raise ValueError('COMMON_INITIAL_LOGITS_DIFFER')
 out.mkdir();write(out/'manifest.json',selected);write(out/'protocol.json',{'scope':'8eligible newTRAIN only; no DEV/TEST used or scored. Not model generalization or new dataset examples.',
  'variable':'Output vocabulary cardinality only. Same exact target strings, features, TRAIN-fitted normalization, backbone initialization and common output rows. Large output adds labels not present in these8targets.',
  'full_classes':len(full_vocab),'restricted_classes':len(small_vocab),'max_optimizer_updates':100,'stop_train_wer':.05,'stop_loss_ratio':.05,'wall_cap_seconds_per_trial':30,
  'optimizer':'AdamW lr.001,wd.0001,clip5,FP32,batch8','common_initial_logit_max_error':common_error,'trained_recognition_weights_reused':False})
 results={}
 for name,model,vocab in [('full_vocab',full,full_vocab),('restricted_vocab',small,small_vocab)]:
  opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);ctc=torch.nn.CTCLoss(blank=0,zero_infinity=False);initial=evaluate(model,selected,data,vocab,'cuda:0');curve=[]
  started=time.perf_counter();best=(float('inf'),float('inf'));best_state=None;steps=0;torch.cuda.reset_peak_memory_stats()
  for update in range(1,101):
   if time.perf_counter()-started>=30:break
   model.train();x,y,l,tl=batch(selected,data,vocab,'cuda:0');opt.zero_grad(set_to_none=True);logits=model(x);loss=ctc(logits.log_softmax(-1).transpose(0,1),y,l,tl)
   if not torch.isfinite(loss):raise ValueError('NONFINITE_CTC')
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();steps=update
   if update==1 or update%10==0:
    score=evaluate(model,selected,data,vocab,'cuda:0');curve.append({'update':update,'train_wer':score['gloss_wer'],'train_loss':score['loss'],'prediction_tokens':sum(len(p['hypothesis']) for p in score['predictions'])})
    rank=(score['gloss_wer'],score['loss'])
    if rank<best:best=rank;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_score=score
    if score['gloss_wer']<=.05 and score['loss']<=initial['loss']*.05:break
  path=out/(name+'.pt');torch.save({'schema':SCHEMA,'model':best_state,'vocabulary':vocab,'optimizer_updates':steps,'scope':'TRAIN_ONLY_CARDINALITY_PROBE','common_initial_logits_error':common_error},path)
  results[name]={'updates':steps,'seconds':time.perf_counter()-started,'initial':{k:v for k,v in initial.items() if k!='predictions'},'best':best_score,'curve':curve,'checkpoint_sha256':sha(path),'peak_allocated_vram_bytes':torch.cuda.max_memory_allocated(),'used_cuda':True}
  print(json.dumps({'trial':name,'updates':steps,'best_train_wer':best_score['gloss_wer'],'seconds':results[name]['seconds']}),flush=True)
 write(out/'results.json',{'status':'COMPLETED_CONTROLLED_TRAIN_ONLY_PROBE','trials':results,'generalization_claim':False,'DEV_or_TEST_used':False,'new_unique_examples_added':0,'production_promoted':False})

if __name__=='__main__':main()
