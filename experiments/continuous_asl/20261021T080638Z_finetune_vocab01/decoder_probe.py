"""No-language-model CTC prefix beam search; exhaustive tiny-case correctness test."""
import itertools,json,time
from collections import defaultdict
import numpy as np
import fine_tune as ft

def beam(logits,width,topk=16,return_score=False):
 x=np.asarray(logits,dtype=np.float64);x=x-x.max(-1,keepdims=True);logp=x-np.log(np.exp(x).sum(-1,keepdims=True))
 states={(): (0.,-np.inf)}
 for row in logp:
  selected=set(np.argsort(row)[-min(topk,len(row)):].tolist());selected.add(0)
  new=defaultdict(lambda:[-np.inf,-np.inf])
  for prefix,(pb,pnb) in states.items():
   total=np.logaddexp(pb,pnb);new[prefix][0]=np.logaddexp(new[prefix][0],total+row[0])
   allowed=selected | ({prefix[-1]} if prefix else set())
   for token in allowed:
    if token==0:continue
    if prefix and token==prefix[-1]:
     new[prefix][1]=np.logaddexp(new[prefix][1],pnb+row[token]);q=prefix+(token,);new[q][1]=np.logaddexp(new[q][1],pb+row[token])
    else:
     q=prefix+(token,);new[q][1]=np.logaddexp(new[q][1],total+row[token])
  states=dict(sorted(new.items(),key=lambda kv:(-np.logaddexp(*kv[1]),kv[0]))[:width])
 best=min(states,key=lambda p:(-np.logaddexp(*states[p]),p));score=float(np.logaddexp(*states[best]))
 return (best,score) if return_score else list(best)

def tests():
 rng=np.random.default_rng(173)
 for n in [1,2,3,4,5]:
  z=rng.normal(size=(n,3));p=np.exp(z-z.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True);mass=defaultdict(float)
  for path in itertools.product(range(3),repeat=n):
   seq=[];previous=-1
   for token in path:
    if token!=previous and token!=0:seq.append(token)
    previous=token
   mass[tuple(seq)]+=float(np.prod([p[t,token] for t,token in enumerate(path)]))
  expected=min(mass,key=lambda s:(-mass[s],s));actual,score=beam(z,1000,3,True)
  assert actual==expected and abs(score-np.log(mass[expected]))<1e-10
 assert beam(np.array([[0,20,-20],[20,0,-20],[0,20,-20]],float),100,3)==[1,1]
 return dict(status='PASS',exhaustive_frames=[1,2,3,4,5],blank_id=0,repeat_after_blank=True,language_model=False)

def main():
 out=ft.RUN/'decoder';out.mkdir(exist_ok=True)
 if (out/'RESULTS.json').exists():print('DECODER_ALREADY_COMPLETE');return
 ft.save(out/'CONFIGURATION.json',dict(widths=[4,8],topk=16,blank_id=0,UNK_id=1,language_model=False,length_penalty=0,temperature=1,scope='Full-sequence DEV80 CTC marginalization; not current streaming display policy',model_checkpoint_selection='A remains selected by fixed greedy DEV WER; beam never selects its checkpoint'))
 ft.save(out/'CORRECTNESS_TESTS.json',tests())
 vocab=ft.read(ft.RUN/'vocabulary.json')['tokens'];rows=ft.read(ft.RUN/'manifests/DEV80.json');results=[];predictions=[]
 checkpoints=[('baseline',ft.RUN/'baseline_logits_DEV80.npz')]
 candidate=ft.RUN/'A/selected_logits_DEV80.npz'
 if candidate.exists():checkpoints.append(('A',candidate))
 for name,path in checkpoints:
  with np.load(path,allow_pickle=False) as cache:
   assert set(cache.files)=={r['sample_id'] for r in rows}
   for width in [0,4,8]:
    total=dict(substitutions=0,deletions=0,insertions=0,reference_tokens=0);exact=0;tokens=set();start=time.perf_counter()
    for r in rows:
     z=cache[r['sample_id']]
     if width==0:
      previous=-1;ids=[]
      for token in z.argmax(-1):
       token=int(token)
       if token!=previous and token!=0:ids.append(token)
       previous=token
     else:ids=beam(z,width)
     hyp=[vocab[i] for i in ids];edits=ft.edit_counts(r['glosses'],hyp)
     for k,v in edits.items():total[k]+=v
     exact+=hyp==r['glosses'];tokens.update(hyp);predictions.append(dict(model=name,decoder='greedy' if width==0 else f'beam{width}',sample_id=r['sample_id'],reference=' '.join(r['glosses']),hypothesis=' '.join(hyp),**edits))
    result=dict(model=name,decoder='greedy' if width==0 else f'beam{width}',WER=(total['substitutions']+total['deletions']+total['insertions'])/total['reference_tokens'],**total,exact_sequences=exact,token_diversity=len(tokens),CPU_seconds=time.perf_counter()-start);results.append(result);print(json.dumps(result),flush=True)
 ft.writecsv(out/'DECODER_COMPARISON.csv',results);ft.writecsv(out/'PREDICTIONS.csv',predictions);ft.save(out/'RESULTS.json',dict(status='COMPLETE',models=[n for n,p in checkpoints],results=results,TEST_used=False,production_decoder_changed=False))
if __name__=='__main__':main()
