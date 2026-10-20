from pathlib import Path
from collections import Counter
import csv,hashlib,json,re
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[2];TREE=RUN.parent
DESIGN=TREE/'20261020T131348Z_source_signer_audit01';PARENT=TREE/'20261020T104054Z_expand_eval01'
SCHEMA='apphear.continuous_asl.candidate01_causal378.v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(name,obj):
 p=(RUN/name).resolve();assert RUN in p.parents;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2);f.write('\n')
def table(name,rows):
 with (RUN/name).open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def family(r):
 m=re.match(r'^(\d{4}-\d{2}-\d{2})_',r['source_group']);return r['signer_id']+'::'+(m[1] if m else 'UNRESOLVED_ALL_UNDATED_MASTERS')
def summarize(predictions):
 c=Counter();hyp=Counter();blank=0;steps=0;loss=[]
 for p in predictions:
  for k in ['substitutions','deletions','insertions','reference_tokens']:c[k]+=p[k]
  hyp.update(p['hypothesis']);blank+=p['blank_frames'];steps+=p['output_frames']
  if p['CTC_loss'] is not None:loss.append(p['CTC_loss'])
 top,n=hyp.most_common(1)[0] if hyp else ('',0)
 return {'gloss_WER':(c['substitutions']+c['deletions']+c['insertions'])/c['reference_tokens'],**dict(c),
  'CTC_loss':sum(loss)/len(loss) if len(loss)==len(predictions) else None,'CTC_loss_samples':len(loss),'utterances':len(predictions),
  'blank_argmax_fraction':blank/steps,'output_frames':steps,'predicted_tokens':sum(hyp.values()),'unique_predicted_tokens':len(hyp),
  'top_predicted_token':top,'top_predicted_token_fraction':n/sum(hyp.values()) if hyp else 0.,'hypothesis_token_counts':dict(hyp),
  'empty_hypotheses':sum(not p['hypothesis'] for p in predictions),'multi_token_hypotheses':sum(len(p['hypothesis'])>=2 for p in predictions),
  'hypothesis_length_histogram':dict(Counter(len(p['hypothesis']) for p in predictions)),'predictions':predictions}
