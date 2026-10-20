"""Read-only integrity, split and metric checks; no torch/model execution."""
from common import *
import subprocess,sys
count=0
for line in (RUN/'artifact_hashes.sha256').read_text().splitlines():
 expected,name=line.split('  ',1);p=(RUN/name).resolve();assert RUN in p.parents and sha(p)==expected,name;count+=1
for line in (RUN/'manifest_hashes.sha256').read_text().splitlines():
 expected,name=line.split('  ',1);assert sha(RUN/name)==expected,name
for path,expected in read(RUN/'input_hashes.json').items():assert sha(path)==expected,path
source=DESIGN/'folds/A_guarded_pooled_family_holdout.json';fold=read(source);assert sha(source)==sha(RUN/'fold_manifest.json')=='87442a6f1241860e369a5c8f0768d583b3a4058f5a9ad03dcc135532c419c72f'
resolved=read(RUN/'resolved_manifest.json');lookup={r['sample_id']:r for r in resolved};assert len(lookup)==1706
fit=[lookup[i] for i in fold['train_sample_ids']];held=[lookup[i] for i in fold['internal_eval_sample_ids']]
assert len(fit)==1500 and len(held)==206 and not ({family(r) for r in fit}&{family(r) for r in held})
for key in ['sample_id','source_group','collection','raw_sha256','decoded_pixel_sha256']:assert not ({r.get(key) for r in fit if r.get(key)}&{r.get(key) for r in held if r.get(key)})
vocab=read(RUN/'vocabulary.json')['tokens'];assert vocab==['<BLANK>','<UNK>']+sorted({t for r in fit for t in r['glosses']})
assert all(set(lookup[i]['glosses'])<=set(vocab) for i in fold['compatible_eval_sample_ids'])
s=read(RUN/'training_status.json');assert s['used_cuda'] and not s['TEST_used'] and not s['production_promoted'] and s['completed_epochs']<=20
for pointer in [s['initialization_checkpoint'],s['selected_checkpoint']]:assert sha(RUN/pointer['path'])==pointer['sha256']
for key,ids in [('primary80',fold['compatible_eval_sample_ids']),('strong27',fold['support5_eval_sample_ids'])]:
 score=read(RUN/('selected_'+key+'.json'));assert {p['sample_id'] for p in score['predictions']}==set(ids)
 recalculated=summarize(score['predictions']);assert recalculated['gloss_WER']==score['gloss_WER'] and recalculated['CTC_loss']==score['CTC_loss']
assert len(fold['compatible_eval_sample_ids'])==80 and len(fold['support5_eval_sample_ids'])==27
subprocess.run([sys.executable,'-B',str(DESIGN/'verify.py')],check=True)
print(json.dumps({'status':'PASS_PILOT_AND_FROZEN_PRESERVATION','new_artifacts_hashed':count,'epochs':s['completed_epochs'],'primary80_WER':s['primary80_WER'],'strong27_WER':s['strong27_WER'],'TEST_used':False,'promoted':False}))
