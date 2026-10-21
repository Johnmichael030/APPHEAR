"""Read-only completed-mission verifier. Does not enumerate or open reserved TEST."""
from pathlib import Path
import csv,json,hashlib,sys
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def validate(require_seal=True):
 required=['BASELINE_AND_ERROR_ANALYSIS.md','VOCABULARY_EXPANSION_PLAN.csv','VOCABULARY_EXPANSION_REPORT.md','DATA_READINESS_AND_SPLITS.md','FINETUNING_PLAN.md','EXPERIMENT_RESULTS.csv','A/TRAINING_CONFIGURATION.json','B/TRAINING_CONFIGURATION.json','C/TRAINING_CONFIGURATION.json','PREDICTION_EXAMPLES.csv','ALIGNMENTS.csv','PROMOTION_DECISION.md','commands_and_results.txt','MAPPING_DIFF.csv','B/EXPANSION_IMPLEMENTATION_TESTS.json','B/SYNTHETIC_OPTIMIZER_SMOKE.json']
 assert all((R/p).is_file() for p in required),'MISSING_DELIVERABLE'
 protected=read(R/'protected_hashes.json');assert all(sha(p)==s for p,s in protected.items()),'PROTECTED_MUTATION'
 features=read(R/'input_features.json');assert len(features)==1580 and all(sha(r['path'])==r['sha256'] for r in features),'FEATURE_MUTATION'
 cfg=read(R/'A/TRAINING_CONFIGURATION.json')
 for n,s in cfg['manifest_sha256'].items():assert sha(R/'manifests'/(n+'.json'))==s
 assert sha(R/'vocabulary.json')==cfg['vocabulary_sha256'] and sha(cfg['initialization'])==cfg['initialization_sha256']
 result=read(R/'A/RESULTS.json');selected=result['selected_checkpoint'];assert sha(R/selected['path'])==selected['sha256']
 assert result['completed_epochs']==10 and result['optimizer_updates']==1880 and not result['TEST_used'] and not result['production_promoted']
 assert result['wall_training_and_epoch_eval_seconds']<cfg['wall_seconds'] and result['peak_allocated_bytes']<cfg['VRAM_cap_bytes']
 v=read(R/'vocabulary.json')['tokens'];assert v[:2]==['<BLANK>','<UNK>'] and len(v)==2289
 with (R/'MAPPING_DIFF.csv').open(newline='',encoding='utf-8') as f:mapping=list(csv.DictReader(f))
 assert len(mapping)==len(v) and all(int(r['old_id'])==int(r['new_id'])==i and r['old_label']==r['new_label']==v[i] for i,r in enumerate(mapping))
 for p in ['baseline_greedy_DEV80.json','baseline_streaming_DEV80.json','A/selected_DEV80.json','A/selected_streaming_DEV80.json']:
  score=read(R/p);pred=score['predictions'];assert len(pred)==80 and sum(x['reference_tokens'] for x in pred)==485
  assert abs(sum(sum(x[k] for k in ['substitutions','deletions','insertions']) for x in pred)/485-score['WER'])<1e-12
  assert sum(x['reference']==x['hypothesis'] for x in pred)==score['exact_sequences']==0
 assert read(R/'B/SYNTHETIC_OPTIMIZER_SMOKE.json')['actual_new_classes']==0
 assert read(R/'DATA_INVENTORY.json')['actual_new_classes_added']==0
 assert not (R/'B/RESULTS.json').exists() and not result['promising_for_C']
 assert read(R/'decoder/CORRECTNESS_TESTS.json')['repeat_after_blank'] is True
 assert read(R/'A/SMOKE_RESULT.json')['restored_source_before_main_fit'] is True
 import torch
 torch.set_num_threads(2)
 checkpoints=sorted((R/'A/checkpoints').glob('epoch_*.pt'));assert len(checkpoints)==10
 source=torch.load(cfg['initialization'],map_location='cpu',weights_only=False)
 for p in checkpoints:
  state=torch.load(p,map_location='cpu',weights_only=False)
  assert state['vocabulary']==v and state['config_sha256']==sha(R/'A/TRAINING_CONFIGURATION.json')
  assert state['model']['output.weight'].shape==(2289,48) and 'optimizer' in state and 'cuda_rng' in state
  assert torch.equal(state['model']['mean'],source['model']['mean']) and torch.equal(state['model']['std'],source['model']['std'])
 hashed=0
 if require_seal:
  for line in (R/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
   s,relative=line.split('  ',1);p=(R/relative).resolve();assert R==p or R in p.parents
   assert sha(p)==s,'SEALED_ARTIFACT_MUTATION:'+relative;hashed+=1
 return dict(status='PASS_INTEGRITY_PARTIAL_RECOGNITION',protected_files=len(protected),feature_files=len(features),sealed_artifacts=hashed,A_completed=True,B_actual_new_classes=0,C_not_run=True,production_promoted=False,TEST_accessed=False,device_modified=False)
if __name__=='__main__':print(json.dumps(validate(),indent=2))
