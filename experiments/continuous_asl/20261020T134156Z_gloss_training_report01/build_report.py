"""Read-only corpus/run inventory; plain-text and HTML reports, no model execution."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone,timedelta
import hashlib,html,json,subprocess
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[2];TREE=RUN.parent
PARENT=TREE/'20261020T104054Z_expand_eval01';LATEST=TREE/'20261020T133148Z_source_transfer_pilot01'
INPUTS={}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):
 p=Path(p);INPUTS[str(p)]=sha(p);return json.loads(p.read_text(encoding='utf-8-sig'))
def save(name,value):
 with (RUN/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def text(name,value):
 with (RUN/name).open('x',encoding='utf-8') as f:f.write(value.rstrip()+'\n')
def alphabet(labels):return sorted(labels,key=lambda t:(t.casefold(),t))
def labelset(rows):return {t for r in rows for t in r['glosses']}
def literal_list(name,title,labels,notes):
 ordered=alphabet(labels)
 assert all('\n' not in t and '\r' not in t and '\t' not in t for t in ordered)
 text(name,f'{title}\nTotal distinct labels: {len(ordered):,}\n{notes}\nRow numbers are list positions, not model token IDs. Literal case, suffixes, fingerspelling and classifier notation are preserved.\n\n'+'\n'.join(f'{i}. {t}' for i,t in enumerate(ordered,1)))

def main():
 eligible=read(PARENT/'eligible_corpus/manifest.json');assert len(eligible)==3121
 canonical={r['sample_id']:r for r in eligible}
 maina=read(TREE/'20261020_continuous_expand08/data/processed/train_dev/manifest.json');mainn=read(TREE/'20261020_continuous_expand09/data/processed/train_dev/manifest.json')
 a=[r for r in maina if r['split']=='train'];n=[r for r in mainn if r['split']=='train'];assert (len(a),len(n))==(2119,906)
 trained={r['sample_id']:r for r in a+n};assert len(trained)==3025
 all_labels=labelset(eligible);trained_labels=labelset(list(trained.values()))
 lateststatus=read(LATEST/'training_status.json');latestrows=read(LATEST/'resolved_manifest.json');latestfit=[r for r in latestrows if r['internal_role']=='TRAIN']
 latestvocab=read(LATEST/'vocabulary.json')['tokens'];assert latestvocab[:2]==['<BLANK>','<UNK>'] and len(latestvocab)==2289
 assert set(latestvocab[2:])==labelset(latestfit)
 signer_lookup={r['sample_id']:r['signer_id'] for r in read(PARENT/'data/processed/train_dev/manifest.json')}
 # Collect every saved status without traversing package internals or executing any fit.
 result=subprocess.run(['rg','--files',str(TREE),'-g','training_status.json'],capture_output=True,text=True,check=True)
 statuses=sorted(Path(p) for p in result.stdout.splitlines());register=[];raw_evidence=[];used_by=defaultdict(list);historical_labels=set();historical_ids=set()
 manifests={
  '20261019T155428Z_cslr01':TREE/'20261019T155428Z_cslr01/data/processed/full/manifest.json',
  '20261020T004837_separate02':TREE/'20261020T004837_separate02/experiment_B/data/processed/full/manifest.json'}
 for key in ['03','04','05','08','09']:manifests['20261020_continuous_expand'+key]=TREE/f'20261020_continuous_expand{key}/data/processed/train_dev/manifest.json'
 cached={};main_register=[]
 for p in statuses:
  st=read(p);rel=p.relative_to(TREE).as_posix();parts=p.relative_to(TREE).parts;runroot=parts[0];isolated='experiment_A' in parts;smoke=any(q.startswith('smoke') for q in parts)
  fitrows=None;evalrows=None;manifest=None;scope='ISOLATED_SIGN_OUTSIDE_CONTINUOUS_TOTALS' if isolated else 'CONTINUOUS_ASL'
  if not isolated:
   if runroot in manifests:
    manifest=manifests[runroot]
    if manifest not in cached:cached[manifest]=read(manifest)
    rr=cached[manifest];fitrows=[r for r in rr if r['split']=='train'];evalrows=[r for r in rr if r['split'] in ['dev','val']]
    if smoke:fitrows=fitrows[:4];evalrows=evalrows[:2]
   elif runroot=='20261020T102748Z_continuation_pilot01':
    manifest=TREE/'20261020_continuous_expand08/data/processed/train_dev/manifest.json';fitrows=a;evalrows=[r for r in maina if r['split']=='dev']
   elif runroot=='20261020T104054Z_expand_eval01':
    manifest=PARENT/'data/processed/train_dev/manifest.json';rr=read(manifest);fitrows=[r for r in rr if r['split']=='train'];evalrows=[r for r in rr if r['split']=='dev']
   elif runroot=='20261020T123237Z_collapse_diagnosis01':
    manifest=p.parent/'manifest.json';rr=read(manifest);fitrows=[r for r in rr if r['split']=='train'];evalrows=[r for r in rr if r['split']=='dev']
   elif runroot=='20261020T124813Z_label_support01':
    manifest=p.parents[1]/'frozen_manifest.json';rr=read(manifest);fitrows=[r for r in rr if r['split']=='train'];evalrows=read(p.parents[1]/'compatible_dev.json')
   elif runroot==LATEST.name:
    manifest=LATEST/'fold_manifest.json';fitrows=latestfit;lookup={r['sample_id']:r for r in latestrows};evalrows=[lookup[i] for i in read(manifest)['compatible_eval_sample_ids']]
  epoch=st.get('completed_epochs',st.get('completed_additional_epochs',st.get('epochs_completed')))
  score=st.get('primary80_WER',st.get('compatible_DEV_WER',st.get('DEV_WER',st.get('best_dev_wer',st.get('best_new_dev_wer')))))
  for k in ['best_dev_score','best_validation_score','best_score']:
   if score is None and k in st and not isolated:score=st[k][0]
  runtime=st.get('epoch_loop_seconds',st.get('fit_loop_seconds',st.get('seconds')))
  peak=st.get('peak_allocated_VRAM_bytes',st.get('peak_allocated_vram_bytes'))
  pointer=st.get('selected_checkpoint',st.get('best_checkpoint'));cp=None;cp_hash=None
  if isinstance(pointer,dict) and pointer.get('path'):
   cp=p.parent/pointer['path'];cp_hash=pointer.get('sha256')
  elif (p.parent/'checkpoints/best.json').exists():
   pointer=read(p.parent/'checkpoints/best.json');cp=p.parent/pointer['path'];cp_hash=pointer.get('sha256')
  if cp and cp.exists() and cp_hash:assert sha(cp)==cp_hash, ('CHECKPOINT_HASH_MISMATCH',str(cp))
  verified_checkpoint=bool(cp and cp.exists() and cp_hash)
  vocab_path=p.parent/'vocabulary.json';classes=st.get('vocabulary_size')
  if classes is None and vocab_path.exists():
   vx=read(vocab_path);classes=len(vx['tokens']) if isinstance(vx,dict) and 'tokens' in vx else len(vx) if isinstance(vx,list) else None
  row={'run':rel.removesuffix('/training_status.json'),'scope':scope,'smoke':smoke,'status':st['status'],'completed_epochs':epoch,
   'actual_TRAIN_utterances':len(fitrows) if fitrows is not None else None,'actual_TRAIN_gloss_tokens':sum(len(r['glosses']) for r in fitrows) if fitrows is not None else None,
   'actual_TRAIN_target_labels':len(labelset(fitrows)) if fitrows is not None else None,'selection_evaluation_utterances':len(evalrows) if evalrows is not None else None,
   'selected_evaluation_WER':score if not isolated else None,'output_classes_including_specials':classes,
   'runtime_seconds_reported':runtime,'runtime_scope':'epoch/fit loop when present; legacy seconds field otherwise; not all timing definitions identical',
   'peak_allocated_VRAM_bytes':peak,'used_CUDA':st.get('used_cuda',st.get('training_used_gpu')),'checkpoint_path':str(cp) if cp else None,
   'checkpoint_sha256':cp_hash,'checkpoint_hash_verified':verified_checkpoint,'status_path':str(p),'status_sha256':INPUTS[str(p)],'manifest_path':str(manifest) if manifest else None,
   'manifest_sha256':sha(manifest) if manifest else None,'counts_scope':'actualoptimizerTRAIN subset; smoke uses first4TRAIN/2DEV per saved trainer' if smoke else 'explicit frozen fit membership; not new independent data per run'}
  if fitrows is not None:
   for r in fitrows:
    assert r['sample_id'] in canonical and r['glosses']==canonical[r['sample_id']]['glosses']
    historical_ids.add(r['sample_id']);historical_labels.update(r['glosses']);used_by[r['sample_id']].append(row['run'])
  register.append(row);raw_evidence.append({'path':str(p),'sha256':INPUTS[str(p)],'saved_training_status':st})
 assert historical_ids==set(trained) and historical_labels==trained_labels
 # Separate probes have no training_status file: preserve them as diagnostic evidence, not added data.
 probe=read(PARENT/'vocab_cardinality_probe/results.json');expand8=read(TREE/'20261020_continuous_expand08/execution_summary.json')
 save('DIAGNOSTIC_PROBES.json',{'vocabulary_cardinality_8TRAIN_probe':probe,'clean_8TRAIN_overfit_diagnostic':expand8['clean_overfit_diagnostic'],
  'meaning':'Repeated TRAIN-only memorization diagnostics; add zero independent utterances and establish no held-out recognition accuracy.'})
 save('TRAINING_RUNS.json',register);save('TRAINING_STATUS_EVIDENCE.json',raw_evidence)
 pools={};counts=[];label_info={t:[] for t in all_labels}
 for pool in sorted({r['training_pool'] for r in eligible}):
  rr=[r for r in eligible if r['training_pool']==pool];fitted=[r for r in rr if r['sample_id'] in trained];ef=Counter(t for r in rr for t in r['glosses']);tf=Counter(t for r in fitted for t in r['glosses']);us=Counter(t for r in rr for t in set(r['glosses']));tus=Counter(t for r in fitted for t in set(r['glosses']))
  short='ASLLRP3' if pool.startswith('ASLLRP') else 'NCSLGR2';pools[short]=set(ef)
  counts.append({'pool':pool,'eligible_utterances':len(rr),'eligible_gloss_labels':len(ef),'eligible_gloss_token_occurrences':sum(ef.values()),'ever_fitted_unique_utterances':len(fitted),
   'ever_fitted_distinct_labels':len(tf),'ever_fitted_gloss_tokens':sum(tf.values()),'prepared_not_fitted_utterances':len(rr)-len(fitted),
   'label_contract':'literal dominant realized occurrence' if pool.startswith('ASLLRP') else 'literal main gloss; legacy/exactEAF temporal strata retained'})
  for t in ef:label_info[t].append({'pool':short,'eligible_token_occurrences':ef[t],'eligible_utterance_support':us[t],'fitted_token_occurrences':tf[t],'fitted_utterance_support':tus[t]})
  literal_list(short+'_FULL_GLOSS_LIST.txt',short+' — complete eligible gloss-label list',set(ef),'Includes prepared eligible examples; separate pool. This is not a merged model vocabulary.')
 literal_list('ALL_ELIGIBLE_GLOSS_LABELS.txt','AppHear — complete eligible continuous-ASL gloss spellings',all_labels,'Alphabetical display union across two separate label pools. Shared spellings appear once; no semantic ontology merge or extra training occurred.')
 literal_list('ALL_TRAINED_GLOSS_LABELS.txt','AppHear — complete gloss spellings used in verified continuous-ASL fits',trained_labels,'Distinct literal target spellings across completed continuous-ASL fits. This does not mean every label is recognized correctly. Repeated epochs/runs do not add new labels or utterances.')
 text('LATEST_PILOT_VOCABULARY.txt',f'Latest source-transfer pilot: {len(latestvocab)-2:,} gloss labels; {len(latestvocab):,} output classes\nFixed mapping in the saved model; special blank0 and UNK1 are not gloss words.\n\n'+'\n'.join(f'{i}\t{t}' for i,t in enumerate(latestvocab)))
 save('GLOSS_LABELS_AND_COUNTS.json',[{'gloss':t,'ever_used_as_TRAIN_target':t in trained_labels,'pool_counts':label_info[t]} for t in alphabet(all_labels)])
 with (RUN/'UTTERANCE_DATA_INVENTORY.jsonl').open('x',encoding='utf-8') as f:
  for r in sorted(eligible,key=lambda r:r['sample_id']):
   entry={'sample_id':r['sample_id'],'pool':r['training_pool'],'dataset':r['dataset'],'source_group':r['source_group'],'collection':r.get('collection'),
    'signer_id':signer_lookup.get(r['sample_id'],r.get('signer_id')),'glosses':r['glosses'],'gloss_tokens':len(r['glosses']),'observations':r['observations'],
    'feature_path':r['feature_path'],'feature_sha256':r['feature_sha256'],'ever_used_in_verified_continuous_fit':r['sample_id'] in trained,'saved_run_memberships':used_by.get(r['sample_id'],[]),
    'inventory_note':'Membership repeats across runs; count this utterance once. No reserved TEST content is exported.'}
   f.write(json.dumps(entry,ensure_ascii=False)+'\n')
 summary={'created_Asia_Shanghai':datetime.now(timezone(timedelta(hours=8))).isoformat(),'eligible_utterances':len(eligible),'eligible_unique_gloss_spellings':len(all_labels),
  'eligible_gloss_token_occurrences':sum(len(r['glosses']) for r in eligible),'ever_fitted_unique_utterances':len(trained),'ever_fitted_unique_gloss_spellings':len(trained_labels),
  'ever_fitted_gloss_token_occurrences_counting_each_utterance_once':sum(len(r['glosses']) for r in trained.values()),'prepared_not_fitted_utterances':len(eligible)-len(trained),
  'shared_exact_spellings_between_eligible_pools':len(pools['ASLLRP3']&pools['NCSLGR2']),'pools':counts,'latest_pilot':lateststatus,
  'saved_status_records':len(register),'continuous_status_records':sum(r['scope']=='CONTINUOUS_ASL' for r in register),'isolated_status_records_not_in_continuous_totals':sum(r['scope']!='CONTINUOUS_ASL' for r in register),
  'special_tokens_excluded_from_gloss_totals':['<BLANK>','<UNK>'],'production_promoted':False,'TEST_contents_exported':False}
 save('COUNTS.json',summary)
 pool_table='\n'.join(f"|{c['pool'].split('_')[0]}|{c['eligible_utterances']:,}|{c['eligible_gloss_labels']:,}|{c['ever_fitted_unique_utterances']:,}|{c['ever_fitted_distinct_labels']:,}|{c['prepared_not_fitted_utterances']}|" for c in counts)
 continuous=[r for r in register if r['scope']=='CONTINUOUS_ASL'];main=[r for r in continuous if not r['smoke']]
 run_table='\n'.join(f"|{r['run']}|{r['actual_TRAIN_utterances']}|{r['actual_TRAIN_target_labels']}|{r['selection_evaluation_utterances']}|{r['completed_epochs']}|{100*r['selected_evaluation_WER']:.3f}%|{r['status']}|" for r in main if r['selected_evaluation_WER'] is not None)
 text('FULL_TRAINING_REPORT.md',f'''# AppHear continuous-ASL data and training report

Generated {summary['created_Asia_Shanghai']} (Asia/Shanghai), from frozen manifests, saved status files, predictions and checkpoints. No training, download, production change or TEST-content export was performed for this report.

## What “continuous words” means here

The dataset targets are **ASL gloss labels**, not a count of ordinary English words or reliably recognized signs. They include lexical signs, classifier descriptions, pointing/locus markers, one/two-hand variants, repetitions and whole fingerspelled words/names. Spelling/case/notation are preserved exactly. Blank and UNK are model-control tokens and are excluded from gloss totals.

There are **{len(all_labels):,} distinct eligible gloss spellings** across both separate pools, and **{len(trained_labels):,} distinct spellings actually used as targets in completed continuous-ASL fits**. The display union deduplicates shared spellings; it does not prove semantic target compatibility or authorize mixing pools. The full eligible pool lists sum to3,801labels, of which431spellings occur in both, giving3,370distinct spellings.

## Data totals

|Pool|Eligible utterances|Eligible gloss labels|Unique utterances actually fitted|Fitted gloss labels|Prepared, not fitted|
|---|---|---|---|---|---|
{pool_table}

- Eligible: **3,121unique utterances**, **23,002gloss-token occurrences** counting each utterance once.
- Actually fitted across completed runs: **3,025unique utterances**, **22,586target-token occurrences** counting each utterance once.
- Prepared but not fitted: **96NCSLGR utterances**, containing416token occurrences. They add142NCSLGR-pool labels,114spellings not already present anywhere in the fitted display union.
- Repeated epochs, warm continuations, smoke tests and diagnostic probes reuse data; their presentations are not additional independent examples.
- Actual fitting membership was reconstructed from all saved continuous run manifests and confirmed to equal the2,119ASLLRP3 plus906NCSLGR union. Historical target strings match the current eligible catalog for those IDs.

The current working ASLLRP split is1,706TRAIN/413DEV; all2,119were TRAIN in an older experiment. Thus “ever fitted” includes records now used as development data. New source-transfer evaluation was valid only because its model started fresh. None of these observations establishes pristine unseen-signer TEST accuracy.

## Complete lists and inventory

- ALL_ELIGIBLE_GLOSS_LABELS.txt — all3,370eligible literal spellings.
- ALL_TRAINED_GLOSS_LABELS.txt — all3,256spellings actually fitted.
- ASLLRP3_FULL_GLOSS_LIST.txt —2,739eligible pool labels.
- NCSLGR2_FULL_GLOSS_LIST.txt —1,062eligible pool labels (920used in completed NCSLGR fits).
- LATEST_PILOT_VOCABULARY.txt — exact latest model IDs:2,287TRAIN gloss labels plus blank/UNK,2,289classes.
- GLOSS_LABELS_AND_COUNTS.json — per-label pool membership, token frequencies and distinct-utterance support, separately for eligible and actually fitted data.
- UTTERANCE_DATA_INVENTORY.jsonl — all3,121stable IDs, ordered labels, source/signer metadata, feature hashes, fitted flag and saved run memberships.

The standalone HTML report includes a searchable complete3,370-label inventory. Report list row numbers are not model token IDs. Only LATEST_PILOT_VOCABULARY.txt gives the frozen current model mapping.

## Completed continuous research runs

Each row below is a saved completed main/bounded run. TRAIN count is actual fit membership; a model's output vocabulary can be larger than labels seen by a tiny smoke subset. Full status evidence, hashes, checkpoint paths, runtime and memory are in TRAINING_RUNS.json and TRAINING_STATUS_EVIDENCE.json. Smoke runs are included there separately, with4TRAIN/2DEV as specified by their saved trainer. Isolated-sign ExperimentA status files are retained as out-of-scope evidence and are excluded from continuous totals.

|Run|TRAIN utterances|TRAIN target labels|Selection eval utterances|Epochs completed|Selected eval WER|Stop status|
|---|---|---|---|---|---|---|
{run_table}

There are{len(continuous)}saved continuous status records, including smoke runs, plus{summary['isolated_status_records_not_in_continuous_totals']}isolated-sign status records outside these totals. Some exploratory overfit/cardinality probes use separate result files rather than training_status.json; they are preserved in DIAGNOSTIC_PROBES.json and add zero new data. Eight-sample memorization is not generalization.

Earlier experiments have different evaluation IDs, target pools, vocabulary and normalization. Their WER values are **not a comparable leaderboard**. The old85.96%nine-NCSLGR result cannot be ranked against the latest80-source-family result. The 48/24channel experiments only have architecture-comparison scope where their own frozen protocol establishes equivalence.

## Latest authorized source-transfer pilot

Frozen internalTRAIN1,500utterances with2,287gloss labels and12,423target tokens; primary80fully compatible held-out sequences; stronger27as a subgroup; full206heldout reported for coverage context. Only ASLLRP3 is used. Same known signers appear in TRAIN and evaluation; recording families are disjoint under the conservative guard. This is not unseen-signer generalization.

- Completed14epochs /2,632optimizer updates; stoppedNO_CREDIBLE_PRIMARY_GAIN.
- Selected epoch12: TRAINWER87.0804%; primary80WER96.9072%; stronger27WER96.6216%; full206WER97.8818%.
- Epoch loop88.502s; including reload/fullTRAIN validation96.487s. RTX3050Laptop, peak allocated1,303,238,656bytes (1.214GiB).
- Best local checkpoint: `{LATEST/lateststatus['selected_checkpoint']['path']}`.
- Checkpoint SHA256: `{lateststatus['selected_checkpoint']['sha256']}`.
- Frozen source split SHA256: `{lateststatus['fold_manifest_sha256']}`.
- TRAIN-learning and transfer screens both failed. Further training is not justified by this run; poor heldout performance cannot establish a source effect while the TRAIN-learning screen fails. No checkpoint was promoted.

## Data contracts and remaining limits

ASLLRP3 uses literal dominant realized-occurrence labels; NCSLGR2 uses literal main-gloss labels. Both use the verified causal378feature schema in the current catalog, but shared feature dimensions or spellings do not make their supervision equivalent. NCSLGR legacy time calibration remains limited. The latest ASLLRP model uses the same causal48channelCTC family, TRAIN-only mean/std and vocabulary, fixed greedy decoding, blank0/UNK1, FP32 and research-only CUDA.

Known publisher signer metadata exists for ASLLRP. NCSLGR has988eligible records without explicit signer IDs; do not infer people from filenames or images. Independent source-dataset and signer effects remain unidentifiable with the current bridge/recording structure. Continuous streaming false-activation, boundary and end-to-end latency acceptance remain unverified; the current report describes utterance-supervised research.

How2Sign has no authenticated gloss supervision in the verified local package and is not counted as fitted gloss data. The RIT archive has a preserved timing/frame-origin blocker and contributes zero eligible training examples. Prior exclusions (six ASLLRP overlaps, frame-count mismatch, NCSLGR overlaps) and17reserved TEST utterances remain protected. TEST labels/predictions were not used to build these lists.

**Status: PASS data/report integrity; PARTIAL recognition; BLOCKED model promotion.** Current lists enumerate data and model targets; they are not a claim that the model correctly recognizes3,370words.
''')
 # A single offline HTML report makes the full list searchable without any server or upload.
 esc=html.escape
 word_rows='\n'.join('<tr class="gloss-row"><td>'+str(i)+'</td><td class="label">'+esc(t)+'</td><td>'+', '.join(esc(p['pool']) for p in label_info[t])+'</td><td>'+('Yes' if t in trained_labels else 'Prepared only')+'</td></tr>' for i,t in enumerate(alphabet(all_labels),1))
 run_rows='\n'.join('<tr>'+''.join('<td>'+esc(str(v))+'</td>' for v in [r['run'],r['actual_TRAIN_utterances'],r['actual_TRAIN_target_labels'],r['selection_evaluation_utterances'],r['completed_epochs'],f"{100*r['selected_evaluation_WER']:.3f}%" if r['selected_evaluation_WER'] is not None else 'Not comparable',r['status']])+'</tr>' for r in register)
 pool_rows='\n'.join('<tr>'+''.join('<td>'+esc(str(v))+'</td>' for v in [c['pool'],c['eligible_utterances'],c['eligible_gloss_labels'],c['ever_fitted_unique_utterances'],c['ever_fitted_distinct_labels'],c['prepared_not_fitted_utterances']])+'</tr>' for c in counts)
 doc='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AppHear gloss inventory and training report</title><style>
body{font:16px/1.55 system-ui,sans-serif;color:#17263b;background:#f4f7fa;margin:0}main{max-width:1280px;margin:auto;padding:28px}h1{font-size:30px}h2{margin-top:32px}section{background:white;padding:22px;margin:20px 0;border:1px solid #d5deea;border-radius:10px}.stats{display:flex;gap:16px;flex-wrap:wrap}.stats div{background:#e8f1f9;padding:16px;border-radius:8px}.stats strong{display:block;font-size:28px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #dce3ed;padding:8px;text-align:left;vertical-align:top}th{background:#edf2f8;position:sticky;top:0}.scroll{overflow:auto;max-height:65vh}.label{white-space:pre-wrap;overflow-wrap:anywhere}input{padding:12px;font:inherit;width:min(90%,650px);border:1px solid #9eafc2;border-radius:6px}a{color:#164e9a}.note{background:#fff4da;padding:14px;border-left:4px solid #b9801a}code{overflow-wrap:anywhere}@media print{body{background:white}.scroll{max-height:none;overflow:visible}input{display:none}th{position:static}section{break-inside:auto}}
</style><main><h1>AppHear continuous-ASL data and training report</h1><p>Frozen research inventory. Gloss labels include variants, pointing/classifier notation and fingerspelled names; they are not a list of reliably recognized English words.</p><div class="stats"><div><strong>3,370</strong>eligible distinct gloss spellings</div><div><strong>3,256</strong>spellings used in completed fits</div><div><strong>3,121</strong>eligible utterances</div><div><strong>3,025</strong>unique utterances actually fitted</div></div>
<p class="note">The two label pools remain separate. Their display union is not a merged training ontology. Ninety-six prepared utterances have not been fitted. Repeated epochs and runs do not create new independent data.</p>
<section><h2>Download the complete lists and evidence</h2><p><a href="ALL_TRAINED_GLOSS_LABELS.txt">All 3,256 trained gloss spellings</a> · <a href="ALL_ELIGIBLE_GLOSS_LABELS.txt">All 3,370 eligible spellings</a> · <a href="ASLLRP3_FULL_GLOSS_LIST.txt">ASLLRP3 list</a> · <a href="NCSLGR2_FULL_GLOSS_LIST.txt">NCSLGR2 list</a> · <a href="LATEST_PILOT_VOCABULARY.txt">Latest model mapping</a></p><p><a href="FULL_TRAINING_REPORT.md">Detailed data/training report</a> · <a href="TRAINING_RUNS.json">Run ledger</a> · <a href="GLOSS_LABELS_AND_COUNTS.json">Per-label frequencies/support</a> · <a href="UTTERANCE_DATA_INVENTORY.jsonl">Full utterance inventory</a></p></section>
<section><h2>Dataset totals</h2><div class="scroll"><table><thead><tr><th>Label pool</th><th>Eligible utterances</th><th>Eligible labels</th><th>Fitted utterances</th><th>Fitted labels</th><th>Prepared only</th></tr></thead><tbody>'''+pool_rows+'''</tbody></table></div><p>23,002 eligible token occurrences; 22,586 fitted token occurrences, counting each utterance once. Of the 3,801 per-pool label entries, 431 spellings occur in both pools, giving 3,370 distinct display spellings.</p></section>
<section><h2>Latest source-transfer pilot</h2><p>1,500 TRAIN; 80 compatible held-out sequences; 27 stronger-support sequences reported separately. Fresh model, known signers and new conservative recording families. No unseen-signer claim.</p><p><strong>14 epochs; selected epoch 12.</strong> TRAIN WER 87.08%; primary held-out WER 96.91%; stronger subgroup 96.62%. Epoch loop 88.50s; peak allocated VRAM 1.214GiB. Both learning screens failed. Further training is not justified by this run; no promotion occurred.</p><p>Frozen baselines, datasets, exclusions, protected TEST data and production assets remain unchanged. Earlier WER values below use different evaluation protocols and are not a leaderboard.</p></section>
<section><h2>Saved training-run register</h2><p>Includes smoke runs and identifies isolated Experiment A records outside continuous totals. Null counts mean the report does not infer an unsupported quantity. Full raw status evidence is available in the JSON ledger.</p><div class="scroll"><table><thead><tr><th>Run</th><th>Actual TRAIN</th><th>Target labels</th><th>Selection eval</th><th>Epochs</th><th>Saved WER</th><th>Status</th></tr></thead><tbody>'''+run_rows+'''</tbody></table></div></section>
<section><h2>Complete gloss inventory</h2><label for="search">Search literal gloss label or pool</label><p><input id="search" type="search" placeholder="For example: BOOK, fs-, IX-1p, ASLLRP3"><span id="count" aria-live="polite"> 3,370 / 3,370 labels</span></p><div class="scroll"><table><thead><tr><th>List row</th><th>Literal gloss</th><th>Eligible pools</th><th>Used as TRAIN target</th></tr></thead><tbody id="words">'''+word_rows+'''</tbody></table></div><p>List row numbers are not model token IDs. Labels retain original case and notation. Blank and UNK are excluded.</p></section>
<script>const rows=[...document.querySelectorAll('.gloss-row')];document.querySelector('#search').addEventListener('input',e=>{let q=e.target.value.toLocaleLowerCase(),n=0;for(const r of rows){let show=r.textContent.toLocaleLowerCase().includes(q);r.hidden=!show;if(show)n++;}document.querySelector('#count').textContent=' '+n.toLocaleString()+' / 3,370 labels';});</script></main></html>'''
 text('FULL_TRAINING_REPORT.html',doc)
 save('input_hashes.json',INPUTS)
 paths=sorted(p for p in RUN.iterdir() if p.is_file() and p.name!='artifact_hashes.sha256')
 text('artifact_hashes.sha256','\n'.join(f'{sha(p)}  {p.name}' for p in paths))
 print(json.dumps({'status':'REPORT_CREATED','eligible_labels':len(all_labels),'trained_labels':len(trained_labels),'eligible_utterances':len(eligible),'trained_unique_utterances':len(trained),'saved_status_records':len(register),'artifacts':len(paths)}))

if __name__=='__main__':main()
