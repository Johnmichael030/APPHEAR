"""Close reports using saved predictions, with no further model execution."""
from support_audit import *

def md(name,text):
 with (RUN/name).open('x',encoding='utf-8') as f:f.write(text.rstrip()+'\n')
def aggregate(predictions):
 c=Counter()
 for p in predictions:
  for k in ['substitutions','deletions','insertions','reference_tokens']:c[k]+=p[k]
 return {**dict(c),'WER':(c['substitutions']+c['deletions']+c['insertions'])/c['reference_tokens'] if c['reference_tokens'] else None,'utterances':len(predictions)}

def main():
 d=read(RUN/'decision.json');details=read(RUN/'candidate_details.json');pools=read(RUN/'pool_details.json');status=read(RUN/'pilot497/training_status.json')
 tr=read(RUN/'pilot497/best_train.json');cd=read(RUN/'pilot497/best_compatible_dev.json');fd=read(RUN/'pilot497/best_original_dev.json')
 original=read(PARENT/'data/processed/train_dev/manifest.json');fulltrain=[r for r in original if r['split']=='train'];sub=read(RUN/'frozen_manifest.json');fit=[r for r in sub if r['split']=='train'];dev=[r for r in sub if r['split']=='dev'];compatible=read(RUN/'compatible_dev.json');compids={r['sample_id'] for r in compatible}
 previous=read(DIAG/'pilot64/best_dev.json');oldfit=[r for r in read(DIAG/'pilot64/manifest.json') if r['split']=='train'];prevtrain=read(DIAG/'pilot64/best_train.json');large=read(PARENT/'pilot_fresh48_v2/best_checkpoint_dev.json')
 err=[]
 for support_name,universe in [('64_effective_TRAIN',oldfit),('1706_parent_TRAIN',fulltrain)]:
  labels=set(support(universe));seen=[p for p in previous['predictions'] if set(p['reference'])<=labels];unseen=[p for p in previous['predictions'] if not set(p['reference'])<=labels]
  for name,preds in [('all_labels_seen',seen),('has_unseen_label',unseen)]:err.append({'support_universe':support_name,'group':name,**aggregate(preds)})
 table('GENERALIZATION_GROUPS.csv',err)
 new_support=support(fit);newstrata=defaultdict(Counter)
 for p in cd['predictions']:
  for t,op in align(p['reference'],p['hypothesis']):
   if op!='I':newstrata[bucket(new_support[t])][op]+=1
 table('PILOT_COMPATIBLE_DEV_ERRORS.csv',[{'retained_TRAIN_support':b,'reference_tokens':sum(c.values()),'correct':c['M'],'substitutions':c['S'],'deletions':c['D'],'reference_error_fraction':(c['S']+c['D'])/sum(c.values())} for b,c in newstrata.items()])
 quality={}
 for name,universe in [('new497TRAIN',fit),('compatible92DEV',compatible),('original413DEV',dev)]:
  quality[name]={key:distribution([r[key] for r in universe if isinstance(r.get(key),(int,float))]) for key in ['hand_observed_fraction','body_available_fraction'] if any(isinstance(r.get(key),(int,float)) for r in universe)}
 save('feature_availability_summary.json',quality)
 contexts=[]
 for name,score in [('prior64_full2519vocab',previous),('prior1706_full2519vocab',large),('new497_restricted425vocab',fd)]:
  for scope,preds in [('original413DEV',score['predictions']),('compatible92DEV',[p for p in score['predictions'] if p['sample_id'] in compids])]:
   contexts.append({'model':name,'evaluation_scope':scope,**aggregate(preds),'comparison_limit':'Same saved literal references; restricted task not full-task baseline replacement; fresh497fit changes membership/vocab/norm together.'})
 table('EVALUATION_CONTEXT.csv',contexts)
 seenlabels=set(support(oldfit));trainpred={t for p in prevtrain['predictions'] for t in p['hypothesis']};devpred={t for p in previous['predictions'] for t in p['hypothesis']}
 save('never_predicted_labels.json',{'prior64_TRAIN_target_labels':len(seenlabels),'never_predicted_in_TRAIN':sorted(seenlabels-trainpred),'never_predicted_in_TRAIN_or_DEV':sorted(seenlabels-(trainpred|devpred)),
  'scope':'Saved evaluated64TRAIN/413DEV only; NCSLGR per-label DEV absence appears TOKEN_SUPPORT.csv; no assertion about unsaved TRAIN predictions.'})
 counts=list(csv.DictReader((RUN/'POOL_COMPARISON.csv').open(encoding='utf-8-sig')))
 pool_table='\n'.join(f"|{r['pool']}|{r['unique_TRAIN_utterances']}|{r['TRAIN_gloss_tokens']}|{r['unique_TRAIN_labels']}|{r['labels_support_1']} ({float(r['labels_support_1_pct']):.2f}%)|{r['labels_support_2-4']} ({float(r['labels_support_2-4_pct']):.2f}%)|{r['labels_support_5-9']} ({float(r['labels_support_5-9_pct']):.2f}%)|{r['labels_support_10-19']} ({float(r['labels_support_10-19_pct']):.2f}%)|{r['labels_support_20+']} ({float(r['labels_support_20+_pct']):.2f}%)|" for r in counts)
 candidate_table='\n'.join(f"|{k}|{v['retained_TRAIN_utterances']}|{v['actual_TRAIN_vocabulary_labels']}|{v['excluded_TRAIN_utterances']}|{v['original_DEV_tokens_covered_pct']:.2f}%|{v['compatible_DEV_utterances']}|{v['recursive_final_TRAIN']}|" for k,v in details.items())
 group_table='\n'.join(f"|{r['support_universe']}|{r['group']}|{r['utterances']}|{100*r['WER']:.3f}%|" for r in err if r['WER'] is not None)
 md('LABEL_SUPPORT_REPORT.md',f'''# Frozen label-support audit

**PASS bounded audit and scoped candidate validation. PARTIAL recognition. BLOCKED production promotion.** No new data acquisition or TEST tuning. The3,121eligible catalog, separate target pools, all prior manifests/exclusions/checkpoints/results and production assets remain unchanged.

The existing diagnostic verifier was located and run first; it also ran the preceding continuation verifier. Both passed, including46diagnostic artifacts,1,061parent artifacts and20,745protected metadata records. Existing feature/CTC/normalization/decoder checks were reused; no completed fit was repeated. New candidate feature files were hash/schema/timestamp/CTC checked before the one new fit.

## Distinct-utterance support

A label contributes once to support per TRAIN utterance, even when repeated in that utterance. Token occurrences are counted separately. Exact per-label counts, DEV-only labels, predicted occurrence counts, reference matches/substitutions/deletions and effective model TRAIN support are in TOKEN_SUPPORT.csv. No label normalization, ontology merge or partial target deletion occurred.

|Pool|TRAIN scope count|Token occurrences|Labels|Support1|Support2–4|Support5–9|Support10–19|Support20+|
|---|---|---|---|---|---|---|---|---|
{pool_table}

ASLLRP3 scope is the frozen1,706TRAIN split, with413same-pool DEV records, three/one publisher signers and39/8sources. TRAIN covers88.97% of DEV reference tokens, but only194/413utterances (46.97%) have every label supported. There are222DEV-only labels. Latest64fit effective support is much narrower:1,399/3,236DEVtokens (43.23%) are unseen in its64TRAIN examples. A full2519-class output vocabulary does not provide visual training for labels absent from those64targets.

NCSLGR2 scope is1,002eligible TRAIN-pool utterances;906were actually used in its historical reference fit. New eligibility must not be confused with completed training. Token and distinct support for the historical906fit are separately identified in TOKEN_SUPPORT.csv. Its oldnineDEV references have98.25% token coverage and8/9whole-utterance coverage under available1,002TRAIN labels, but incomplete signer metadata (988TRAINrecords unknown; two explicit signer IDs across14records), mixed source-group granularity (688keys, not688independent recordings) and tinyDEV make it less defensible for a new generalization pilot.

Main-gloss NCSLGR2 and literal dominant realized-occurrence ASLLRP3 are distinct supervision contracts. Sharing378features or some label spellings does not establish target equivalence. ASLLRP frame/event joins are verified; NCSLGR legacy timing retains its uncalibrated limits. Sequence/feature-length percentiles, minimumCTCslack, source/signer intersections and token-frequency top30s are in pool_details.json. No CTC-feasibility violations were found in reused full evidence or the new fit's checks. Signer-independent product or boundary accuracy remains unproven.

## Why12.22%TRAIN and97.47%DEV?

The old64fit's TRAIN CTC fell to0.3575 while DEV CTC rose to11.8044; it memorized343labels, not a generalizing recognizer. The saved alignment demonstrates errors even on supported labels:

|Effective64TRAIN support|DEV reference tokens|Correct|Reference-token error rate|
|---|---|---|---|
|Unseen|1399|0|100%|
|1utterance|609|3|99.51%|
|2–4|648|37|94.29%|
|5–9|305|8|97.38%|
|10–19|178|23|87.08%|
|20+|97|13|86.60%|

These are deterministic reference-token alignment errors, excluding insertions; complete WER retains insertions. They are associations with support, not causal estimates. Frequent-label errors are still severe, so rarity is **not** a sufficient explanation. The earlier full1706model errs on96.15% of1,687DEV tokens with20+TRAIN-utterance support, reinforcing that conclusion.

Saved64-model WER by whole-sequence support:

|Support universe|DEV group|Utterances|WER|
|---|---|---|---|
{group_table}

Best-supported explanation (moderate confidence): small effective training data and sparse fine-grained labels permit TRAIN memorization while visual/source/signer generalization remains poor. The singleDEV signer is genuinely unseen and sources are disjoint; source and signer shifts are confounded, and one held-out signer cannot characterize population performance. Cross-pool conventions explain why old85.96%baseline is not equivalent, but do **not** explain the within-ASLLRP64fit gap, since TRAIN/DEV share a contract. Prior audits found no mapping, blank-ID, decoder or normalization defect; genuine TRAIN sequences argue against a hard CTC inability to learn. Weak feature information, model capacity and optimization remain hypotheses, not ruled out.

## Frozen whole-sequence candidates

Thresholds5and10 and coverage gates were written before the candidate fit. Threshold means support in the original TRAIN universe. Keep a complete utterance only when every literal label clears it. Recompute support afterward; never claim threshold holds in retained data. All excluded IDs and below-threshold labels are in EXCLUDED_UTTERANCES.csv; original manifests were not edited.

|Candidate|Retained TRAIN|Actual labels|Excluded TRAIN|OriginalDEVtoken coverage|CompatibleDEV|Recursive closed-core TRAIN|
|---|---|---|---|---|---|---|
{candidate_table}

Selected≥5ASLLRP:497TRAIN/423labels/2,665tokens, three signers,37sources;92compatibleDEV records across all8DEVsources;22.28% of originalDEV utterances. The≥10alternative is199TRAIN/175labels/31DEV and loses one TRAIN signer's representation sharply. The≥5choice follows predeclared coverage gates, not prediction scores.

Important: after whole-sequence filtering,≥5candidate has90singleton labels and279labels below5retained examples. Recursively demanding≥5or≥10inside retained ASLLRP removes **all** utterances. The proposed experiment is a reduced-support **diagnostic**, not proof of an adequately supported closed vocabulary. NCSLGR's≥5recursivecore retains258utterances, but its tiny/incomplete signer DEV contract prevents choosing it over ASLLRP.

The frozen manifest contains497newTRAIN plus the original413DEV; compatible_dev.json is a separate evaluation view. OriginalDEV remains intact, including unsupported labels. New425class vocabulary and normalization are fit only on497TRAIN; no older recognition weights are reused. Unsupported originalDEVtargets map toUNK forCTCloss only; original literal references remain in WER. Restricted92DEV chooses the local checkpoint by(WER,loss), and original413DEV is reported alongside. Neither restricted-task score replaces a full-task baseline.

The completed pilot's compatibleDEV has **zero unsupported reference labels**, yet WER remains{100*cd['gloss_wer']:.3f}%. Thus unseen labels cannot be the sole cause of failure. The final retained support is still uneven, so this is not proof that label sparsity is irrelevant. Rising DEV CTC with falling TRAIN CTC supports overfitting/source-or-signer transfer failure; those effects remain confounded. PILOT_COMPATIBLE_DEV_ERRORS.csv attributes new errors by retained distinct support. feature_availability_summary.json reports existing hand/body availability separately by split; these numerical checks do not constitute visual quality or linguistic alignment adjudication.

Manifest SHA256 `{d['manifest_sha256']}`. The one justified pilot was executed: see PILOT_RESULTS.md. It stopped `{status['status']}` after{status['completed_epochs']}epochs. Selected TRAIN WER{100*tr['gloss_wer']:.3f}%, compatible92DEV{100*cd['gloss_wer']:.3f}%, original413DEV{100*fd['gloss_wer']:.3f}%. No promotion.
''')
 evals=[json.loads(s) for s in (RUN/'pilot497/training_log.jsonl').read_text().splitlines() if 'train_WER' in s]
 eval_table='\n'.join(f"|{r['epoch']}|{r['train_WER']*100:.3f}%|{r['compatible_DEV_WER']*100:.3f}%|{r['original_DEV_WER']*100:.3f}%|{r['train_CTC']:.3f}|{r['compatible_DEV_CTC']:.3f}|{r['train_blank']:.4f}|{r['compatible_DEV_blank']:.4f}|{r['train_unique']}|{r['compatible_DEV_unique']}|" for r in evals)
 example_text='\n'.join(f"- `{p['sample_id']}`: `{json.dumps(p['reference'],ensure_ascii=False)}` → `{json.dumps(p['hypothesis'],ensure_ascii=False)}`" for p in cd['predictions'][:6])
 md('PILOT_RESULTS.md',f'''# One controlled restricted-task CUDA pilot

**PASS execution; PARTIAL recognition; BLOCKED promotion.** Fresh425class/48channel model, samecausal378schema, no dropped feature blocks or downsampling. TRAIN497, compatibleDEV92, originalDEV413; all original split boundaries and exclusions preserved. Choice≥5parent support was frozen without scores. This is not a clean one-factor ablation: membership, vocabulary and TRAIN-fitted normalization change together for the new task.

Protocol saved before fitting: pilot497/protocol.json. Seed20261020, FP32,AdamWlr.001wd.0001, batch4accumulation2,clip5;30epoch/90second bound, collapse gate10, meaningful-learning/DEV gate20, compatibleDEVpatience2 thereafter. Only one new fit executed. No prior weights loaded. All checkpoints have optimizer/RNG/config/manifest; selected predictions reproduced after checkpoint reload. CUDACTCnondeterminism warning retained.

|Epoch|TRAINWER|Compatible92DEVWER|Original413DEVWER|TRAINCTC|CompatibleDEVCTC|TRAINblank|CompatibleDEVblank|TRAINlabels|CompatibleDEVlabels|
|---|---|---|---|---|---|---|---|---|---|---|
{eval_table}

Stopped `{status['status']}` at{status['completed_epochs']}epochs /{status['optimizer_updates']}updates. Fit loop{status['fit_loop_seconds']:.3f}s; with reload{status['with_reload_seconds']:.3f}s; wrapper{read(RUN/'pilot_wall.json')['wall_seconds']:.3f}s. GPU{status['gpu']}, peakallocated{status['peak_allocated_vram_bytes']:,}bytes ({status['peak_allocated_vram_bytes']/1073741824:.3f}GiB),reserved{status['peak_reserved_vram_bytes']:,}bytes. Per-epoch actual fit/evaluation durations and memory gates logged. No estimate substituted for a measurement.

Selected epoch{status['best_checkpoint']['epoch']} by compatibleDEV only. TRAINWER{tr['gloss_wer']*100:.4f}%,CTC{tr['loss']:.6f},blank{tr['blank_argmax_fraction']:.6f},diversity{tr['unique_predicted_tokens']}; compatibleDEVWER{cd['gloss_wer']*100:.4f}%,CTC{cd['loss']:.6f},blank{cd['blank_argmax_fraction']:.6f},diversity{cd['unique_predicted_tokens']}; originalDEVWER{fd['gloss_wer']*100:.4f}%,CTC{fd['loss']:.6f},blank{fd['blank_argmax_fraction']:.6f},diversity{fd['unique_predicted_tokens']}.

Checkpoint `{RUN/'pilot497'/status['best_checkpoint']['path']}`

SHA256 `{status['best_checkpoint']['sha256']}`.

Full originalDEV and compatibleDEV results are separate. EVALUATION_CONTEXT.csv also rescored saved older predictions on the identical92IDs, without retraining or tuning. Restricted-label WER must not be compared directly with the full-vocabulary baseline or nine-utterance85.96%NCSLGRbaseline. The original full-task results remain frozen.

Literal compatibleDEV examples:

{example_text}

All hypotheses, references, lengths, token counts and blank fractions are in pilot497/best_*.json and epoch JSONs. NoTESTaccess or promotion.

Next action: keep this bounded run stopped. Review its support-stratified and signer/source error evidence before any longer fit. A useful next no-fit diagnostic is whether supported, correctly acquired TRAIN labels transfer within unseen source recordings versus the unseen signer, using a predeclared additional TRAIN-only source split. That can distinguish memorization/source dependence from a signer effect more directly. It requires no new data or TEST access. Do not infer that label filtering solved generalization, or tune thresholds against theseDEV scores.
''')
 py=ROOT/'experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe'
 md('commands_and_results.txt',f'''Initial existing verification (executed first):
& '{py}' -B '{DIAG/'verify.py'}'
PASS46newpriorfiles; parentPASS1061files,20745protectedmetadata,3121eligible; noTESTscoring.

Support audit:
& '{py}' -B '{RUN/'support_audit.py'}'
PASS; whole-sequence≥5 and≥10 candidates for each separatepool; distinct-utterance counts; saved thresholds/protocol beforefit. audit_console.txt contains measured output.

One new CUDA fit:
& '{py}' -B '{RUN/'pilot.py'}'
Exit{read(RUN/'pilot_wall.json')['exit_code']}; {status['completed_epochs']}epochs; {status['status']}. pilot_console.txt, pilot_wall.json, protocol and trainingstatus contain exact evidence.
FrozenmanifestSHA:{d['manifest_sha256']}
Noacquisition,previousfitrerun,TESTtuning,production/Android/GLBchange. Existing frozenpreservationvalidators reused.

Close reports:
& '{py}' -B '{RUN/'finish.py'}'

NEXT READ-ONLY VERIFICATION:
& '{py}' -B '{RUN/'verify.py'}'
Do not rerun completedpilot.py; it rejects existing output. No queued fit. Nextresearchaction is a predeclared TRAIN-only source-vs-signer transfer diagnostic, not another longfit or threshold sweep.
''')
 md('SESSION_HANDOFF.md','Bounded label-support audit and one497TRAIN restricted-task CUDA pilot completed. Read LABEL_SUPPORT_REPORT.md and PILOT_RESULTS.md. All3,121eligible items and two label pools remain unchanged. No pending GPU jobs/acquisition. Existing root memory/progress preserved; session state here. Review: scope and boundaries PASS; recognition PARTIAL; production BLOCKED.')
 # Close all files before the index. No active training/log writer may remain.
 paths=sorted(p for p in RUN.rglob('*') if p.is_file() and p.name!='artifact_hashes.sha256' and '__pycache__' not in p.parts)
 md('artifact_hashes.sha256','\n'.join(f'{sha(p)}  {p.relative_to(RUN).as_posix()}' for p in paths))
 print(json.dumps({'status':'CLOSED','artifacts':len(paths),'pilot':status}))

if __name__=='__main__':main()
