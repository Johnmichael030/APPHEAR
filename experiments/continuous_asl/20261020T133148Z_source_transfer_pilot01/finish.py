"""Reports and hashes after the sole authorized fit is closed."""
from common import *

def md(name,text):
 with (RUN/name).open('x',encoding='utf-8') as f:f.write(text.rstrip()+'\n')
def main():
 s=read(RUN/'training_status.json');cfg=read(RUN/'config.json');pre=read(RUN/'preflight.json');result=read(RUN/'command_result.json')
 scores={k:read(RUN/('selected_'+k+'.json')) for k in ['train1500','monitor','primary80','strong27','full206']}
 metrics=list(csv.DictReader((RUN/'METRICS.csv').open(encoding='utf-8-sig')));evaluated=[r for r in metrics if r['primary80_WER']]
 primaryids=set(read(RUN/'evaluation_views.json')['primary_compatible80']);strongids=set(read(RUN/'evaluation_views.json')['support5_subgroup27'])
 examples=[]
 for role,key in [('TRAIN','train1500'),('HELD_OUT','full206')]:
  for p in scores[key]['predictions']:
   examples.append({'split':role,'sample_id':p['sample_id'],'primary80':p['sample_id'] in primaryids,'strong27_subgroup':p['sample_id'] in strongids,
    'signer_id':p['signer_id'],'source_group':p['source_group'],'capture_family':p['capture_family'],'reference':json.dumps(p['reference'],ensure_ascii=False),'hypothesis':json.dumps(p['hypothesis'],ensure_ascii=False),
    'CTC_loss':p['CTC_loss'],'substitutions':p['substitutions'],'deletions':p['deletions'],'insertions':p['insertions'],'reference_tokens':p['reference_tokens'],
    'blank_frames':p['blank_frames'],'output_frames':p['output_frames']})
 table('PREDICTION_EXAMPLES.csv',examples)
 groups=[]
 for view,score in scores.items():
  if view=='monitor':continue
  for axis in ['capture_family','signer_id']:
   by={}
   for p in score['predictions']:by.setdefault(p[axis],[]).append(p)
   for key,preds in sorted(by.items()):
    q=summarize(preds);groups.append({'evaluation':view,'group_axis':axis,'group_id':key,'utterances':q['utterances'],'WER':q['gloss_WER'],'CTC_loss':q['CTC_loss'],
     'blank_argmax_fraction':q['blank_argmax_fraction'],'predicted_tokens':q['predicted_tokens'],'unique_predicted_tokens':q['unique_predicted_tokens'],'empty_hypotheses':q['empty_hypotheses'],'multi_token_hypotheses':q['multi_token_hypotheses']})
 table('GROUP_RESULTS.csv',groups)
 # Prior baseline equivalence fails on membership, vocabulary and normalization-fit populations.
 table('comparison_equivalence.csv',[
  {'baseline':'Frozen85.96%NCSLGR9DEV','evaluation_equivalent':False,'labels_equivalent':False,'vocabulary_equivalent':False,'normalization_equivalent':False,'decoder':'Same greedy CTC family; insufficient for equivalence','status':'NON_EQUIVALENT; no WER ranking'},
  {'baseline':'ASLLRP1706TRAIN413DEV pilot','evaluation_equivalent':False,'labels_equivalent':True,'vocabulary_equivalent':False,'normalization_equivalent':False,'decoder':'Reused unchanged core greedy decoder','status':'NON_EQUIVALENT; previous weights also leak internal holdout'},
  {'baseline':'ASLLRP497TRAIN92compatibleDEV pilot','evaluation_equivalent':False,'labels_equivalent':True,'vocabulary_equivalent':False,'normalization_equivalent':False,'decoder':'Reused unchanged core greedy decoder','status':'NON_EQUIVALENT; restricted target task and different normalization'}])
 inputs={str(DESIGN/'folds/A_guarded_pooled_family_holdout.json'):cfg['fold_manifest_sha256'],str(PARENT/'data/processed/train_dev/manifest.json'):read(RUN/'fold_manifest.json')['universe_manifest_sha256'],str(ROOT/'tools/continuous_asl/ctc_experiment.py'):cfg['core_model_decoder_sha256']}
 save('input_hashes.json',inputs)
 control_files=['fold_manifest.json','resolved_manifest.json','evaluation_views.json','train_monitor_manifest.json','vocabulary.json','normalization.npz','config.json',s['initialization_checkpoint']['path']]
 md('manifest_hashes.sha256','\n'.join(f'{sha(RUN/p)}  {Path(p).as_posix()}' for p in control_files))
 epoch_table='\n'.join(f"|{r['epoch']}|{float(r['monitor_WER'])*100:.3f}%|{float(r['primary80_WER'])*100:.3f}%|{float(r['strong27_WER'])*100:.3f}%|{float(r['primary80_CTC']):.4f}|{float(r['primary80_blank']):.4f}|{r['primary80_diversity']}|" for r in evaluated)
 selected_table='\n'.join(f"|{k}|{v['utterances']}|{100*v['gloss_WER']:.4f}%|{v['CTC_loss'] if v['CTC_loss'] is not None else 'N/A: unsupported full206targets'}|{v['blank_argmax_fraction']:.6f}|{v['unique_predicted_tokens']}|{v['empty_hypotheses']}|" for k,v in scores.items())
 reference_examples='\n'.join(f"- `{p['sample_id']}`: reference `{json.dumps(p['reference'],ensure_ascii=False)}` → hypothesis `{json.dumps(p['hypothesis'],ensure_ascii=False)}`." for p in scores['primary80']['predictions'][:5])
 if s['further_training_justified']:
  decision='The engineering screens passed. This supports considering a separately authorized replication on another frozen recording-family holdout; no continuation or promotion is scheduled. Two family clusters do not establish population or unseen-signer generalization.'
 elif not s['TRAIN_learning_screen_pass']:
  decision='Further training is not justified by this pilot. The TRAIN-learning screen failed, so poor held-out WER is inconclusive as a source-transfer diagnosis. Do not attribute the failure to recording or signer effects. Review existing feature/label alignment and the optimization/representation limitation before another fit; no automatic continuation is scheduled.'
 else:
  decision='Further training is not justified by the held-out screen. Meaningful TRAIN learning was observed, but transfer to these known-signer recording families failed the declared screen. This rules out an unseen signer as a necessary condition for this failure, while leaving source/topic/label/feature effects confounded. No automatic continuation is scheduled.'
 md('PILOT_REPORT.md',f'''# Controlled known-signer source-transfer pilot

**PASS: frozen split verification, CUDA execution and checkpoint reload. PARTIAL: recognition. BLOCKED: promotion and causal/unseen-signer claims.** Exactly one fresh research fit ran. It stopped `{s['status']}` after **{s['completed_epochs']} complete epochs** and{s['optimizer_updates']}optimizer updates. No acquisition, prior-fit rerun, TEST use, Android/APK/GLB change or checkpoint promotion occurred.

## Frozen design and controls

The authorized fold was copied byte-for-byte; SHA256 `{cfg['fold_manifest_sha256']}`. Source: `{DESIGN/'folds/A_guarded_pooled_family_holdout.json'}`. InternalTRAIN1,500, full internal heldout206, primary compatible evaluation80, stronger-support subgroup27. The80/27IDs exactly match the frozen membership and recomputed TRAIN-only compatibility/support. No split or subgroup was reconstructed differently. Three TRAIN signers; two evaluation signers already inTRAIN. Thirty-four TRAIN masters and five held-out masters lie in disjoint conservative families (six/two); related undated Ben masters stay together. Dates remain conservative filename-prefix proxies, not independently verified capture timestamps.

The new resolved manifest preserves original globalTRAIN roles and adds an internal role; original manifests are unchanged. Feature hashes,378-float schema, finite values, increasing timestamps and CTC length constraints passed for all1,706used records. All pose/hand feature blocks and the unchanged causal48channel model/greedydecoder are retained. Only the ASLLRP3 literal realized-occurrence pool is used.

Fixed vocabulary:2,287literal TRAIN labels plus blank0/UNK1,2,289classes. All80and27references map exactly; no English translations or new label aliases. Vocabulary and mean/std fit only on1,500TRAIN. Fresh random initialization, seed20261020; no old recognition weights loaded. Initialization optimizer updates0. Initial checkpoint hash `{s['initialization_checkpoint']['sha256']}`; vocabulary hash `{s['vocabulary_sha256']}`; config hash `{s['config_sha256']}`; normalization hash `{s['normalization_sha256']}`. All controlled-file hashes are in manifest_hashes.sha256.

AdamWlr0.001,weightdecay0.0001, batch4, accumulation2, FP32,gradientclip5; actual CUDA tensors and finite gradients confirmed in cuda_smoke.json without a smoke optimizer update. Fixed128-TRAIN monitor chosen by deterministic source-stratified IDs, entirely outside heldout. Primary80(WER,CTCloss) selects the local checkpoint; subgroup27 and full206 never select it.

Bounds:20epochs or180seconds;175second conservative stop request reserves flush time. Configured evaluations at1,2,4and every two epochs thereafter. Persistent blank/frequent-token collapse and four evaluations without≥0.5percentage-point credible primary improvement stop early. These controls were written before updates. Optimizer/RNG/config are saved with checkpoints; a mid-epoch budget stop, if any, preserves its order and pending gradients separately. No checkpoint is automatically promoted.

## Measured trajectory

|Epoch|Fixed TRAIN monitor WER|Primary80WER|Stronger27WER|PrimaryCTC|Primaryblank|Primarylabel diversity|
|---|---|---|---|---|---|---|
{epoch_table}

Selected checkpoint epoch{s['selected_checkpoint']['epoch']}:

|Evaluation|Utterances|WER|CTCloss|Blankargmaxfraction|Labeldiversity|Emptyoutputs|
|---|---|---|---|---|---|---|
{selected_table}

The27stronger-support sequences remain a subgroup of the80primary sequences, not a replacement. Their predictions are derived from the same full held-out decoding. The full206literal WER is diagnostic coverage context; CTC loss is not aggregated for it because126sequences have unsupported labels. Those labels were not silently mapped into a surrogate score. Loss for all compatible80and27is valid and reported. PREDICTION_EXAMPLES.csv includes all selected1,500TRAIN/206heldout pairs and subgroup flags. Every evaluation checkpoint has saved per-example predictions and blank counts. GROUP_RESULTS.csv reports both family and signer breakdowns; the80view remains74participant3and6participant2, so per-participant2 estimates are descriptive.

## Runtime, GPU and integrity

Measured epoch loop (fit, regular evaluation and checkpoint I/O): **{s['epoch_loop_seconds']:.3f}s**. Including selected-checkpoint reload and fullTRAIN validation: **{s['with_reload_full_TRAIN_validation_seconds']:.3f}s**. Whole command including preflight/startup/initial evaluation: **{result['wall_seconds']:.3f}s**, exit{result['exit_code']}. Per-epoch fit and total durations are in METRICS.csv. These are wall-clock measurements, not estimates.

GPU: {s['GPU']}, total{pre['GPU_total_bytes']:,}bytes. Peak framework allocated **{s['peak_allocated_VRAM_bytes']:,}bytes ({s['peak_allocated_VRAM_bytes']/1073741824:.3f}GiB)**; reserved{s['peak_reserved_VRAM_bytes']:,}bytes. Allocated-memory cap2.5GiB and free-memory256MiB guards checked every microbatch, disk headroom10GiB. Fixed seeds and deterministic algorithms were used in warn-only mode; CUDACTC nondeterminism warning is retained, so bitwise cross-run reproducibility is not claimed. Reloaded selected hypotheses matched saved primary80predictions exactly.

Best local checkpoint: `{RUN/s['selected_checkpoint']['path']}`

SHA256 `{s['selected_checkpoint']['sha256']}`.

## Interpretation and next decision

TRAIN-learning screen passed: **{s['TRAIN_learning_screen_pass']}**. Primary transfer screen passed: **{s['primary_transfer_screen_pass']}**. The predeclared screen requires selected fullTRAIN and monitorWER≤50%,diversity≥10,multi-token output≥25%; primary80WER≤90%,diversity≥10,multi-token hypotheses≥20/80. These are research engineering gates, not production acceptance criteria.

{decision}

The old85.96%nine-NCSLGR baseline and prior413/92ASLLRPDEV experiments are **non-equivalent** in evaluation IDs,vocabulary,normalization-fit population and training membership. Decoder reuse alone does not make them comparable. No cross-run WER improvement claim is made; existing full-task results remain unchanged. See comparison_equivalence.csv. Within this frozen experiment, untrained predictions and the100%all-empty baseline provide transparent context.

This tests new recording families of known signers. It does not isolate a causal signer effect, source-dataset effect or unseen-signer generalization. Source/topic/session/label-frequency effects remain bundled, with only two held-out family clusters and uncertain parent lineage beyond the conservative guard.

Literal primary examples:

{reference_examples}
''')
 py=ROOT/'experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe'
 md('commands_and_results.txt',f'''Initial existing integrity verification (executed before preparation):
& '{py}' -B '{DESIGN/'verify.py'}'
PASS46source-design artifacts;65label-support artifacts;46prior diagnosis artifacts;1061continuation artifacts;20745protectedmetadata;3121eligible preserved.

Exactly one authorized training command:
& '{py}' -B '{RUN/'train.py'}'
Output training_console.txt; exit{result['exit_code']}; wrapper{result['wall_seconds']:.6f}s; {s['completed_epochs']}completeepochs; {s['status']}.
Config:config.json; exact sourcecopy:fold_manifest.json; resolvedmanifest:resolved_manifest.json; vocab:vocabulary.json; init:initialization.json/checkpoints/initialization.pt.
No pretrained recognition checkpoint was read for fitting. Only this run's selected checkpoint was reloaded for validation. No extra seeds/fits/acquisition/TESTinference.
FoldSHA:{s['fold_manifest_sha256']}
VocabSHA:{s['vocabulary_sha256']}
ConfigSHA:{s['config_sha256']}
InitializationSHA:{s['initialization_checkpoint']['sha256']}
SelectedcheckpointSHA:{s['selected_checkpoint']['sha256']}

Finalize:
& '{py}' -B '{RUN/'finish.py'}'

POST-RUN READ-ONLY CHECK (executed after closure):
& '{py}' -B '{RUN/'verify.py'}'
It verifies new control/checkpoint/artifact hashes and runs the existing design verification chain; no training, model inference or TESTscoring.

Next action: {decision}
Do not rerun train.py: it rejects existingconfig. Any further fit needs a new directory and authorization. All frozen assets, manifests, checkpoints, exclusions and test results remain protected. No APK or production integration changes.
''')
 md('SESSION_HANDOFF.md','One authorized source-transfer pilot complete. Read PILOT_REPORT.md, training_status.json and commands_and_results.txt. No queued training or acquisition. New80/27evaluations come from frozen1500TRAIN source-family design. No unseen-signer or causal claim, no model promotion. Root continuity/progress and all old protected files remain unchanged. Scope/integrity PASS; recognition and transfer readiness depend on saved screens, not training completion.')
 paths=sorted(p for p in RUN.rglob('*') if p.is_file() and p.name!='artifact_hashes.sha256' and '__pycache__' not in p.parts)
 md('artifact_hashes.sha256','\n'.join(f'{sha(p)}  {p.relative_to(RUN).as_posix()}' for p in paths))
 print(json.dumps({'status':'CLOSED','artifacts':len(paths),'training_status':s}))

if __name__=='__main__':main()
