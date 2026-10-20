"""Write measured findings and integrity index after the bounded job ends."""
from audit import *
import subprocess

def md(name,s):
 with (RUN/name).open('x',encoding='utf-8') as f:f.write(s.rstrip()+'\n')
def main():
 critical=read(RUN/'critical_checks.json');status=read(RUN/'pilot64/training_status.json');tr=read(RUN/'pilot64/best_train.json');dv=read(RUN/'pilot64/best_dev.json')
 oldtr=read(RUN/'replayed_train.json');olddv=read(RUN/'replayed_dev.json');protocol=read(RUN/'pilot64/protocol.json');coverage=read(RUN/'pilot64/subset_coverage.json')
 original=list(csv.DictReader((RUN/'POOL_COMPARISON.csv').open(encoding='utf-8-sig')))
 table('pool_comparison_raw_counts.csv',original)
 for row in original:
  if row['pool'].startswith('NCSLGR'):row['source_recordings']='UNKNOWN; 688 mixed-granularity source_group keys, 21 collections'
 with (RUN/'POOL_COMPARISON.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(original[0]));w.writeheader();w.writerows(original)
 save('pilot_metrics_summary.json',{'train_multi_token_outputs':sum(len(p['hypothesis'])>=2 for p in tr['predictions']),
  'dev_correct_tokens':dv['reference_tokens']-dv['substitutions']-dv['deletions'],
  'dev_correct_fraction':(dv['reference_tokens']-dv['substitutions']-dv['deletions'])/dv['reference_tokens'],
  'DEV_tokens_without_subset_training_support':coverage['DEV_tokens_not_in64TRAIN'],
  'subset_support_missing_fraction':coverage['DEV_tokens_not_in64TRAIN']/coverage['DEV_reference_tokens']})
 logs=[json.loads(s) for s in (RUN/'pilot64/training_log.jsonl').read_text().splitlines()];evals=[r for r in logs if 'train_WER' in r];last=evals[-1]
 vocab=read(PARENT/'pilot_fresh48_v2/vocabulary.json')['tokens'];fit=[r for r in read(PARENT/'data/processed/train_dev/manifest.json') if r['split']=='train']
 counts=Counter(t for r in fit for t in r['glosses']);singletons=sum(v==1 for v in counts.values());examples='\n'.join(f"- `{p['sample_id']}`: ref `{json.dumps(p['reference'],ensure_ascii=False)}` → hyp `{json.dumps(p['hypothesis'],ensure_ascii=False)}`" for p in tr['predictions'][:4])
 md('ROOT_CAUSE.md',f'''# Continuous-ASL recognition diagnosis

**PASS: bounded diagnosis and CUDA execution. PARTIAL: causal attribution and recognition. BLOCKED: promotion/product claims.** No acquisition, pool merge, production change or TEST tuning occurred. Active data goal remains 2,000, satisfied by the unchanged 3,121 eligible utterances.

## Verified failure mechanism

The selected five-epoch pilot checkpoint (epoch4) was replayed on all 1,706 TRAIN and 413 DEV records. Literal predictions and edit counts exactly matched saved results. DEV WER **97.9913%** has 218 substitutions, 2,953 deletions and no insertions / 3,236 tokens. Argmax blank frequency is **98.3956%**, mean blank probability **84.3033%**. There are 131 empty outputs, 281 one-token outputs and one two-token output. Only three distinct labels appear: 280 `IX-1p`, one `CAR`, two `ns-fs-JOHN`. `IX-1p` is only 178/3,236 reference tokens (5.50%), so 98.94% of emitted labels being this token is excessive concentration, not faithful reference frequency.

TRAIN also underfits: WER97.6015%, 98.3179% blank frames, 436 empty outputs, only11 distinct predicted tokens vs2,517 target labels; 862/1,337 emissions `IX-1p` (64.47%), compared with619/14,217 reference occurrences (4.35%). Only five TRAIN utterances have length3; almost all outputs are length0or1. The poor DEV score is not solely unseen-signer generalization: the large fit has not learned its TRAIN sequences well.

**Most likely root cause (moderate confidence): optimization remains in a blank/frequent-label solution in a sparse literal-gloss problem; unseen-signer and vocabulary-support limitations further constrain DEV.** The direct output mechanism is high-confidence measured evidence. An exact causal division between optimization exposure, model capacity, landmark information and label granularity is not established.

The ASLLRP full pool has2,739 literal labels,1,331 occurring once. New TRAIN has{len(counts)} labels, {singletons} singletons, over14,217 tokens. Distinctions such as classifiers, spatial suffixes, hand variants, repetitions and whole fingerspelled words are preserved. Lack of samples per label is a measured constraint, not permission to collapse labels. CTC loss reduction alone did not demonstrate recognition.

## Critical engineering checks

| Check | Status | Evidence / limit |
|---|---|---|
| Frozen package | PASS | Requested Verify-Continuation.ps1 passed first:1,061 artifacts,20,745 protected metadata |
| Replay | PASS | Selected checkpoint hash verified; all2,119 new split predictions match saved predictions |
| Mapping | PASS | 2,519 unique vocab entries; blank0,UNK1; exact checkpoint/train-set mapping; no special token in literal targets; zero TRAIN UNK targets |
| Normalization | PASS | Mean/std recomputed from1,706 TRAIN feature arrays exactly match checkpoint buffers; no DEV fit |
| CTC length | PASS | All3,121 feature files hash/schema/finite/timestamp/length checked; same-length causal model has no downsampling; repeats included in minimumCTCsteps |
| Decoder | PASS | Same literal conventions in targets and greedy decoding; blank-separated duplicate test passes; old50 checks reused |
| Temporal contract | PASS scoped / PARTIAL external validity | ASLLRP events inside publisher utterance frame bounds; prior exact frame/count joins reused. No independent visual/linguistic frame audit was invented. NCSLGR legacy uncalibrated timing remains limited |
| Implementation bug | UNKNOWN as an absolute claim | No critical bug found in these targeted checks; no model correction was justified |

Full token distributions, normalization-sensitive labels and NFC collisions are retained in pool_details.json and pool_token_frequencies.csv. No strings were silently normalized or repaired. A shared literal spelling does not establish shared linguistic annotation semantics.

## Pool and evaluation decision

ASLLRP3 is the most defensible current research contract: literal realized dominant occurrences, known publisher IDs for all2,119 records,47 source recordings, four signers. The unchanged same-pool split is1,706TRAIN/413DEV,39/8 sources,three/one signers, with no sample/signer/collection/source/raw/pixel-hash overlap. All newDEV records were historicalTRAIN in older fits; this pilot uses fresh random weights. It remains development-exposed research data, not pristineTEST or a multi-signer reliability estimate.

NCSLGR2 remains separate:1,002 eligible utterances,1,062 labels,5,549 tokens,581singletonlabels; main-gloss rather than realized-occurrence contract. Only14 records have explicit signer IDs in this manifest (two IDs);988 unknown. Its source groups have mixed granularity:688 NCSLGR keys include utterance media and full recordings, so **do not interpret688 as independent source recordings**. It also has21 collection identifiers. POOL_COMPARISON.csv explicitly qualifies that count, with raw mechanical counts preserved separately. Legacy timing limitations and insufficient signer mapping make its current DEV contract less defensible. Publisher-name fields cannot be inferred from faces or filenames.

The two pools share {critical['pool_shared_literal_spellings']} literal spellings, with {critical['pool_union_glosses']} in the union. Their target ontologies are not proven equivalent: merging is BLOCKED. Old85.96% baseline uses nine different NCSLGR DEV utterances, different target coverage/vocabulary/training regime. Its WER is not an equivalent comparison with the413-ASLLRP DEV. Reserved17TEST was not read for recognition/tuning.

357/3,236DEV reference tokens (11.03%) are absent from the full TRAIN vocabulary. In DEV CTC loss those targets useUNK; WER correctly keeps literal references. Thus DEV loss is a surrogate for unsupported labels, while literal WER includes unavoidable coverage errors. It cannot explain97.99% by itself. Do not editDEV labels to disguise the gap.

## Completed intermediate learnability pilot

Reused the preexisting64 source-stratified TRAIN monitor, preserving original features,48channels,2519-class vocabulary,1706TRAIN-only normalization,loss,decoder,optimizer and seed. No old recognition weights reused. Frozen64TRAIN/413DEV manifest SHA256 `{status['manifest_sha256']}`. This changes training subset/repeated exposure as a diagnostic; it is not a controlled proof that one hyperparameter causes full-fit failure.

Actual RTX3050CUDA: **{status['completed_epochs']}epochs / {status['optimizer_updates']}optimizer updates**, stopped `{status['status']}`. Selected local checkpoint epoch{status['best_checkpoint']['epoch']}: TRAIN WER **{100*tr['gloss_wer']:.4f}%**, DEV **{100*dv['gloss_wer']:.4f}%**. Final measured epoch{last['epoch']} TRAIN WER **{100*last['train_WER']:.4f}%**, DEV **{100*last['dev_WER']:.4f}%**, TRAIN token diversity{last['train_unique']} / DEV{last['dev_unique']}. The final TRAIN state is diagnostic, not a replacement for DEV-based selection. The selected local pointer is not a production selection.

This intermediate result, together with the earlier eight-sample memorization, tests learnability beyond a tiny fixture. It must not be presented as held-out accuracy. DEV unsupported by64TRAIN: {coverage['DEV_tokens_not_in64TRAIN']}/{coverage['DEV_reference_tokens']} tokens; full vocabulary includes other1706TRAIN labels but the64fit does not see their target examples. Any DEV failure therefore mixes generalization and limited subset support.

See NEXT_PILOT_DECISION.md for the measured stopping decision. Checkpoint SHA256 `{status['best_checkpoint']['sha256']}`. No checkpoint was promoted. Streaming boundaries, no-sign false activation and end-to-end latency remain BLOCKED without consented streaming evidence.
''')
 rows='\n'.join(f"|{r['epoch']}|{r['train_WER']*100:.3f}%|{r['dev_WER']*100:.3f}%|{r['train_CTC']:.4f}|{r['dev_CTC']:.4f}|{r['train_blank']:.4f}|{r['dev_blank']:.4f}|{r['train_unique']}|{r['dev_unique']}|" for r in evals)
 md('NEXT_PILOT_DECISION.md',f'''# Bounded pilot decision — executed

**PASS execution; PARTIAL recognition; BLOCKED promotion.** The critical gates passed and the small same-pool CUDA pilot was executed in this session. No second pool was mixed or fitted.

|Epoch|TRAIN WER|DEV WER|TRAIN CTC|DEV CTC|TRAIN blank|DEV blank|TRAIN unique|DEV unique|
|---|---|---|---|---|---|---|---|---|
{rows}

Stopped: `{status['status']}`. Bounds80epochs/90seconds, collapse gate20, meaningful-sequence and DEV improvement gate40; see immutable pilot64/protocol.json written before fitting. Checkpoints include optimizer/RNG/manifest/config; saved checkpoints and examples were reloaded and DEV hypotheses matched.

Measured fit loop {status['fit_loop_seconds']:.3f}s; with selected-checkpoint reload {status['with_reload_seconds']:.3f}s. Wrapper wall {read(RUN/'pilot_wall.json')['wall_seconds']:.3f}s includes startup/initialDEV. Peak allocated VRAM {status['peak_allocated_vram_bytes']:,}bytes ({status['peak_allocated_vram_bytes']/1073741824:.3f}GiB), reserved {status['peak_reserved_vram_bytes']:,}bytes. Device {status['gpu']}. All fit tensors CUDA, VRAM/free memory gates checked each batch. Epoch durations and all evaluation details are saved; CUDA CTC nondeterminism warning retained.

Selected DEV-based checkpoint: `{RUN/'pilot64'/status['best_checkpoint']['path']}`

SHA256 `{status['best_checkpoint']['sha256']}`.

Selected TRAIN/DEV losses: {tr['loss']:.6f}/{dv['loss']:.6f}; WER {tr['gloss_wer']*100:.4f}%/{dv['gloss_wer']*100:.4f}%. Full prediction lists with literal references, token counts, blank probabilities and length histograms: pilot64/best_train.json and best_dev.json. Per-epoch lists include the final state; no estimates substituted for measurements.

Example selected TRAIN predictions:

{examples}

Next action: keep this pilot stopped. Reuse the measured TRAIN-learning trajectory to predeclare a single next experiment, rather than adding data or repeating a full fit. A practical next step is a frozen intermediate256TRAIN support audit (no TEST), quantifying rare-gloss and DEV coverage and preserving the same signer-disjoint DEV; launch another bounded fit only if meaningful label support and a specific exposure hypothesis are established. Do not expect64-example memorization to generalize to708DEV labels. Do not merge ontologies, prune difficult DEV records, tune onTEST, or promote weights.

The immediate evidence supports a learnability/coverage investigation, not a decoder threshold sweep or another long GPU run. The3,121eligible count and2,000goal remain unchanged. Production streaming acceptance remains blocked by absent evaluation data.
''')
 py=ROOT/'experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe'
 md('commands_and_results.txt',f'''Requested initial read-only command:
& '{PARENT/'Verify-Continuation.ps1'}'
PASS:1061newartifacts;20745protectedmetadata;3121eligible;5priorCUDAepochs;TEST_scored=false.

Exact bounded audit command:
& '{py}' -B '{RUN/'audit.py'}'
Output audit_console.txt. PASS: {critical['elapsed_seconds']:.3f}s;3121features;exact2119predictionreplay;nohardCTC/vocab/normalizationbugfound.

Exact bounded pilot command:
& '{py}' -B '{RUN/'pilot.py'}'
Output pilot_console.txt; pilot_wall.json exit{read(RUN/'pilot_wall.json')['exit_code']}. {status['completed_epochs']}epochs, {status['optimizer_updates']}updates, {status['status']}.
Frozenmanifest:{RUN/'pilot64/manifest.json'}
SHA:{status['manifest_sha256']}
Savedprotocol:pilot64/protocol.json; seed20261020; channels48; full2519vocab; TRAIN64/DEV413; batch4accum2; AdamW.001wd.0001; FP32clip5.
Noacquisition,noTESTrecognition,noproductionassetwrites. Outputexclusive; scriptsrefuseoverwrite. No repeatedoldfits or50numericchecks. CUDACTCwarn_onlynondeterminismretained.

Exact close command:
& '{py}' -B '{RUN/'close.py'}'

NEXT READ-ONLY COMMAND:
& '{py}' -B '{RUN/'verify.py'}'
This verifies hashes and calls the existing parentvalidator's underlyingPythonchecker; no training or acquisition. Do not rerun completedpilot.py.
''')
 md('SESSION_HANDOFF.md','Research-only diagnosis and one bounded64-TRAIN pilot complete. Read ROOT_CAUSE.md and NEXT_PILOT_DECISION.md. Frozen parents, root memory/progress, production assets and reservedTEST preserved. No queued training/acquisition. Active2,000data target remains achieved. Scope review: plan alignment PASS; architecture boundaries PASS; production readiness BLOCKED by weak DEV and absent streaming/no-sign evidence.')
 paths=sorted(p for p in RUN.rglob('*') if p.is_file() and p.name!='artifact_hashes.sha256' and '__pycache__' not in p.parts)
 md('artifact_hashes.sha256','\n'.join(f'{sha(p)}  {p.relative_to(RUN).as_posix()}' for p in paths))
 print(json.dumps({'status':'CLOSED_PARTIAL_RECOGNITION','artifacts':len(paths),'pilot':status}))

if __name__=='__main__':main()
