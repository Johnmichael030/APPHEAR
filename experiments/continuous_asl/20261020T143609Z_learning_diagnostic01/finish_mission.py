"""Summarize completed mission from saved artifacts; no model loading or training."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import csv,hashlib,json,statistics
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[2];OLD=RUN.parent/'20261020T133148Z_source_transfer_pilot01'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def out(name,text):
 with (RUN/name).open('x',encoding='utf-8') as f:f.write(text)
def table(name,rows):
 with (RUN/name).open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
a=read(RUN/'gate_A.json');b=read(RUN/'gate_B.json');d=read(RUN/'decision.json');env=read(RUN/'environment.json');integ=read(RUN/'input_integrity.json');cfg=read(RUN/'config.json')
cad=read(RUN/'cadence_metadata_assessment.json')
with (RUN/'subset_learning_log.csv').open(encoding='utf-8-sig',newline='') as f:subsetlog=list(csv.DictReader(f))
metrics=[]
if (RUN/'training_metrics.csv').exists():
 with (RUN/'training_metrics.csv').open(encoding='utf-8-sig',newline='') as f:metrics=list(csv.DictReader(f))
# Only the baseline primary predictions were needed/read; other baseline summaries are in hashed training_status.
# Report only already hashed baseline score artifacts. TRAIN/strong metrics are in prior training_status for fallback.
oldstatus=read(OLD/'training_status.json')
# Existing prior status is included in input scope; primary predictions were hashed by run.
olddev=read(OLD/'selected_primary80.json')
comparisons=[dict(run='frozen_epoch12',split='primary80',utterances=80,reference_tokens=485,WER=oldstatus['primary80_WER'],substitutions=olddev['substitutions'],deletions=olddev['deletions'],insertions=olddev['insertions'],checkpoint_sha256=oldstatus['selected_checkpoint']['sha256'])]
train=dev=strong=None
if d.get('full_candidate_trained'):
 train=read(RUN/'selected_train1500.json');dev=read(RUN/'selected_primary80.json');strong=read(RUN/'selected_strong27.json')
 comparisons.append(dict(run='corrected_accumulation_candidate',split='primary80',utterances=80,reference_tokens=485,WER=dev['WER'],substitutions=dev['substitutions'],deletions=dev['deletions'],insertions=dev['insertions'],checkpoint_sha256=d['selected_checkpoint']['sha256']))
table('comparison.csv',comparisons)
examples=[]
for stage,obj in [('subset_initial',read(RUN/'subset_initial_predictions.json')),('subset_final',read(RUN/f"subset_predictions/epoch_{b['completed_epochs']:03d}.json"))]+([('candidate_TRAIN',train),('candidate_primary80',dev),('candidate_strong27',strong)] if train else []):
 for p in obj['predictions']:
  examples.append(dict(stage=stage,sample_id=p['sample_id'],reference=json.dumps(p['reference'],ensure_ascii=False),hypothesis=json.dumps(p['hypothesis'],ensure_ascii=False),substitutions=p['substitutions'],deletions=p['deletions'],insertions=p['insertions'],reference_tokens=p['reference_tokens'],CTC_loss=p['CTC_loss'],blank_frames=p['blank_frames'],output_frames=p['output_frames']))
table('prediction_examples.csv',examples)
now=datetime.now(timezone(timedelta(hours=8))).isoformat()
py=ROOT/'experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe'
command=f"& '{py}' -B '{RUN/'run_mission.py'}' *> '{RUN/'run.log'}'"
verify=f"& '{py}' -B '{RUN/'verify_mission.py'}'"
gain=d.get('absolute_WER_gain_percentage_points',0)
loopfit=sum(float(r['fit_seconds']) for r in metrics)
extra=""
if train:
 selected_epoch=d['selected_checkpoint']['epoch']
 extra=f"""
## Full candidate results

The candidate started from scratch, independent of smoke and subset weights, with the original 48-channel CausalCTC, seed20261020, FP32 AdamW lr0.001/weightdecay0.0001, microbatch4, accumulation2, clip5. It used identical frozen TRAIN1500/primary80/subgroup27, 2,287 literal gloss labels plus blank0/UNK1 and unchanged TRAIN-only normalization/378 features. The modeling/training-variable correction was the accumulation tail; cadence/features/labels/architecture/optimizer were unchanged.

Stopped {d['reason']} after {d['completed_epochs']} complete epochs; selected epoch{selected_epoch} by the primary80 tuple (WER,CTC loss). FullTRAIN was evaluated every epoch; the same fixed128 monitor was derived from those predictions. Primary80 was evaluated at declared intervals. Strong27 never selected a checkpoint. Every complete epoch has a resumable model/optimizer/RNG checkpoint and logged hash.

|Scope|Utterances|Reference tokens|WER|CTC loss|S/D/I|Blank fraction|Token diversity|
|---|---:|---:|---:|---:|---|---:|---:|
|Selected TRAIN|1500|{train['reference_tokens']}|{100*train['WER']:.4f}%|{train['CTC_loss']:.6f}|{train['substitutions']}/{train['deletions']}/{train['insertions']}|{train['blank_fraction']:.6f}|{train['token_diversity']}|
|Primary held-out|80|{dev['reference_tokens']}|{100*dev['WER']:.4f}%|{dev['CTC_loss']:.6f}|{dev['substitutions']}/{dev['deletions']}/{dev['insertions']}|{dev['blank_fraction']:.6f}|{dev['token_diversity']}|
|Support5 subgroup|27|{strong['reference_tokens']}|{100*strong['WER']:.4f}%|{strong['CTC_loss']:.6f}|{strong['substitutions']}/{strong['deletions']}/{strong['insertions']}|{strong['blank_fraction']:.6f}|{strong['token_diversity']}|

Primary80 frozen baseline:96.9072%,470/485 edits (157S/312D/1I). Candidate:{100*dev['WER']:.4f}%,{dev['substitutions']+dev['deletions']+dev['insertions']}/485 edits. Absolute WER reduction:{gain:.4f} percentage points; relative reduction:{d['relative_WER_gain_percent']:.4f}%. Positive means improvement; negative means worse. One edit is100/485=0.206186 percentage points. Credible improvement gate passed:{d['credible_gain']}. FullTRAIN learning screen passed:{d['TRAIN_learning_screen_pass']}.

Data, label units, normalization, vocabulary, decoder/edit counter and primary membership are equivalent. Execution is not identical: fullTRAIN-per-epoch evaluation adds overhead and can consume the wall budget earlier. CUDA CTC backward remains nondeterministic in warn-only mode; even an equal epoch/seed comparison is not proof the tail correction alone caused a score difference. The earlier nine-NCSLGR85.96% baseline remains non-equivalent and is not used for the claim.

Selected experimental checkpoint:
{RUN/d['selected_checkpoint']['path']}
SHA256:{d['selected_checkpoint']['sha256']}

Candidate-stage elapsed including fit/evaluation/checkpoint I/O and final selected reload/scoring:{d['training_loop_seconds']:.3f}s. The field training_loop_seconds in decision.json includes finalization; it is not optimizer-only time. Complete-epoch fit time sums to{loopfit:.3f}s; last logged complete-epoch elapsed is{float(metrics[-1]['elapsed_seconds']):.3f}s. Peak allocated:{d['peak_allocated_bytes']:,}bytes ({d['peak_allocated_bytes']/2**30:.3f}GiB); reserved:{d['peak_reserved_bytes']:,}bytes. See runtime_estimate_after_epoch1.json for the estimate made after the first epoch; estimates are not measured full-run duration.

The180s bound is a soft stop request checked before optimization groups. A group/evaluation/checkpoint/final scoring may finish after it. If partial-epoch work occurred, it is not selected and the last completed-epoch checkpoint is retained; incomplete work is not claimed as a complete epoch. No automatic second fit or continuation ran.
"""
selected_examples=""
if dev:
 for p in dev['predictions'][:5]:selected_examples+=f"- {p['sample_id']}: reference {json.dumps(p['reference'],ensure_ascii=False)}; hypothesis {json.dumps(p['hypothesis'],ensure_ascii=False)}.\n"
report=f"""# Fast continuous-ASL learning diagnostic and controlled CUDA candidate

Generated {now} (Asia/Shanghai).

Overall PARTIAL: execution and fixed-subset learnability passed; continuous recognition is not accepted. Candidate promotion:NO. No reserved TEST evaluation, production/Android changes, data acquisition, source relabeling or cadence feature regeneration occurred.

## Answers

1. Fixed subset learned:{b['status']=='PASS'}. WER{100*b['initial']['WER']:.4f}% → {100*b['final']['WER']:.4f}%; CTC{b['initial']['CTC_loss']:.6f} → {b['final']['CTC_loss']:.6f}.
2. Full candidate trained:{d.get('full_candidate_trained',False)}. Complete epochs:{d.get('completed_epochs',0)}.
3. Compatible held-out WER improved descriptively:{gain>0}; credible gated improvement:{d.get('credible_gain',False)}. See exact denominator/counts below.
4. Subset elapsed{b['elapsed_seconds']:.3f}s; peak{b['peak_allocated_bytes']/2**20:.2f}MiB allocated. Candidate duration/memory below.
5. Higher cadence was assessed from metadata only; no new landmarks/features or recognition comparison.
6. Qualified semantic/visual alignment review, streaming/no-sign/latency validity and unseen-signer generalization remain unresolved.
7. Safe to promote:NO. This is exploratory known-signer recording-family research.

## Gate A and preservation

Reused the latest alignment report, prior pilot summary, verified cached-feature and annotation evidence rather than redoing preprocessing. Located the source checkpoint via saved training_status.json, verified its SHA256 e2a6e0046bca4a3f97f33f023dc7346dd9d6226e19e9bd855357a00c8f1579bf. Strict load, vocabulary and normalization buffers passed. A disposable source-weight model performed one real TRAIN CUDA forward/backward/optimizer-step smoke; model state was discarded, never used to warm-start the subset or candidate. Source checkpoint bytes stayed unchanged.

All1,580 used feature files (TRAIN1500+compatible80) passed hash, schema-before-read, float32/378width, finite values, timestamp ordering, transcript metadata and CTC feasibility T>=U+adjacent repeats. All target IDs>=2; blank0/UNK1 remain specials. The existing batch helper's unknown fallback cannot silently affect these verified targets. Frozen IDs and literal targets match the globalTRAIN universe; master/family/content/signer constraints remain unchanged. No two-pool merge occurred.

Accumulation now forms groups of up to two microbatches and divides each loss by the actual group count, then clips/steps once. With375microbatches this gives187pairs plus one correctly scaled singleton. This is experiment-local. It fixes a small confirmed defect, not a claim of the recognition root cause.

## Gate B — fixed subset learnability

Sixteen deterministic, family-balanced TRAIN utterances, stable SHA ordering with prefix subset-20261020|, seed20261020. Exact IDs:subset_manifest.json. Full2,289-class output, saved TRAIN-only normalization, labels, model and optimizer were retained. No DEV/TEST example was used.

Predeclared bounds:200epochs/60s, measurements at1and every10epochs. Pass:WER<=20%,loss<=50%initial,and>=10percentage-point WER reduction (low-initial-WER safeguard documented in config). Passed at epoch{b['completed_epochs']}, with{b['final']['substitutions']+b['final']['deletions']+b['final']['insertions']}/{b['final']['reference_tokens']}errors,blank fraction{b['final']['blank_fraction']:.6f},diversity{b['final']['token_diversity']},empty outputs{b['final']['empty_outputs']}. Initial random outputs caused insertion-heavy WER above100%; this is valid token WER, not accuracy. The transient all-blank state at epoch10 resolved without changing data or hyperparameters.

This proves the wiring can memorize these16existing examples. It does not establish full-task learning or transfer. Saved checkpoint:{RUN/b['checkpoint']['path']}; SHA256:{b['checkpoint']['sha256']}.
{extra}
## Gate C — metadata-only cadence assessment

Current retained indices0,3,6,... were compared arithmetically with all native indices at29.97fps on the frozen12-event review manifest. Current focal frame total:{sum(len(e['current_inside']) for e in cad['events'])}; native-policy expected total:{sum(len(e['native_inside']) for e in cad['events'])}. All12intervals would include a native frame; the four A intervals regain index coverage. New hand observability, landmark quality and recognition outcomes are UNKNOWN: higher-cadence features/perception were not executed. Qualified review remains blocked and did not block the independent learning diagnostic.

## Example predictions

Full reference/hypothesis/error data:prediction_examples.csv and per-epoch JSONs. Literal labels are unchanged.

{selected_examples}
## Hardware and uncertainty

{env['GPU']},total{env['GPU_total_bytes']/2**30:.3f}GiB,free at startup{env['GPU_free_bytes']/2**30:.3f}GiB; torch{env['torch']},CUDA{env['CUDA']}. Device tensors and gradients were exercised. Allocated guard2.5GiB/free256MiB/disk10GiB. No OOM or skipped/invalid batches are hidden. No paired CPU speed benchmark was performed, so no CUDA speedup claim. CUDA CTC nondeterminism warning is preserved in run.log.

## Decision and next action

Do not promote. If TRAIN remains above the50%learning screen and held-out errors remain near baseline, this is insufficient full-task learning; a tiny-subset pass does not remedy a long-tail task. The tail correction is not established as a material recognition fix. If TRAIN improves while held-out loss worsens, that is an overfitting/generalization signal, not success. Do not merely repeat longer same-data fits.

Next useful research step: review saved fullTRAIN token-support/error strata and input representation adequacy against a predeclared compact TRAIN learning control; independently obtain qualified adjudication for the12short/hand-missing events. Any architecture/representation or cadence experiment must be separately controlled. No reserved TEST, new acquisition or production action is authorized by these suggestions. No unseen-signer claim: held-out recording families contain known signers and only two family clusters.

## Integrity, scripts and continuation limits

{integ['status']}: {integ['existing_files_checked']}declared existing/new-source input files have identical before/after SHA256,size,mtime; changed paths:{len(integ['changed'])}. input_integrity.json provides exact scope. Production was not modified or independently hashed; no whole-workspace proof. Previous manifests/checkpoints/features/exclusions remain untouched.

Model/optimizer/CPU+CUDA RNG, completed epoch, best/staleness/collapse state and elapsed budget are saved every complete epoch. CONTROL_REVIEW.md records independent source review. The in-script --resume-candidate branch was not exercised; its exclusive integrity-output filename conflicts after a finished/failed invocation and its checkpoint hash needs external verification. Do not use it blindly. Completed checkpoint state is resumable in a new authorized continuation harness/directory; partial timed-out work has not been persisted. This completed bounded run needs no resume. Preserve the frozen script/config hashes rather than modifying them now.

Construction/training command (already executed; do not repeat over completed outputs):
{command}

Read-only validation command:
{verify}

Detailed commands/results:commands_and_results.txt. Final artifact hash index:artifact_hashes.sha256. All new files remain in {RUN}.
"""
out('MISSION_REPORT.md',report)
out('commands_and_results.txt',f"""Working directory:{ROOT}
Read latest review report, prior pilot summary, AGENTS.md, current trainer/core/config/status; source hash and GPU/process checks.
New directory created with UTC timestamp; no prior file changed.
Exact training:
{command}
Actual results:GateA={a['status']};GateB={b['status']};subsetepochs={b['completed_epochs']};subsetseconds={b['elapsed_seconds']};candidate={d.get('full_candidate_trained',False)};epochs={d.get('completed_epochs',0)};reason={d.get('reason')}.
Full warnings, progress and final decision:run.log.
FullTRAIN/heldout epoch metrics:training_metrics.csv;subset:subset_learning_log.csv.
Read-only finalization:
& '{py}' -B '{RUN/'finish_mission.py'}'
Read-only validation:
{verify}
No automatic repeat/resume/production promotion. Inputs preserved;TEST unused.
""")
files=[p for p in RUN.rglob('*') if p.is_file() and p.name!='artifact_hashes.sha256']
out('artifact_hashes.sha256','\n'.join(f"{sha(p)}  {p.relative_to(RUN).as_posix()}" for p in sorted(files))+'\n')
print(json.dumps(dict(report=str(RUN/'MISSION_REPORT.md'),artifacts_hashed=len(files),status=d['status'],candidate_WER=d.get('primary80_WER')),indent=2))

