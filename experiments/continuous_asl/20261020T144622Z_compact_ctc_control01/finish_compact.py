"""New report and scoped integrity verification; no model load, prediction or fitting."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import csv,hashlib,json
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[2]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def out(name,text):
 with (RUN/name).open('x',encoding='utf-8') as f:f.write(text)
def save(name,v):out(name,json.dumps(v,indent=2)+'\n')
d=read(RUN/'decision.json');diag=read(RUN/'diagnosis_summary.json');cov=read(RUN/'subset_coverage.json');cfg=read(RUN/'control_config.json')
initial=read(RUN/'initial_predictions.json');final=read(RUN/'final_predictions.json');env=read(RUN/'environment.json')
before=read(RUN/'control_input_hashes_before.json')
for group in [read(RUN/'diagnosis_input_hashes.json')['before'],read(RUN/'frozen_control_inputs.json')['before']]:
 for p,e in group.items():
  if p in before:assert before[p]==e,p
  before[p]=e
changed=[]
for p,e in before.items():
 q=Path(p);s=q.stat()
 if sha(q)!=e['sha256'] or s.st_size!=e['bytes'] or s.st_mtime_ns!=e['mtime_ns']:changed.append(p)
assert not changed,changed
assert sha(RUN/d['checkpoint']['path'])==d['checkpoint']['sha256']
integrity=dict(status='PASS',input_files_checked=len(before),before=before,changed=[],no_TEST=True,production_writes=False,scope='Explicit prior artifacts, saved TRAIN/heldout predictions/metadata,96selected TRAIN features,and frozen current plans; not whole workspace')
save('input_integrity.json',integrity)
passed=(final['WER']<=cfg['pass_WER'] and final['CTC_loss']<=cfg['pass_loss_fraction_of_initial']*initial['CTC_loss'] and initial['WER']-final['WER']>=cfg['minimum_absolute_WER_drop'] and max(final['family_WER'].values())<=cfg['max_family_WER'] and final['rare_bearing_WER']<=cfg['max_rare_bearing_WER'] and final['upper_input_quartile_WER']<=cfg['max_long_input_WER'])
assert passed==d['compact_control_learned']
save('mission_decision.json',dict(status='PARTIAL',numerical_pipeline='PASS',compact_learning='PASS' if passed else 'FAIL',full_task_cause='PARTIAL_NOT_CAUSALLY_IDENTIFIED',compact_control_decision_path='decision.json',next_single_variable='TRAIN-set size only:96 to nested192; no automatic execution',full_training_started=False,promote=False,TEST_used=False))
familyrows='\n'.join(f"|{f}|16|{100*w:.3f}%|" for f,w in d['family_WER'].items())
trajectory=[]
with (RUN/'diagnostic_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
 for r in csv.DictReader(f):trajectory.append(f"|{r['epoch']}|{r['optimizer_steps']}|{100*float(r['WER']):.3f}%|{float(r['CTC_loss']):.4f}|{float(r['blank_fraction']):.4f}|{float(r['prediction_reference_length_ratio']):.4f}|{r['token_diversity']}|")
examples='\n'.join(f"- {p['sample_id']}: reference {json.dumps(p['reference'],ensure_ascii=False)}; hypothesis {json.dumps(p['hypothesis'],ensure_ascii=False)}." for p in final['predictions'][:5])
py=ROOT/'experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe'
cmd=lambda name,args='':f"& '{py}' -B '{RUN/name}'{args}"
report=f"""# Full-task CTC diagnosis and one compact learning control

Generated {datetime.now(timezone(timedelta(hours=8))).isoformat()} (Asia/Shanghai).

**PARTIAL overall diagnosis; PASS numerical checks and compact learning.** One new96-utterance TRAIN-only CUDA fit ran. It passed every predeclared global/subgroup learning gate. No full-data fit, prior16-example rerun, held-out evaluation of the new control, acquisition, new feature extraction or promotion occurred.

## Confirmed failure signature from saved full-task evidence

The previous selected epoch12 checkpoint remains unchanged: SHA25614d0e4669674265ea2be8a3161fa6699e4f813daf4b97b384863dce1daa038ff. Historical metrics are preserved, not replaced by this easier-scope experiment.

|Saved scope|WER|S/D/I|Reference tokens|Hypothesis tokens|Hyp/ref length|Argmax blank|Distinct predictions|
|---|---:|---|---:|---:|---:|---:|---:|
|TRAIN1500|88.4005%|1526/9444/12|12423|2991|0.2408|95.5894%|450|
|Compatible held80|95.6701%|136/328/0|485|157|0.3237|93.1360%|49|

Deletions and short output dominate; this is not strictly a single-token collapse. TRAIN has236empty hypotheses; held80has5. Saved posterior probabilities/logits do not exist, so posterior confidence/entropy/calibration are UNKNOWN. Blank figures are argmax frame fractions, not mean blank probability.

All485held-out reference tokens have TRAIN-seen labels. Aligned held-out recall is21/251=8.37% for labels seen in20+distinct TRAIN utterances and0/234 for lower support. TRAIN20+recall is only15.68%. Thus rare/unseen labels cannot be the sole explanation. Long-tail pressure is substantial:1182/2287labels occur in one distinct TRAIN utterance and1808/2287in fewer than5. diagnosis_by_label.csv preserves every literal label's distinct-utterance support, token counts, aligned recall, deletions/substitutions and confusion alternatives. Alignments use the verified diagonal/substitute,delete,insert tie order; label recall is alignment-dependent, not a new ground-truth annotation.

All1580saved sequences are CTC feasible (minimum slack5);17adjacent-repeat targets are accounted for. No empty targets, blank/UNK targets, downsampling length mismatch, normalization/vocabulary/decoder drift or frozen-membership inconsistency was found. Existing exact feature reconstruction and source joins were reused; selected96feature files were rechecked for hash/schema/dtype/finite/timestamp/transcript/length integrity, not regenerated.

FullTRAIN WER continues100%epoch1→88.40%epoch12→84.61%epoch13; loss6.591→3.728. Held-out loss worsens6.627epoch1→7.606epoch12. These show incomplete full-task learning plus weak transfer, not training convergence or proof that further epochs solve it.

Cached observation-gap associations are inconsistent across splits: TRAIN gap-flagged WER91.35% versus79.16% without flag, while held-out reverses94.94% versus97.67%. Duration/support/source are confounded; no causal visibility claim follows. Missing-hand flags are detector states, not physical absence. Detailed strata, temporal lengths and per-utterance S/D/I/length ratios are in the diagnosis CSV/JSON files.

## Frozen compact design

The manifest was selected before this fit using only authorized internalTRAIN metadata/targets, never held-out outcomes. Seed prefix:compact96-seed20261020; model seed20261020. Six TRAIN families receive16utterances each. Greedy selection first covers unused combinations of family/input-length quartile/target-length quartile/observation category/rare-bearing status, then masters and underrepresented marginal categories, with stable SHA ties. Existing manifests and source labels were not edited.

Coverage:96utterances,all3available TRAIN signers,all34masters,404literal labels,756tokens. Among selected labels, full-TRAIN distinct-utterance support is68singletons,90with2–4support,82with5–9,78with10–19,and86with20+. Input lengths15–100observations; target lengths2–27tokens. Input quartile counts26/26/24/20; target counts26/25/23/22. There are33utterances with unsampled publisher intervals and54with sampled/no-observed-hand intervals; categories may overlap. No low-quality example or literal target was discarded.

Equal-family quotas overrepresent small families; signer distribution is16/64/16. This is coverage-balanced TRAIN learning evidence, not a random population sample or unseen-signer test. It covers404/2287labels, not the entire task or all ambiguous examples. The previous16-example experiment was not rerun.

Frozen subset SHA256:c7b78f1aef43b5501fce5fafbf69e91753e5d9d85cb594a3e18fc693ecfb5198. All subset IDs are in the original1500TRAIN; full2,289-class vocabulary, blank0/UNK1, saved fullTRAIN-only normalization,378features and original48-channel causal model remain unchanged. AdamW0.001/weightdecay0.0001,FP32,batch4,accumulation2,clip5. Actual-group-count scaling preserves the prior tail correction. No warm start, architecture/feature/cadence/label/optimizer change was bundled.

Predeclared limits:120epochs,1440optimizer steps or90second stop request. Evaluate initial,epoch1,and every5epochs. Pass requires WER<=30%,loss<=50%initial,>=20percentage-point WER reduction,and WER<=50%for every family,rare-bearing utterances and upper input-length quartile. Numerical/OOM/resource faults stop immediately. No new full training was authorized automatically.

## Measured compact learning

Passed at epoch{d['complete_epochs']},optimizer steps{d['optimizer_steps']}. All metrics below are in-sample TRAIN; they are not comparable held-out recognition scores.

|Epoch|Optimizer steps|WER|CTC loss|Argmax blank|Hyp/ref length|Distinct predicted labels|
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(trajectory)}

Final errors:{final['substitutions']}S/{final['deletions']}D/{final['insertions']}I = {final['substitutions']+final['deletions']+final['insertions']}/{final['reference_tokens']}. Empty outputs:{final['empty_outputs']}. CTC loss fell{100*(1-final['CTC_loss']/initial['CTC_loss']):.2f}%. Random initial WER>100% reflects insertion-heavy token sequences, not negative accuracy.

|Family|Utterances|Final WER|
|---|---:|---:|
{familyrows}

Rare-bearing WER:{100*d['rare_bearing_WER']:.3f}%; upper-input-quartile WER:{100*d['upper_input_quartile_WER']:.3f}%. Worst family:{100*max(d['family_WER'].values()):.3f}%. Thus the improvement is not confined to one easy family or short examples. Blank fraction remains82.25%while WER is12.04%; a high CTC blank fraction alone is not evidence of failed recognition.

Measured compact-stage elapsed:{d['elapsed_including_initial_and_final_scoring_seconds']:.3f}s,including initial/final scoring and checkpoint I/O,but excluding Python/CUDA import and preparation/final integrity. Peak allocated:{d['peak_allocated_bytes']:,}bytes ({d['peak_allocated_bytes']/2**20:.2f}MiB); reserved:{d['peak_reserved_bytes']/2**20:.2f}MiB. GPU:{env['GPU']},torch{env['torch']},CUDA{env['CUDA']}. No OOM/nonfinite/invalid batch occurred. The90sbound is a soft group-boundary stop request; final scoring/hashing may add overhead. Here it passed well within budget. Diagnostic fit_seconds includes periodic evaluation/save overhead and is not pure optimizer time. No CPU/GPU speedup was measured or claimed. CUDA CTC nondeterminism warning remains in run.log.

Checkpoint:{RUN/d['checkpoint']['path']}
SHA256:{d['checkpoint']['sha256']}

The checkpoint contains model,optimizer,CPU/CUDA RNG,exact vocabulary/config/subset hashes,completed epoch and optimizer count. Partial checkpoints,if a budget stop occurs,include shuffled IDs and next group index at an optimizer boundary. No resume CLI was implemented or exercised; state is sufficient for a separately authorized continuation harness. This successful fit needs no continuation.

## Examples

Complete final examples:final_predictions.json. No labels were silently normalized or substituted.

{examples}

## Evidence versus hypothesis

|Explanation|Assessment|
|---|---|
|Fundamentally invalid CTC/decoder/target mapping|NOT_EVIDENCED; saved checks and representative compact learning pass.|
|Inadequate full-task learning under the previous bound|CONFIRMED state; not fullTRAIN convergence. Compact success strengthens this diagnosis but does not identify the sufficient remedy.|
|Severe long-tail sparsity as the sole cause|NOT_SUPPORTED; common labels fail too. Sparse support remains a plausible contributor to optimization and transfer.|
|Optimization dynamics/task scale/effective exposure|LIKELY contributor, not causally isolated: same unchanged pipeline learns96varied examples in30passes, whereas full1500remains poor after13passes. Dataset size,interference and per-example exposure differ.|
|48-channel capacity/representation limit at full scale|POSSIBLE; compact success rules out inability to learn these404labels,not capacity/information adequacy for2287labels.|
|Missing input observability/annotation semantics|POSSIBLE and unresolved; compact includes flagged samples but cannot establish their visual/linguistic sufficiency.|
|New held-out/unseen-signer improvement|NOT_TESTED by this compact run; old held-out95.67%remains frozen.|

No single root cause is established. The strongest supported explanation is a scale-dependent full-task learning failure with long-tail/representation/optimization factors still interacting,rather than a demonstrated blank-ID or WER implementation defect.

## Next single experiment — recommendation only

Test **TRAIN-set size only:96→192 nested utterances**. Retain these exact96and add16deterministic unused TRAIN utterances per family under the same coverage rule. Keep full vocabulary,original1500-fitted normalization,model48,features,cadence,labels,optimizer,seed and30complete passes/evaluation schedule fixed. Use matched strata and report label-support/composition drift. Dataset size is the intended changed variable; total optimizer steps necessarily double360→720and must be reported as a consequence,not claimed matched compute.

This is a cheap scale probe before full training: if learning deteriorates sharply,it narrows the scaling/interference/capacity question; if192alsolearns,it still does not prove full-task transfer or that more full-data epochs suffice. A later matched-step control may be needed for causality. Do not change channels,blank bias,loss weights,labels or cadence simultaneously. No192fit or full new training ran now; any follow-up requires an explicit bounded compute allowance.

## Integrity, commands and disposition

PASS:{len(before)}explicit input paths have unchanged SHA256,size,mtime. Preparation verified all90prior output hashes; Phase1 verified its38saved inputs; the control rechecked96features and frozen local copies/manifest/config. Exact merged scope:input_integrity.json. No existing checkpoint,label,manifest,feature,report,production/Android/GLB asset or frozen evaluation was modified. Production assets were not independently hashed; no whole-workspace claim. No TEST access,acquisition,install,Git change or competing GPU job.

decision.json records PASS for the compact gate. mission_decision.json records PARTIAL for unresolved overall cause. Candidate promotion:NO; no product accuracy or unseen-signer claim.

Executed preparation:
{cmd('control.py',' --prepare')}
{cmd('verify_prepared.py')}

Executed one fit:
{cmd('control.py',' --train')} *> '{RUN/'run.log'}'

Finalization:
{cmd('finish_compact.py')}

To validate completed artifacts without retraining:
{cmd('verify_compact.py')}

commands_and_results.txt records commands and results. Do not rerun preparation/training over completed outputs.
"""
out('MISSION_REPORT.md',report)
out('commands_and_results.txt',f"""Working directory:{ROOT}
Read completed prior MISSION_REPORT.md,decision.json,training_metrics.csv,prediction_examples.csv,artifact_hashes.sha256,run.log,CONTROL_REVIEW.md.
GPU/process check:nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader
No competing Python/GPU job found.
{cmd('control.py',' --prepare')}
Result:90prior artifact hashes PASS;96features PASS;404labels/756tokens;subset/config frozen.
{cmd('verify_prepared.py')}
Result:9local prepared inputs hash/membership PASS;subset hashc7b78f1aef43b5501fce5fafbf69e91753e5d9d85cb594a3e18fc693ecfb5198.
Phase1 command/execution detail:diagnosis_execution_notes.txt;numeric validityPASS;no posterior arrays saved.
{cmd('control.py',' --train')} *> '{RUN/'run.log'}'
Exit0;one fit;30epochs/360steps;WER{d['initial_WER']}->{d['final_WER']};CTC{d['initial_CTC']}->{d['final_CTC']};{d['elapsed_including_initial_and_final_scoring_seconds']}s;peak{d['peak_allocated_bytes']}bytes;no full training.
{cmd('finish_compact.py')}
Creates final report,merged scoped integrity,mission decision,output hashes;no fitting.
{cmd('verify_compact.py')}
Read-only current validation command.
No prior16rerun;noheld-outselection/new evaluation;noTEST;no promotion;no acquisition or production changes.
""")
paths=[p for p in RUN.rglob('*') if p.is_file() and p.name!='artifact_hashes.sha256']
out('artifact_hashes.sha256','\n'.join(f"{sha(p)}  {p.relative_to(RUN).as_posix()}" for p in sorted(paths))+'\n')
print(json.dumps(dict(report=str(RUN/'MISSION_REPORT.md'),input_paths_verified=len(before),new_outputs_hashed=len(paths),overall='PARTIAL',compact_gate='PASS')))

