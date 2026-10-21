"""Summarize completed evidence; never fit, select, or change a checkpoint."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import csv,hashlib,json

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[2]
OLD=RUN.parent/'20261020T144622Z_compact_ctc_control01'
PREV=RUN.parent/'20261020T143609Z_learning_diagnostic01'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def link(p,label=None):return f'[{label or p.name}](<{p.as_posix()}>)'

def main():
 results=read(RUN/'RESULTS.json');d=results['decision'];h=results['heldout80'];coverage=results['coverage']
 log=(RUN/'phone_instrumentation.log').read_text(encoding='utf-8-sig')
 assert 'OK (2 tests)' in log
 raw=log.split('APPHEAR_MODEL_TIMING=',1)[1]
 timing=json.JSONDecoder().raw_decode(raw)[0]
 after=read(RUN/'phone_after.json');assert after['apphear_apk_path_unchanged']
 provenance=read(RUN/'phone_parity/harness_provenance.json')
 for p,v in provenance['source_hashes'].items():assert sha(p)==v
 summary=dict(status='PASS',device=timing['device'],model_numerical_parity=True,synthetic_model_timing=timing['results'],scope=timing['scope'],apphear_package_unchanged=True,model_under_test='Existing packaged full-task epoch12; NOT new192 checkpoint',camera_used=False,participant_data_used=False,feature_math_or_full_app_lifecycle_tested=False)
 (RUN/'PHONE_PARITY_RESULTS.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
 strata=list(csv.DictReader((RUN/'HELDOUT_SUPPORT_STRATA.csv').open(encoding='utf-8-sig')))
 rows=results['learning_comparison'];now=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
 lines=[
 '# Controlled 192-example continuous-ASL learning experiment', '',
 f'Generated {now} (Asia/Shanghai). Exactly one new CUDA training run was completed.', '',
 '**PASS bounded TRAIN learnability and Android CPU model parity; PARTIAL overall recognition; BLOCKED promotion, new participant collection and independent streaming acceptance.**', '',
 '## Main finding', '',
 f'The nested 192-example control learned its TRAIN data: **{100*d["final_WER"]:.2f}% WER**, versus 12.04% in the frozen 96-example control. Its one-time post-fit evaluation scored **{100*h["WER"]:.2f}% WER** on 80 held-out sequences, with **0/80 exact matches**. This does not improve recognition or justify changing the app model.', '',
 'There is no demonstrated learnability breakdown at 192 examples. The model can memorize a larger, varied subset. Generalization remains poor, including sequences whose labels were fitted. Label coverage, representation/annotation adequacy, recording-family distribution and optimization remain interacting explanations; no single cause is proven.', '',
 '## Frozen design and execution', '',
 '- Retained every original 96 TRAIN ID, literal target, feature path/hash and selection descriptor. Added exactly 16 unused authorized TRAIN utterances per family: 32 per family, six families, three known signers, 192 unique utterances total.',
 f'- Targets: **631 unique labels, 1,464 token occurrences**. Full output vocabulary remains **2,287 gloss labels + blank0/UNK1**. Only the ASLLRP3 literal dominant-realized-occurrence pool was used.',
 '- Fixed 48-channel causal CTC, 378 features, original full-TRAIN normalization, seed20261020, FP32 AdamW lr0.001/weight_decay0.0001, batch4, accumulation2 with actual-tail scaling, clip5 and literal greedy CTC. No feature extraction or label changes.',
 '- Initial model tensors were verified **byte-identical** to the 96-example initialization. Checkpoint container hashes differ because their configuration/subset metadata differ.',
 '- Fixed **30 complete passes**; maximum720 updates/90-second soft stop. Measurements at0,1,5,10,15,20,25,30. No stop at an earlier successful learning gate; no held-out-based selection.',
 '- Gate reused unchanged: TRAIN WER≤30%, CTC≤50%initial, WER reduction≥20percentage points, and each family/rare-bearing/upper-input-quartile WER≤50%. The final control passed all gates.',
 '- Actual **720 optimizer steps / 5,760 utterance presentations**, versus360/2,880 in the 96-example control. This is matched passes, **not matched compute**. Larger vocabulary coverage and example composition also change with added data.',
 f'- Measured fit/evaluation/checkpoint stage: **{d["elapsed_including_initial_and_final_scoring_seconds"]:.3f}s**. Sum of30 optimizer-epoch durations **{results["optimizer_epoch_seconds_total"]:.3f}s**, mean **{results["optimizer_epoch_seconds_mean"]:.3f}s/epoch**. First stage excludes imports/preparation and separate held-out scoring; per-epoch timing excludes periodic scoring/save.',
 f'- Peak PyTorch allocated **{d["peak_allocated_bytes"]/1024**2:.2f}MiB**, reserved **{d["peak_reserved_bytes"]/1024**2:.2f}MiB**. Separate nvidia-smi device-wide observed peak **{results["devicewide_peak_used_MiB"]:.0f}MiB**, including CUDA/context overhead. RTX3050 Laptop4GiB. No OOM, skipped batch or nonfinite failure.',
 '- Early blank collapse resolved: epoch1 all blank, epoch5 WER97.40%, epoch15 WER74.86%, epoch25 WER20.29%, epoch30 WER10.59%. CTC CUDA backward remains nondeterministic in warn-only mode; no seed-equivalent causality claim.', '',
 '## Results by scope', '',
 '| Scope | Utterances | Reference tokens | WER | Exact sequences | CTC loss |',
 '|---|---:|---:|---:|---:|---:|',
 ]
 for r in rows:lines.append(f'| {r["scope"]} | {r["utterances"]} | {r["reference_tokens"]} | {100*r["WER"]:.3f}% | {r["exact_sequences"]}/{r["utterances"]} | {r["CTC_loss"]:.6f} |')
 lines += [
 '',
 'The original96 portion remains at11.90% WER, close to its previous12.04%; exact sequence matches decrease48→34. The added96 reach9.18% WER. Aggregate10.59% should not be read as a proven causal quality gain because the denominators/composition and compute differ. New192 exact sequence accuracy is76/192=39.58%, versus48/96=50% in the old control.', '',
 'Final TRAIN errors:100substitutions/43deletions/12insertions,155/1,464. No empty predictions;610distinct predicted labels;81.56%argmax blank frames. This blank fraction is not mean blank probability or evidence of collapse by itself.', '',
 '## Held-out validity, coverage and limitations', '',
 '- Epoch30 was frozen and hashed before the80evaluation. All80feature hashes/schemas/timestamps/target joins/CTC lengths passed. No recording family crosses the192/80boundary. Targets map exactly into the fixed vocabulary. The original stronger27 are reported as a subgroup and do not select the checkpoint.',
 '- These are known signers and two held-out recording families. The80set has been development-exposed in earlier research. It is held out from this fit, **not a pristine final TEST or unseen-signer evaluation**.',
 f'- Model vocabulary covers485/485reference tokens. Actual192training targets cover**386/485({100*386/485:.2f}%)**; only**22/80**sequences have all reference labels represented in192TRAIN. The99unsupported tokens limit interpretation of the complete80result.',
 '- The fully supported22 still score**97.50%WER**,117/120edits and0/22exact. The remaining58 score102.19%WER. These descriptive strata retain the original80 score; no restricted subset replaces the main metric or tunes the model.',
 '- The original27strong-support definition refers to support in1500TRAIN, not support in192TRAIN. Its100.68%WER is reported under that unchanged definition.',
 '- The earlier1500-fit checkpoint scored95.67%WER on the same80 under the same label/decoder/normalization conventions, but used different training membership, target support, pass count and checkpoint-selection policy. The new101.03% is descriptively worse, not an equal-budget architecture comparison.',
 '- New held-out output has338S/124D/28I and389hypothesis tokens, with0empty outputs. Substitutions now dominate. Merely suppressing blank or forcing longer predictions has no demonstrated benefit.', '',
 '## Phone parity and measured model time', '',
 'Both synthetic tests passed on Samsung **SM-A226B**. A separate camera-free package compiled the actual AppHear model-loader source and synthetic fixtures. The installed AppHear APK path stayed unchanged; no AppHear install/clear, participant capture or microphone use occurred.', '',
 '**This tests the existing packaged epoch12 model, not promotion/export of the new192 checkpoint.** The harness uses contract constants rather than camera feature extraction, so it establishes CPU model numerical parity, not whole-app lifecycle/feature/camera parity.', '',
 '| Synthetic input observations | Timed runs after3warmups | p50 model ms | p95 model ms |',
 '|---|---:|---:|---:|',
 ]
 for r in timing['results']:lines.append(f'| {r["observations"]} | {r["runs"]} | {r["p50_ms"]:.3f} | {r["p95_ms"]:.3f} |')
 lines += [
 '',
 'These are CPU inference+tensor/output-copy timings only. No camera cadence, missing-hand rate, streaming false activations, missed signs, sign boundaries or end-to-end latency was measured. Utterance-level metrics cannot fill those gaps.', '',
 '## Data support, linguistic review and collection', '',
 'The existing1500TRAIN pool still has**1,182singleton labels** and**1,808labels with fewer than5distinct supporting utterances**. Support counts use distinct utterances, not repeated token occurrences. The new control adds zero newly acquired examples; the earlier3,121eligible/3,025ever-fitted inventory remains unchanged.', '',
 '- '+link(RUN/'LABEL_COLLECTION_PRIORITIES.csv')+' lists all1,808under-supported literal TRAIN labels with existing signer/family coverage and an explicit planning gap to5utterances. Five is a collection-planning target, not a validated accuracy threshold or instruction to collect every row.',
 '- No qualified ASL reviewer is available, per the user. No consent/permission for new participant recordings has been established. No recording, upload, new annotation or invented gloss was performed.',
 '- Smallest next step: qualified review of the existing12TRAIN events in '+link(RUN.parent/'20261020T141110Z_alignment_observability_review01/review_manifest.csv')+'. Reuse that frozen package to distinguish label/timing/visibility problems before another large fit.',
 '- Then choose a small practical set of reviewed existing labels and collect natural continuous phrases with repeated occurrences across consented qualified signers. Reserve independent new signers and recording sessions for evaluation before collecting/training. Preserve authentic annotation provenance; do not translate English into fabricated gloss ground truth.',
 '- Reuse the existing consented-pilot protocol/templates under `20261020T004837_separate02/pilot_package_20261020T014259`. No collection begins until permissions and a qualified annotator are established. Keep ASLLRP3/NCSLGR2 separate.', '',
 '## Acceptance protocol still required', '',
 'Before further model selection, freeze consenting signer/session splits and labels, capture conditions, decoder and the following definitions/thresholds. No thresholds were fit in this experiment.', '',
 '| Measure | Required evidence | Current status |',
 '|---|---|---|',
 '| Continuous gloss WER + exact sequences | Authentic held-out utterance/stream references; per-signer/source results; frozen comparison with confidence limits | Only known-signer development80 diagnostic available |',
 '| False activations | Unconfirmed/confirmed emissions per no-sign minute; idle, conversational motion and hands-visible negatives | BLOCKED: no consented no-sign set |',
 '| Missed signs | Annotated reference gloss deletions/recall and missed events by signer/support | Token edit counts exist; genuine streaming event recall BLOCKED |',
 '| Boundaries | Qualified onset/release labels; onset/end error distributions; split/merge/repeat counts | BLOCKED: qualified review unavailable |',
 '| End-to-end latency | Capture→draft and sign-end→review-ready p50/p95, delivered cadence and missing observations on actual streams | Model-only synthetic time PASS; full pipeline BLOCKED |',
 '| Unseen-signer accuracy | Signers absent from all training/selection; frozen final evaluation | BLOCKED: no verified independent signer pilot |',
 '',
 '## Decision and exact artifacts', '',
 '**Do not promote or automatically launch a larger fit.** The192learning control passes, while held-out transfer remains unsuccessful even for supported labels. The result narrows the failure: fundamental inability to learn192examples or a persistent all-blank decoder is not supported. Feature/label alignment adequacy and support across people/contexts need evidence before assuming additional epochs or a bigger model will solve it.', '',
 'The next practical action is qualified review of existing examples and permissioned targeted repetition/evaluation collection. A later384nested learning probe is an option for a separately bounded study, not executed or automatically authorized here. Neither the app model nor its thresholds changed.', '',
 '- New checkpoint: '+link(RUN/d['checkpoint']['path'].replace('\\','/')), 
 '- Checkpoint SHA256: `'+d['checkpoint']['sha256']+'`.',
 '- Frozen192manifest SHA256: `'+sha(RUN/'subset_manifest.json')+'`.',
 '- Logs: '+link(RUN/'run.log')+', '+link(RUN/'epoch_timing.csv')+', '+link(RUN/'gpu_monitor.csv')+'.',
 '- Predictions: '+link(RUN/'PREDICTION_EXAMPLES.csv')+'; '+link(RUN/'HELDOUT_SUPPORT_STRATA.csv')+'.',
 '- Phone evidence: '+link(RUN/'phone_instrumentation.log')+'; '+link(RUN/'PHONE_PARITY_RESULTS.json')+'.',
 '- '+link(RUN/'control_input_integrity.json')+' and '+link(RUN/'postfit_evaluation_integrity.json')+' preserve exact input scope. Prior experiments, exclusions, reserved17TEST, existing checkpoints/manifests and production assets remain unchanged.',
 '',
 'Revalidate completed artifacts without fitting:', '',
 '```powershell',
 "$python = '"+str(ROOT/'experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe')+"'",
 "& $python -B '"+str(RUN/'verify192.py')+"'",
 '```', '',
 'The initial training-launch path omitted one nested project-directory component and failed before Python opened the script. The corrected command completed the single fit. The first harness build lost a dotted PowerShell argument; quoting corrected it. The first device command referenced absent SDKbuild-tools36; discovery selected installed35. These setup failures caused no extra fit or model/data change and are recorded in commands_and_results.txt.',
 ]
 (RUN/'PILOT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps(dict(status='PARTIAL',training='PASS',phone_model_parity='PASS',heldout_WER=h['WER'],new_collection='BLOCKED',qualified_review='BLOCKED',report=str(RUN/'PILOT_REPORT.md')),indent=2))

if __name__=='__main__':main()
