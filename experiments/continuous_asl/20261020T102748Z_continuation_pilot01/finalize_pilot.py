"""Report the completed new pilot; never trains or modifies frozen sources."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[2]
DIAG=ROOT/'experiments/continuous_asl/20261020T051631Z_devwer_diagnosis01'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def save(name,value,text=False):
 with (RUN/name).open('x',encoding='utf-8') as f:
  if text:f.write(value)
  else:json.dump(value,f,indent=2);f.write('\n')

def main():
 status=read(RUN/'training_status.json');config=read(RUN/'config.json')
 with (RUN/'training_log.csv').open(newline='') as f:logs=list(csv.DictReader(f))
 if len(logs)!=status['completed_additional_epochs'] or digest(RUN/'train_pilot.py')!=status['training_script_sha256']:raise ValueError('PILOT_LOG_SCRIPT_CONTRACT')
 expected=read(DIAG/'preservation_before.json')['files']
 changed=[]
 for name,meta in expected.items():
  stat=Path(name).stat()
  if [stat.st_size,stat.st_mtime_ns]!=meta:changed.append(name)
 if changed:raise ValueError('FROZEN_INPUT_CHANGED:'+str(changed))
 gpu=[]
 for path in RUN.glob('gpu_*.csv'):
  with path.open() as f:
   for r in csv.reader(f):
    if len(r)!=5:continue
    try:gpu.append(float(r[2].strip().split()[0]))
    except ValueError:pass
 wall=[read(p) for p in RUN.glob('command_wall_*.json')]
 checkpoint_hashes={p.name:digest(p) for p in sorted((RUN/'checkpoints').glob('*.pt'))}
 summary={'execution':'PASS','recognition':'PARTIAL_NO_DEV_GAIN','training_status':status,'preservation_checked_files':len(expected),'preservation_changed':changed,
  'epoch_median_seconds':statistics.median(float(r['epoch_seconds']) for r in logs),'epoch_min_seconds':min(float(r['epoch_seconds']) for r in logs),
  'epoch_max_seconds':max(float(r['epoch_seconds']) for r in logs),'measured_training_presentations_per_epoch_second':status['utterance_presentations']/status['epoch_loop_seconds'],
  'whole_device_sampled_peak_used_mib':max(gpu) if gpu else None,'whole_device_samples':len(gpu),'command_wall_records':wall,
  'checkpoint_sha256':checkpoint_hashes,'reserved17_test_evaluated':False,'production_promoted':False,
  'scope':'Same frozen2119TRAIN/9oldDEV. Additional-budget test only; no signer-independent or production accuracy claim.'}
 save('execution_summary.json',summary)
 initial=read(RUN/'initial_train_monitor.json');best=read(RUN/'best_checkpoint_full_train.json');last=read(RUN/'last_checkpoint_full_train.json');dev=read(RUN/'best_checkpoint_dev.json')
 lines=['# Bounded continuation after the DEV-WER diagnosis','',
  '**PASS execution and preservation; PARTIAL recognition. Stopped after the initial three-epoch window because DEV did not improve.**','',
  'The user explicitly authorized this isolated research pilot after the completed engineering checks found no critical implementation bug or data-integrity failure. It continued the frozen epoch-17 weights, AdamW state and RNG state on the exact original 2,119 TRAIN / 9 DEV manifest. No acquisition or preprocessing was repeated. The prior epoch-2 selection and every earlier artifact remain unchanged.','',
  'Only the continuation budget and diagnostic stop protocol changed. Architecture48/RF61,378features, batch4/accumulation2, AdamW lr0.001/weight-decay0.0001, gradient clip5, FP32, literal vocabulary2741, original TRAIN normalization and greedyCTC blank0/UNK1 stayed fixed. New directory contains all outputs.','',
  '| Cumulative epoch | TRAIN optimization CTC loss | Fixed64 TRAIN monitor WER | DEV WER | DEV S/D/I | Epoch seconds |','|---|---:|---:|---:|---:|---:|']
 for row in logs:lines.append(f'| {row["epoch"]} | {float(row["train_optimization_loss"]):.6f} | {100*float(row["train_monitor_wer"]):.2f}% | {100*float(row["dev_wer"]):.2f}% | {row["substitutions"]}/{row["deletions"]}/{row["insertions"]} | {float(row["epoch_seconds"]):.4f} |')
 lines+=['',
  f'The initial fixed64 monitor had {100*initial["gloss_wer"]:.2f}%WER. Full2,119TRAIN WER improved from frozen epoch17:71.9532% to epoch20:{100*last["gloss_wer"]:.4f}% ({last["substitutions"]}S/{last["deletions"]}D/{last["insertions"]}I over{last["reference_tokens"]}tokens). These are TRAIN diagnostics, not generalization.',
  '',
  f'The new pilot selected epoch18 on its own DEV WER/loss tuple: full TRAIN WER{100*best["gloss_wer"]:.4f}%, DEV100% ({dev["substitutions"]}S/{dev["deletions"]}D/{dev["insertions"]}I). The original selected epoch2 is not changed. New local-best checkpoint SHA256: `{status["best_checkpoint"]["sha256"]}`. The final epoch20 checkpoint is also retained, with SHA256 `{checkpoint_hashes["epoch_020.pt"]}`.',
  '',
  'Predeclared gates: first3additional epochs; maximum8additional epochs/120seconds, wall checked between epochs. Extend beyond3only if DEV improves at least2of57tokens versus the epoch17 parent and TRAIN monitor loss does not worsen. Stop on nonfinite loss/gradients, VRAM/resource guard, TRAIN blank collapse, or lack of TRAIN learning after2epochs. DEV remained100%, so the continuation gate failed and status is NO_CREDIBLE_DEV_GAIN_INITIAL_WINDOW. It was not extended to8epochs. The two-token gate is a research heuristic, not statistical significance on this tiny dependent DEV.',
  '',
  f'Actual CUDA epoch loop: {status["epoch_loop_seconds"]:.4f}s; fit/reload/fullTRAINdiagnostics: {status["fit_reload_full_train_diagnostics_seconds"]:.4f}s. Complete process wall time including Python startup/data checks is separately recorded: {wall[0]["seconds"]:.4f}s. Median epoch: {summary["epoch_median_seconds"]:.4f}s. Peak PyTorch allocated: {status["peak_allocated_vram_bytes"]}bytes ({status["peak_allocated_vram_bytes"]/2**30:.4f}GiB); reserved: {status["peak_reserved_vram_bytes"]}bytes. Whole-device sampled occupancy peak: {summary["whole_device_sampled_peak_used_mib"]}MiB over{len(gpu)}samples at200ms; device samples can miss brief peaks and differ from allocator memory. No OOM occurred.',
  '',
  'Inputs/logits/loss are logged on cuda:0, optimizer state is inherited on CUDA, gradients are finite, and new optimizer steps completed. Each new checkpoint contains optimizer and RNG state. The selected new checkpoint was reloaded from disk and reproduced its DEV hypotheses exactly. Fixed seeds do not guarantee bitwise CUDA CTC backward determinism; the framework warning is preserved in the console log.',
  '',
  f'Frozen manifest SHA256: `{status["manifest_sha256"]}`. All2,119unique TRAIN utterances were consumed in each of3epochs (6,357presentations). The nine DEV examples are unchanged. The reserved17TEST set was not used. The other906eligible pool was not mixed in; all3,025eligible utterances and the six overlap/one frame-count exclusions remain preserved. Final metadata checks passed for{len(expected)}protected/indexed paths.',
  '',
  '## Prediction example','',
  'Literal references/hypotheses below use the existing conventions; all nine pairs for every epoch are saved in dev_epoch_*.json.','',
  '| Reference | New selected epoch18 hypothesis |','|---|---|']
 for example in dev['predictions'][:3]:lines.append('| `'+ ' '.join(example['reference'])+'` | `'+(' '.join(example['hypothesis']) if example['hypothesis'] else '<EMPTY>')+'` |')
 lines+=['',
  'Interpretation: more optimization improved TRAIN without any DEV token-error reduction. This is consistent with the diagnosis of weak early fitting plus source/label/evaluation mismatch, but does not uniquely identify a cause. The same old DEV is a valid fixed literal-token diagnostic; signer-independent generalization remains unverified. Source scope, OOV tokens and feature-observation differences are quantified in the linked diagnostic directory. No checkpoint beats the frozen85.96%DEV baseline and none is promoted.',
  '',
  'Next action: finish signer/source mapping and establish a same-convention independent evaluation protocol before another optimization loop. Do not use the reserved17TEST set for that tuning. No automatic longer training is recommended after this failed DEV gate.','',
  'Exact execution command (completed; wrapper verifies the completed checkpoint on repeat):','',
  '```powershell',f"& '{RUN/'Run-Pilot.ps1'}'",'```','',
  'Read-only verification:', '', '```powershell',f"& '{RUN/'Verify-Pilot.ps1'}'",'```','',
  'Full diagnostic report: `../20261020T051631Z_devwer_diagnosis01/DIAGNOSIS.md`. The epoch2 checkpoint remains frozen and must not be promoted. Production integration, thresholds, Android app data, APKs, avatars and GLB assets are unchanged.']
 save('PILOT_REPORT.md','\n'.join(lines)+'\n',True)
 save('REVIEW.md','# Research boundary review\n\nPASS plan alignment: reused the critical diagnosis and executed only the explicitly authorized bounded continuation. PASS system integrity within this research scope: same frozen data/schema/vocabulary/normalizer/decoder; no reserved TEST or production writes. PARTIAL evaluation readiness: original nine-item cross-corpus DEV and incomplete signer mapping do not establish product accuracy. BLOCKED promotion. Two important open issues: signer-independent evaluation and cross-version label-scope review. No critical implementation defect was found by the bounded checks. Recommendation: stop optimization after the failed DEV gate and resolve evaluation scope.\n',True)
 files=[p for p in sorted(RUN.rglob('*')) if p.is_file() and p.name!='artifact_hashes.sha256']
 save('artifact_hashes.sha256','# Closed pilot artifacts; excludes this checksum file.\n'+''.join(f'{digest(p)}  {p.relative_to(RUN).as_posix()}\n' for p in files),True)
 print(json.dumps({'status':'PASS_PILOT_REPORTED','stop':status['status'],'epochs':status['completed_additional_epochs'],'last_train_wer':last['gloss_wer'],
   'dev_wer':dev['gloss_wer'],'checkpoint_sha256':status['best_checkpoint']['sha256'],'artifacts':len(files),'frozen_metadata_verified':len(expected)}),flush=True)

if __name__=='__main__':main()
