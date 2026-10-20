# Active execution checkpoint — historical snapshot, not final results

Actual repo: D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite.
Latest user directive: attachment0989cb37-5f10-498a-aae0-2853e580fe5d,
maximum practical speed, no waiting for2,000 before useful bounded training.
100GiB total acquisition cap explicitly authorized. No production changes.

Completed: expand04 and05 fits preserved;552 unique prepared old catalog;
expand06 added451 verified NCSLGR TRAIN utterances from231 accepted media
sources (232 downloaded/decoded, one source rejected for overlapping spans).
Eligible NCSLGR TRAIN906, fixed9DEV, reserved17TEST untouched.

ASLLRP official CSV and archive downloaded; archive CRC passed;2126 exact
annotation/media joins. Six publisher utterance overlaps rejected without
trimming. Two cached feature records are retained but excluded; four later
records have explicit metadata rejection files. Global event-ID collisions0.
Expected usable ASLLRP subset is not claimed until processing and canonical
checks complete. All completed features must be reused.

Expand07 numerical diagnostic v1 is preserved but invalid as a main gate.
Clean v2 passed with8 eligible TRAIN-only utterances:75updates, TRAIN WER
2.941%, CTC17.4526→0.11523,6.253s fit/reload, checkpoint hash in
overfit_diagnostic_v2/result.json. Do not repeat it. Warm-start draft unexecuted
and superseded; never invoke its Run-Bounded.ps1.

Expand09 real pilot COMPLETED:906TRAIN/9DEV,20epochs, wall-budget stop,
selected13, DEV WER96.4912% (16S39D0I/57), final TRAIN WER51.7047%, selected
TRAIN WER73.4853%; monitor loss ratio0.0727085. Learning gate PASS. Checkpoint
SHA5cb69c89ee91210debca4afcc6f45ec9b9b91fb147730e667f8fd18461c13ebd,
manifestSHAefcb25826cab6d6ac5361ad80f12f65d722fdf483369d755c62f9791515cece6.
Successful script train_pilot.v2.py, wrapper Run-PilotV2.ps1. Original failed
smoke used an empty stratified-monitor intersection before epoch1; preserved
models/smoke/logs and fixed into smoke_v2, without changing the manifest.

ACTIVE CPU jobs: original unchanged prepare_asllrp.py, four disjoint global
index modulo4 shards. Tool sessions42658,45977,24650,70043 correspond0,1,2,3.
Earlier primary/tail jobs were stopped only after ownership checks to prevent
catch-up and then balance measured clip durations. All1115 completed feature
hashes passed after balancing;0 incomplete outputs. Those old jobs' exit1
logs are retained and are intentional worker maintenance, not data failures.

ACTIVE staged driver: tool session22822, expand08/Run-FinishAndTrain.ps1.
It waits for all scheduled ASLLRP rows or explicit rejection records, freezes
expand06/canonical, chooses the largest internally consistent publisher label
pool, requires clean v2 diagnostic AND expand09 learning gate, then executes
tests, CUDA smoke and one bounded60epoch/600s/patience15 fit. Same48-channel
model, TRAIN-only norm/vocab, exact original9DEV and greedyCTC. No TEST reads.
Do not launch another copy while it is active. Its logs/corpus_freeze_*.log
provide saved counters. ASLLRP and NCSLGR annotation conventions stay separate.

After driver completes: inspect comparison_dev.json/status/checkpoint evidence;
finish preservation, reports and artifact inventories using expand08/tools/
finalize.py (created but NOT RUN yet). Need create/read-only final verifier and
REPRODUCE.ps1 before inventory. Finalizer must run after all workers and main
GPU sampler close. It must preserve existing reports and final tests;17TEST
remains reserved, old14 and isolated2620 results unchanged. No How2Sign
archive CRC rescan, no download/preprocessing repeat. Review unexecuted finalizer
for field names before running. Never claim main training or2,000 before logs.

Frozen source protection baseline: expand06/preservation_baseline.json includes
27 fresh content hashes and inherited4606 metadata checks; original03 inventory
1268content hashes and4prior evidence guards must pass final verification.
Prior How2Sign failed URL/error remain in source_status.json, with authentic
gloss unavailable. Cokely/Gallaudet403 not bypassed. No consented nonsign or
signer-independent streaming evaluation data supplied; those gates stay blocked.
