# Independent training-control review

Scope: read-only source review of `run_mission.py`, the historical source-transfer `train.py`/`common.py`/configuration, and the shared CTC implementation. This review did not run inference, fitting, dataset acquisition or reserved TEST access. The current experiment script was not changed.

## PASS — accumulation and training math

The new `fit_epoch` constructs accumulation groups before backward and divides each microbatch loss by the group's actual microbatch count. With 1,500 examples, batch size 4 and accumulation 2, there are 375 microbatches: 187 groups of two and one singleton. The final singleton correctly uses divisor 1 and receives its optimizer step. Gradient clipping follows accumulation and precedes the optimizer step.

All microbatches in a group must use the same preplanned denominator. Dividing by the number of microbatches remaining at each backward call would incorrectly weight the second microbatch.

The existing model is causal and has no temporal downsampling. The checked CTC feasibility condition is therefore input length >= target length + the number of adjacent repeated targets. Explicitly rejecting target IDs below 2 prevents silent unknown/blank target mapping by the shared batch helper. The existing greedy decoder correctly collapses consecutive repeated IDs and removes blank 0; UNK predictions remain scored as errors.

Fresh model/optimizer creation and seed resetting separate the selected-checkpoint optimizer smoke, the TRAIN-only subset diagnostic, and the fresh full candidate. The source checkpoint is preserved. Checkpoint mean/std equality assertions protect against normalization buffers being overwritten with incompatible values when loading a model state.

## PARTIAL — comparison equivalence

The prior and new candidate use identical frozen TRAIN and primary held-out identities, literal label mapping, saved vocabulary, normalization, architecture, optimizer hyperparameters, greedy decoder and WER conventions. The intended changed modeling/training variable is accumulation-tail scaling.

Execution is not identical: the new candidate scores full TRAIN each epoch, while the historical run primarily used a fixed 128-example TRAIN monitor at evaluation intervals. Additional scoring changes epoch wall time and may change how many epochs fit inside a wall budget. Report modeling-variable equivalence separately from execution-time equivalence. Do not attribute any result to the tail correction with causal certainty from one stochastic run.

The code's collapse gate uses the fixed TRAIN monitor plus primary held-out predictions. Label this `TRAIN_monitor128`, even though full TRAIN1500 metrics are also saved. It is not a collapse gate calculated from every TRAIN prediction.

The shared seed helper enables deterministic algorithms with `warn_only=True`. Fixed seeds do not establish bitwise CUDA reproducibility.

## PARTIAL — timing and budget semantics

The configured candidate 180-second deadline is checked before accumulation groups. Full TRAIN/held-out evaluation, checkpoint serialization/hashing, and final selected-checkpoint reload/scoring are not interrupted by the deadline. Treat it as a soft optimization-loop stop-request budget, not a strict total experiment-runtime cap.

The reported `training_loop_seconds` is computed after final checkpoint reload and selected full TRAIN/held-out scoring, so it includes finalization work. Report that field as candidate-stage elapsed time or explicitly disclose the included operations. Epoch fit time and total epoch time are separately measured. The epoch-1 projection is an estimate, not observed 20-epoch runtime; evaluation schedules differ across epochs.

On a subset failure or deadline stop, the last saved subset metrics may come from the last evaluation interval while the checkpoint contains later complete or partial-epoch updates. State the metrics' evaluated epoch separately from checkpoint/update state; do not imply they are automatically equivalent.

## PARTIAL — continuation limitations

A completed epoch checkpoint contains model/optimizer state, RNG state, selection/patience state and elapsed time. An interrupted partial epoch is not checkpointed by the new loop. Continue from the last completed epoch and disclose that partial work is discarded and would be repeated.

The current resume branch checks a configuration hash but does not authenticate the candidate checkpoint against its previously recorded SHA-256 before loading. A continuation must verify that hash first.

The final integrity writer exclusively creates `input_integrity.json`. Once that file exists, a later invocation of the same script will raise a file-exists error at shutdown. A separate continuation wrapper should preserve the prior integrity result and write uniquely named continuation artifacts.

Saved checkpoint elapsed time excludes work after that checkpoint, including a later timed-out partial epoch. A resumed run must account for that already-spent time when making cumulative budget claims. New output paths must remain exclusive and prior files must stay unchanged.

No automatic second fit is authorized by this review. Any continuation should preserve the frozen script, verify checkpoint/input hashes, and use a separate explicit continuation script rather than editing the running artifact.
