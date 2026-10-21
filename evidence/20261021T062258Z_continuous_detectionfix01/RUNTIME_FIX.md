# Continuous-ASL runtime fixes

Date: 2026-10-21. Scope: two software defects, not model retraining or a measured recognition-quality improvement.

## Changes

1. `ContinuousAslCore` now tracks explicit suspension. Selecting continuous ownership does not resume inference. `pause()` blocks subsequent observations/model calls, invalidates editing through a new revision, drops pending temporal state, and preserves displayed text and ERROR/FULL state. Explicit `resume()` starts fresh temporal context while preserving ERROR/FULL; Clear remains the explicit recovery action. Inactive ownership cannot be enabled by resume.
2. `RecognitionActivity` explicitly resumes on `onStart` and asynchronous model-load completion only while its lifecycle is STARTED and the selected owner is continuous. A model finishing loading after the screen stopped remains suspended. The existing serialized camera worker still owns Core calls.
3. A no-hand observation no longer forces `decoder.finish()` into the displayed sequence. It discards the pending run through the existing temporal reset and abstains. Already-settled text remains. Genuine short CTC peaks that end on another **observed** blank/nonblank remain supported, as do genuine blank-separated repeats.

No feature math, normalization, model, vocabulary, English dictionary, timing threshold, WORD/FINGERSPELL code, asset or camera configuration changed. The raw missing-hand/reset serving policy otherwise remains intact. Its broader difference from continuous missingness-containing training sequences remains a separate, unproven quality hypothesis.

## Synthetic BEFORE / AFTER regressions

These are engineering traces, not actual ASL signing or accuracy evidence. Baseline source and tests are preserved under `before/`. AFTER rows are assertions in the updated host tests; execution is delegated to the parent/root agent and was not run by this implementation subtask.

| Synthetic case | BEFORE behavior | AFTER expected regression |
|---|---|---|
| Token IX-1p at0/100ms; no hands110ms; IX-1p120ms; no hands220ms | Existing baseline test asserted `IX-1p IX-1p`, despite only three model calls and neither run reaching200ms | Empty text; neither absent-hand frame may publish a pending token. A later valid200ms sustained run may publish once. |
| Settled IX-1p, then pause, then three queued visible observations | `pause()` left LISTENING/active, allowing model inference and text updates | Zero additional model calls; identical paused snapshot. `select(true)` alone cannot resume. Explicit resume makes next model context length1. |
| Error, then pause/resume | Old pause converted ERROR to LISTENING, implicitly permitting retry | ERROR and message persist, model calls remain blocked until explicit Clear. |
| Full output, pause/resume | No explicit suspension prevented additional processing in other states | FULL/text persist; no model calls. Clear while paused does not itself resume inference. |
| Editing, then pause/resume and old dismiss/save callback | Old pause changed phase but did not suspend future inference | Old edit ticket is rejected; old dismiss cannot cancel resumed input; text persists and resumed context is fresh. |
| Visible short peaks IX-1p,blank,IX-1p,WANT,blank at100ms intervals | Valid CTC sequence `IX-1p IX-1p WANT` | Same sequence. This fix does not impose a universal minimum-duration sign filter. |

The no-hand correction can increase deletions when a genuine token is observed only briefly and disappears before an observed CTC transition. That tradeoff is explicit: absence of evidence must not itself finalize an unsettled prediction. No claim of improved WER, false-activation rate, unseen-signer accuracy or translation quality follows from these synthetic cases.

## Tests and changed surface

- `ContinuousAslCoreTest.kt`: existing14 tests updated to explicitly resume; eight new lifecycle/no-hand/CTC regressions, total22 continuous-core tests.
- `ContinuousAslCore.kt`: suspension/resume and absent-hand publication fix only.
- `RecognitionActivity.kt`: two lifecycle-safe resume call sites only; all recognition owner routes remain unchanged.
- No Gradle build, APK installation, camera capture or training executed by this subtask. Parent/root owns host testing, integrity checks, packaging and final acceptance report.
