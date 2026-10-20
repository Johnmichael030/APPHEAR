# Regression and release gates

84 existing tests passed; one existing test fails because its frozen Keras source artifact is missing. Five new CTC engineering checks pass; six hazard regressions pass; nine archive safety tests pass; 15 downloader policy tests pass. The earlier test-harness argument error is retained in logs/regression_hazards.txt; corrected hazards_02 is six passes. Real HTTP range resume passed and a cached sample was reused after hash verification. Completed-run -Resume was checked to start no further epochs and preserve training_log.csv. PowerShell AST and Python syntax checks are recorded separately. Full Android instrumentation and broad write-producing training/eval suites were not run blindly.

| Gate / finding | Actual status |
|---|---|
| A long ACTIVE event loses start | Reproduced with existing Python decoder/ring and the Kotlin materializer precondition: start=0, retained first=13, 13 evictions. No Kotlin/device execution claimed. |
| A recovery policy | Isolated event_guard.py returns EXPIRED_EVENT and skips recognition if any required index is absent. Never synthesizes/truncates a sign. Age-boundary, count eviction, missing commit and bad ordering tests pass. Not wired into Android; production issue remains. |
| B unconditional top1 commitment | Source verified in PhoneProductionSession.consume: SentenceBuilder adds top1; negative calibration is blocked. No arbitrary threshold patch. |
| C throughput | Component + CPU RGB-to-gloss replay measured; 10 Hz sampling / no Android means 15 FPS product gate not passed. |
| D recognition flag | RECOGNITION_ENABLED declared but unused by main runtime; source finding persists. Enabled/disabled runtime correction not implemented or tested. Existing configuration selection is distinct. |
| E merge overlap | Existing merge overwrites its source train NPZ on a disposable new synthetic fixture. New safety helper rejects identical/nested source/output paths. Original merge source remains unchanged. |
| Leakage | Collection/URL/exact-content checks pass; signer leakage unresolved. |
| Export CPU parity | PASS; no decoded differences on 37 clips. |
| Non-sign + boundary evaluation | BLOCKED, user confirmed no extra data. |
| Existing word/fingerspelling | Relevant unit checks pass; full live-device behavior is not established by unit tests. |
| Assets / manifests / avatars | All 2,932 original files and 1,569,935,151 bytes match hashes and modification times, including production models, Original/FemaleASL GLBs and manifests. |
| Production / release | BLOCKED. No runtime integration, enforced new Android flag, signed-phrase benchmark or authorized install. |

Original preservation list SHA256: `7511ec95167574e8727e858e4ec04adad53bd43fdcb7cb101c1e1cc038181ce1`. The first timestamp comparison incorrectly compared a deserialized DateTime to a string and flagged all times despite matching hashes; corrected UTC tick comparison confirms zero changes. Faulty diagnostic logs are retained. No original file was repaired or touched to make the verification pass.
