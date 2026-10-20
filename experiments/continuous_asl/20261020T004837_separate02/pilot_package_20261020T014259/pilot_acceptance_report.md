# Pilot acceptance — PARTIAL; actual collection/training BLOCKED

Prepared on2026-10-20 Asia/Shanghai after running the existing safe CollectionPlan phase. No archive verification, GPU training, frozen experiment, held-out inference or production/Android action was repeated. This package is confined to its new directory.

| Gate | Decision and evidence |
|---|---|
| Recording/annotation/consent preparation | PASS: diagnostic sequences,10 natural topics,2 narrative prompts, conditions/session metadata and consent/data-handling checklist supplied |
| Strict validator | PASS software checks: schemas350/378/raw.v1, source times/indices, interval/sequence joins/bounds, single-sign windows, numerical recipes and identity/permission gates |
| First-signer dry-run | PASS_SYNTHETIC:1 synthetic identity/recording; no real consent, gloss truth or model result |
| Second-signer dry-run | PASS_SYNTHETIC:2 distinct synthetic registry keys; not evidence of real participant independence |
| Targeted tests |47passed,0failures,0errors; includes permission-before-artifact-read, expired consent, signer aliases, duplicate/missing/nonfinite timestamps, wrong schemas, English labels, skew, missing joins and invalid windows |
| Actual first-signer recording/annotation pipeline | BLOCKED: no participant permissions, real recording, real source/frame index, qualified annotation or producer provenance was supplied; its validity has NOT been established |
| Actual second signer | BLOCKED: independent identity/aliases/history, qualification, consent and the same capture/annotation joins must be verified after first-signer preflight |
| Current automatic event selection, Android parity and normal-app latency | NOT_MEASURED: reference-boundary350 windows are diagnostic; they cannot establish end-to-end pipeline behavior |
| Later training/data split/evaluation | BLOCKED: both pilot roles are evaluation-only; no verified new FIT/calibration/final corpus or scored protocol exists |
| Preservation | PASS targeted guards: existing contract/fixture/context/selection/report hashes and timestamps checked; prior full preservation evidence reused; old manifests unchanged |

The synthetic geometry comes from the existing recognition-v2 unit fixture. Its media files are explicit NOT_VIDEO byte markers, tokens are SYN_TOKEN_A/B, and permission fields say SYNTHETIC_NOT_CONSENT. Neither0.2seconds of synthetic negative metadata nor its schema checks count as reviewed exposure, WER, false activation, missed sign, boundary or latency evidence. The first/second outputs and hashes are retained. No additional data or consent was invented.

The real intake template was run only as deidentified empty metadata. It returned exit2/PERMISSION_NOT_ESTABLISHED before data-root/artifact access. No consent document was read/generated/signed and no participant media was processed. Human attestations must be truthful; this validator cannot authenticate identity, legal authority or linguistic correctness. Even a later actual-mode compatibility pass remains PARTIAL until the real provenance/runtime/qualification gates pass.

Both pilot roles stay evaluation-only. Verify signer2 against signer1 and all covered historical roles using a private identity registry, not just pseudonyms. Repeat the same core conditions, source-clock checks,350/378 checks and independent blinded expert review; retain per-signer failures. Pilot1 must not become threshold tuning followed by an allegedly untouched pilot2 test. The two-signer pilot cannot establish population generalization or authorize new training. Current baseline/vocabulary/model/producer identities and normal-app logging availability still require explicit verification.

The first PowerShell Tests wrapper invocation exited4 because its JUnit path argument split incorrectly; it was corrected and the failed log retained. Final47-test XML is authoritative. An existing pyreadline3 console-destructor warning is retained in test logs after successful completion; no dependency/environment was changed. Earlier broader suite remains810passed/130failed/54errors/10skipped and was not rerun.

Files: recording_protocol.md, consent_and_handling.md, manifest_schema.md/manifest.schema.json, intake_template.json, validate_pilot.py, synthetic_fixture.py, test_validator.py, Invoke-Pilot.ps1, review.md, pilot_acceptance.json, preservation.json, checks/, runs/ and artifact_hashes.json. The latter lists deliverable SHA-256 values, excluding itself and disposable synthetic unit-test t/ directories. Exact commands are in README.md. Verify and optional fresh synthetic DryRun/Tests never train or acquire data.

**Next action:** the responsible human establishes signer1 consent/applicable permissions and private identity/proficiency audit, then captures only the short preflight. No identifiable data action should occur beforehand. There is no next training command.
