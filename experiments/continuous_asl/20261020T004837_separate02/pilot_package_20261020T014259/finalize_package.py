"""Finalize only this NEW pilot package; reuse completed checks, never train."""
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import validate_pilot as v

P = v.PACKAGE
RUN = P.parent

def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')

def main():
    for name in ('pilot_acceptance.json', 'pilot_acceptance_report.md', 'preservation.json', 'artifact_hashes.json'):
        if (P / name).exists():
            raise FileExistsError('FINALIZATION_MUST_NOT_OVERWRITE_' + name)
    for name in ('validate_pilot.py', 'synthetic_fixture.py', 'test_validator.py', 'finalize_package.py'):
        ast.parse((P / name).read_text(encoding='utf-8'))
    v.read_json(P / 'manifest.schema.json')
    first = v.read_json(P / 'checks/final_first_synthetic.json')
    second = v.read_json(P / 'checks/final_second_synthetic.json')
    assert first['status'] == second['status'] == 'PASS_SYNTHETIC'
    assert not first['real_first_signer_validated'] and not second['model_evaluation_evidence']
    blocked = v.read_json(P / 'checks/real_intake_blocked.json')
    assert blocked['reason'] == 'PERMISSION_NOT_ESTABLISHED'
    junit = P / 'runs/tests_20261020T015459_684/tests.xml'
    cases = list(ET.parse(junit).getroot().iter('testcase'))
    assert len(cases) == 47 and all(case.find('failure') is None and case.find('error') is None for case in cases)

    protected = json.loads((RUN / 'protected_sources.json').read_text(encoding='utf-8-sig'))
    relative = set(v.CONTRACT_FILES) | {
        'tests/test_perception_vnext_recognition_schema_v2.py', 'context/feature-schema.md',
        'context/progress-tracker.md', 'AGENTS.md', 'android/app/src/main/assets/model_manifest.json'}
    guards = []
    for item in protected:
        path = Path(item['path'])
        if path.is_relative_to(v.ROOT) and path.relative_to(v.ROOT).as_posix() in relative:
            ok = v.sha(path) == item['sha256'] and path.stat().st_mtime_ns == item['mtime_ns']
            guards.append({'path': str(path), 'hash_and_mtime_match_prior_protection': ok})
    assert len(guards) == len(relative) and all(row['hash_and_mtime_match_prior_protection'] for row in guards)
    old_manifest = RUN / 'artifact_manifest.final.json'
    old_continuation = RUN / 'continuation_20261020T013509/artifact_hashes.json'
    assert v.sha(old_manifest) == '1081281f0a6d28426d4804d202300381549d4374d25a7bb204050671f8a2d8b4'
    assert v.sha(old_continuation) == '28d4ebd8b0196bba0603839c60cac4f18a08386fef644a0e71ec3c2de3f5d882'
    old = v.read_json(old_manifest)
    frozen_names = {'final_test_freeze.json', 'final_test_results.json', 'final_test_report.md',
                    'final_status.md', 'experiment_A/warmstart/checkpoints/best.json',
                    'experiment_A/warmstart_repeat/checkpoints/best.json', 'experiment_B/checkpoints/best.json'}
    frozen_guards = []
    for item in old['artifacts']:
        if item['scope'] == 'run' and item['path'] in frozen_names:
            frozen_guards.append({'path': item['path'], 'hash_matches': v.sha(RUN / item['path']) == item['sha256']})
    assert len(frozen_guards) == len(frozen_names) and all(row['hash_matches'] for row in frozen_guards)
    preservation = {'status': 'PASS_TARGETED_ADDITIVE_GUARDS', 'source_guards': guards,
                    'frozen_report_and_selection_guards': frozen_guards,
                    'old_final_manifest_unchanged': True, 'old_continuation_manifest_unchanged': True,
                    'scope': 'Only new pilot directory written. Prior full preservation evidence reused; no repeat archive/model/test-payload verification or inference.',
                    'prior_full_preservation_reference': '../continuation_20261020T013509/evidence/preservation.json'}
    write(P / 'preservation.json', preservation)
    result = {
        'schema': 'apphear.streaming_pilot.acceptance.v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'overall': 'PARTIAL', 'package_preparation': 'PASS', 'targeted_tests_passed': len(cases),
        'test_xml': str(junit.relative_to(P)), 'first_synthetic': first, 'second_synthetic': second,
        'permission_guard': blocked, 'real_first_signer_recording_pipeline': 'BLOCKED_NOT_COLLECTED_OR_ANNOTATED',
        'real_second_signer_independence': 'BLOCKED_IDENTITY_CONSENT_AND_DATA_UNVERIFIED',
        'authentic_gloss_ground_truth': 'NOT_CREATED_OR_VERIFIED', 'current_feature_schema_dry_run': 'PASS_SYNTHETIC_ONLY',
        'current_model_vocabulary_and_producer_binding': 'PENDING_REAL_REVIEW',
        'runtime_event_selection_android_parity_and_end_to_end_latency': 'NOT_MEASURED',
        'real_recordings_processed': 0, 'consent_documents_generated_or_signed': 0,
        'synthetic_is_model_evaluation': False, 'training_allowed': False,
        'preservation': preservation['status'], 'broader_suite': 'NOT_RERUN_PRIOR_SUITE_NOT_GREEN',
        'next_action': 'Establish signer1 human permissions and identity/proficiency audit before a short consented preflight; then validate source/features/review and independently admit signer2.'}
    write(P / 'pilot_acceptance.json', result)
    report = '''# Pilot acceptance — PARTIAL; actual collection/training BLOCKED

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
'''
    with (P / 'pilot_acceptance_report.md').open('x', encoding='utf-8') as stream:
        stream.write(report)
    files = []
    for path in sorted(P.rglob('*')):
        if path.is_file() and 't' not in path.relative_to(P).parts and '__pycache__' not in path.parts and path.name != 'artifact_hashes.json':
            files.append({'path': path.relative_to(P).as_posix(), 'bytes': path.stat().st_size, 'sha256': v.sha(path)})
    write(P / 'artifact_hashes.json', {'schema': 'apphear.streaming_pilot.artifacts.v1', 'files': files,
                                      'excluded': 'This hash manifest itself and disposable synthetic unit-test t/ directories; future runs are not implicitly included.'})
    print(json.dumps({'overall': result['overall'], 'tests_passed': len(cases), 'new_artifacts_hashed': len(files),
                      'real_first_signer': 'BLOCKED', 'training': 'BLOCKED', 'preservation': preservation['status']}))

if __name__ == '__main__':
    main()
