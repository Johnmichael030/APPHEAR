"""One-time finalization; refuses to overwrite validation or checksum evidence."""
import fine_tune as ft
from verify_completed import validate

result=validate(require_seal=False)
ft.save(ft.RUN/'VALIDATION.json',dict(**result,recognition_status='PARTIAL_NOT_PROMOTABLE',vocabulary_expansion_status='BLOCKED_ACTUAL_RECORDS_NOT_ESTABLISHED',baseline_before_fit=True,baseline_predictions_reproduced=True,decoder_probe_executed_after_A=True,physical_boundaries_and_no_sign_and_unseen_signer_gates='BLOCKED',head_extension_tests='SYNTHETIC_ENGINEERING_ONLY',runtime_tests='Reused prior301passinghosttests; unchanged runtime source/assets verified, not rerun',new_B_data_training='NOT_EXECUTED; future driver not validated on real expanded data',checkpoint_header_and_resume_state_checks='PASS_all10completedepochs'))
checkpoints=sorted((ft.RUN/'A').rglob('*.pt'))
with (ft.RUN/'checkpoint_hashes.sha256').open('x',encoding='utf-8') as f:
 for p in checkpoints:f.write(ft.sha(p)+'  '+p.relative_to(ft.RUN).as_posix()+'\n')
paths=sorted(p for p in ft.RUN.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt')
with (ft.RUN/'SHA256SUMS.txt').open('x',encoding='utf-8') as f:
 for p in paths:f.write(ft.sha(p)+'  '+p.relative_to(ft.RUN).as_posix()+'\n')
print('SEALED',len(paths),'artifacts;',len(checkpoints),'real experimental checkpoints. No TEST/production mutation.')
