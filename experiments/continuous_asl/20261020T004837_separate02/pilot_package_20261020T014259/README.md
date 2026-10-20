# Consented streaming pilot package — PARTIAL

Preparation only. **No real participant has consented, been recorded, or had identifiable data processed by this package. No training, inference, threshold tuning or checkpoint selection occurs.** Source/archive and frozen experiments are reused without repeating their runs.

Read `pilot_acceptance_report.md`, `recording_protocol.md`, `consent_and_handling.md`, then `manifest_schema.md`. `intake_template.json` is intentionally PENDING/empty. Complete real intake only in access-controlled storage outside this repository after the responsible human establishes permission. Never turn synthetic placeholders into alleged consent or actual glosses.

The validator checks permission attestations before artifact access; raw source-time joins; current350/378 feature schemas and existing numerical recipes; exact single-sign windows; authentic annotation provenance/joins; and signer aliases/history. A human must establish permissions/linguistic truth. The validator cannot authenticate them, decode/verify video truth, prove RGB-to-observation provenance or establish Python/Android end-to-end parity. A real compatibility pass is PARTIAL and does not authorize evaluation scoring, training or deployment.

Safe PowerShell commands (new dry-run/test outputs always get fresh directories):

```powershell
$pilot = 'D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020T004837_separate02\pilot_package_20261020T014259'
& "$pilot\Invoke-Pilot.ps1" -Phase Verify
& "$pilot\Invoke-Pilot.ps1" -Phase DryRun
& "$pilot\Invoke-Pilot.ps1" -Phase Tests
& "$pilot\Invoke-Pilot.ps1" -Phase Protocol
```

Verify hashes this package only; it does not repeat archive/GPU/frozen-experiment checks. DryRun uses only generated metadata/arrays from the existing synthetic unit fixture. Tests check rejection behavior. `SYN_TOKEN_A/B` are fake tokens; no synthetic numbers count as recognition or model-evaluation evidence. Tests may print an existing pyreadline3 console-destructor warning after passing; the saved JUnit XML/exit status distinguish this from test failures. The broader repository suite remains not green and was not rerun.

**After human permissions, actual consented capture and qualified annotation only**, validate private real intake using paths you have populated; do not execute these placeholder paths as an acquisition step:

```powershell
& "$pilot\Invoke-Pilot.ps1" -Phase ValidateReal -Stage first `
  -Manifest 'D:\ASL_PILOT_PRIVATE\intake_first.json' -DataRoot 'D:\ASL_PILOT_PRIVATE'
# After signer1 compatibility/review passes and signer2 identity/permissions are verified:
& "$pilot\Invoke-Pilot.ps1" -Phase ValidateReal -Stage second `
  -Manifest 'D:\ASL_PILOT_PRIVATE\intake_two_signers.json' -DataRoot 'D:\ASL_PILOT_PRIVATE'
```

Exit0 means artifact checks passed, not that training is ready; actual mode reports PARTIAL_REAL_COMPATIBILITY_ONLY. Exit2 means blocked/rejected, including missing/expired/withdrawn consent, unavailable fields/artifacts, invalid timestamps/joins or wrong schemas. `real_first_signer_validated` in the machine validator refers only to actual-mode artifact compatibility; full recording/annotation/runtime acceptance is separately decided in the acceptance report. No scoring script or training command is included. Baseline binding, producer provenance, live event windows, Android parity and normal-app clock/delivery evidence still require verification.

First obtain human-established consent/permissions for signer1, privately audit identity/proficiency, and collect only the short preflight. If its raw-time/features/review joins pass, expand to the planned pilot and then signer2. Both remain evaluation-only. Genuine train/validation/final signer separation and a verified evaluation protocol remain prerequisites for any later training proposal.
