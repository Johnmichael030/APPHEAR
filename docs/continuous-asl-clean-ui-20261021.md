# Continuous ASL compact controls

Date: 21 October 2026, Asia/Shanghai. User-requested presentation change only.

The continuous recognition screen no longer shows the static explanation heading or paragraph. The edit dialog also omits that paragraph. Live status/error messages and the experimental mode labels remain visible.

The full-width Edit text button is replaced by a 48 dp circular pencil icon beside the live status. It has an “Edit recognized text” accessibility label and tooltip, focus support, disabled/enabled tint and a touch ripple. It calls the same optional edit handler with the existing empty/error/editing restrictions, revision protection, Save and Cancel behavior.

The model, gloss vocabulary, renderer, decoder, feature math, camera ownership, lifecycle safeguards and WORD/FINGERSPELL controls were not changed. This is not a recognition-accuracy improvement or model promotion.

Verification: 301 host tests passed, zero failures/errors/skips; debug and instrumentation APKs compiled. APK signature and version metadata pass. All 528 previous APK asset entries retain size/CRC, and continuous model/output asset bytes match exactly. All 220 snapshotted protected paths remain unchanged. No emulator/device visual test or phone installation was performed.

New debug APK: `AppHear-continuous-clean-ui-20261021.apk`, version `1.0-continuous-clean-ui-20261021`, 852,149,874 bytes (812.673 MiB), SHA256 `5c19ae04128108d1abaea26c896923e2e4711f9493bf601c152a2bd8eba61448`.

This public update contains the APK, scoped UI source and aggregate results. Detailed build logs, preservation records, machine paths, participant metadata and the user camera image are excluded from this update and retained locally. Git LFS stores the APK under `apk/`; the previous APK is retained. See DEVELOPER_HANDOVER.md and handover/APK.json for the current entry point.
