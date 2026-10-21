# Authorized implementation scope

The latest user explicitly requested automatic continuous-ASL output, gloss-to-English and optional editing, followed by an updated APK. This supersedes the earlier mandatory-review UI choice. Reuse the frozen experimental recognizer; do not retrain or promote it.

1. Preserve the existing APK, scoped sources, model/checkpoint hashes and assets.
2. Add a separate display policy and conservative offline translation stage. Keep unsupported labels explicit. Preserve feature math and exclusive recognition ownership.
3. Display text automatically, allow optional revision-guarded editing, and make Hear operate without a forced editing dialog. No automatic history or TTS.
4. Test temporal and edit failure modes and actual language assets; build and inspect APK; verify preservation.
5. Deliver an immutable APK and report limited translation coverage and pending device/recognition evaluation.

Existing local architect/imprint/review/remember workflow applied under the user's explicit implementation authorization. No additional approval or production promotion is implied. No new third-party dependency or external service was needed.
