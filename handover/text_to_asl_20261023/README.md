# Text-to-ASL update — 23 October 2026

**647 word/expression entries; 764 explicit English input forms.**
This includes 215 entries beyond the previous 432 and 51 additional English aliases.
Aliases and arbitrary fingerspelling are not counted as additional lexical entries.

- [Complete numbered word list](../../WORD_LIST.md).
- [GLB files: direct paths and download instructions](GLB_FILES.md).
- [Machine-readable vocabulary and provenance](VOCABULARY_647.json).
- [All English inputs](ENGLISH_INPUTS_764.json).
- [Latest 647 preview APK](../../apk/AppHear-text-asl-647-preview-20261023.apk).
- [Exact build command](BUILD_COMMAND.txt), run inside `android/` after provisioning the Android SDK/JDK and dependencies.
- [Delivered file sizes and SHA-256](FILES.sha256).

## Included implementation

The `android/app/src/aslWebVocabulary`, `aslWordMotionFixes`, and `aslVocabulary647`
asset packages, their domain contracts/loaders, catalog hooks, Gradle flags,
alias mappings, tests and verification/conversion tools are updated in their normal
project paths. Existing files and older APKs are preserved.

Five word-motion substitutions (WHY, WHERE, YOU, YOUR/YOURS, TODAY) are retained.
FROM/WHAT source limitations and ARE/IS spelling behavior remain unchanged.
Default/release flags remain false; the supplied debug preview enables the new path.
Continuous-ASL recognition accuracy is not changed or claimed improved by this vocabulary update.

## Motion license and review

New motion: [3D-LEX ASL Animation Data](https://osf.io/mqzyk/), CC BY 4.0.
Author credit, license links, adaptation details and source hashes accompany every
asset package in `ATTRIBUTION.md` and `manifest.json`. No raw dataset archive,
NGT motion, source avatar mesh or source texture is included. AppHear's existing
avatar geometry/motions are retained; the new source license does not establish
rights to those older assets. Their unresolved provenance remains documented.

These are **UNREVIEWED animation previews**. Qualified ASL review and physical-phone
rendering/memory/latency acceptance are pending. This is not validated English-to-ASL
sentence translation. No participant recordings, private recording metadata,
recognition training data, credentials, caches or environments are part of this update.

## Completed engineering checks

- 313 unit tests passed; zero failures, errors or skips.
- All 534 previous APK asset files byte-identical; three new package files added.
- 678 named actions; all 463 previous actions preserved in the new resident model.
- Independent binary, rotation, timing, manifest and source-hash verification passed.
- APK v2 signature verification passed; APK is 933,228,481 bytes.
- APK SHA-256: `21e84bf37092750b760712728b94759e37fecb2590fa73928e90230a714d707a`.

These checks completed at 12:33:27 UTC+8 on 23 October. They are engineering checks,
not recognition accuracy or linguistic approval. No new build, training or installation
was performed for the upload.

## Download and validate

Large `.apk` and `.glb` files use Git LFS. Use the root README clone instructions;
GitHub ZIP downloads may contain pointers. From a complete checkout:

```powershell
git lfs pull
python -B .\android\tools\verify-web-asl-vocabulary.py
python -B .\android\tools\verify-word-motion-fixes.py
python -B .\android\tools\verify-vocabulary647.py
```

Engineering PASS; linguistic/phone readiness PARTIAL. Older handover inventories are
dated historical inventories; `FILES.sha256` describes this incremental delivery.
