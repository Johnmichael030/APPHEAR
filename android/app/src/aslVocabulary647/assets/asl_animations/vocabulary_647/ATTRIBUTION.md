# 3D-LEX ASL motion attribution

Source: Oline Ranum, Gomèr Otterspeer, Jari I. Andersen, Robert G. Belleman,
and Floris Roelofsen (2024). 3D-LEX v1.0: 3D Lexicons for American Sign
Language and Sign Language of the Netherlands. Only the ASL Animation Data
component is used here; no NGT animations are included.

Dataset: https://osf.io/cv276/
ASL animation component: https://osf.io/mqzyk/
Paper: https://aclanthology.org/2024.signlang-1.33/
Dataset licence: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
Legal terms: https://creativecommons.org/licenses/by/4.0/legalcode
License record: https://api.osf.io/v2/licenses/563c1cf88c5e4a3877f9e96a/
Official release licence index: https://www.sign-lang.uni-hamburg.de/lrec/data/3dlexv10.html

Changes: 200 source motions retargeted to AppHear's pre-existing human
skeleton at 24 Hz, rotation only, with 0.2-second neutral entry and exit.
Fixed bone-local axes are calibrated from median source offsets. Source
rotation tracks are transferred with fixed target bone lengths. Mechanically
invalid short reset suffixes are excluded with exact source ranges recorded
per entry in manifest.json; interior corruption rejects the entire candidate.
Source archives are unchanged. Abrupt rotations and excessive directional
error reject candidates. Source meshes, textures and animated scale are not copied.
All pre-existing avatar geometry and 258 actions are retained unchanged.
Per-motion source names and SHA-256 identities are in manifest.json.

The CC BY licence covers the imported 3D-LEX motion. It does not establish
rights to AppHear's pre-existing avatar or motions; those remain separately
documented/unresolved. No endorsement by the authors is implied.

UNREVIEWED PREVIEWS: source lexical labels and retargeted signing have not
received qualified ASL review. Facial blendshape grammar is unavailable.
These entries are not validated sentence translations or linguistic ground truth.

This package substitutes only WHY, YOU, YOUR/YOURS, and TODAY motion previews.
No new lexical entries or aliases are added. All previous 458 actions remain unchanged.
The new motions remain UNREVIEWED; no qualified ASL or phone approval is implied.

WHERE is also substituted. Its final source calibration-reset frame at 4.833333492s is excluded; retained through 4.791666508s. This is mechanical export cleanup, not a verified linguistic boundary. Source archive unchanged.

Authorized 647-entry preview: 215 additional source motions and 51 English aliases from the same verified ASL release, reusing frozen prior curation without reprocessing. All previous463 actions and five word substitutions are preserved. Aliases are not counted as additional signs. This remains an unreviewed word/expression preview, not qualified sentence translation.
