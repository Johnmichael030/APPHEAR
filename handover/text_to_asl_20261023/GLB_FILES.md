# Text-to-ASL GLB animation files

These files are already stored in this repository under the Android asset packages,
using **Git LFS**. They are not inside this handover directory. Each `avatar.glb`
contains the avatar and multiple named animation actions.

| Package | GLB file | Actual size | Role |
|---|---|---:|---|
| 647-entry preview | [avatar.glb](../../android/app/src/aslVocabulary647/assets/asl_animations/vocabulary_647/avatar.glb) | 79,906,716 bytes | Complete current 647-entry resident bundle; 678 named actions including alphabet, neutral and preserved previous actions |
| 432-entry preview | [avatar.glb](../../android/app/src/aslWebVocabulary/assets/asl_animations/web_vocabulary_200/avatar.glb) | 54,902,340 bytes | Earlier 200-entry motion expansion retained for build dependencies and rollback |
| Five word-motion fixes | [avatar.glb](../../android/app/src/aslWordMotionFixes/assets/asl_animations/word_motion_fixes/avatar.glb) | 55,421,552 bytes | WHY, WHERE, YOU, YOUR/YOURS and TODAY substitutions; also included in the complete 647 bundle |

Open a file link and select **Download raw file**. GitHub does not render a signing
preview of these GLBs. A small text file beginning `version https://git-lfs.github.com/spec/v1`
is an LFS pointer, not the actual animation. Use Git LFS or the raw download button.
The 647 entries are named actions/catalog records inside the resident bundle, not
647 individually named GLB files.

For a clone that already exists:

```powershell
git pull --ff-only
git lfs install
git lfs pull --include="android/app/src/aslVocabulary647/**,android/app/src/aslWebVocabulary/**,android/app/src/aslWordMotionFixes/**"
```

The latest 647-entry GLB's SHA-256 is:
`7840f979713d878f273a71975c38e1d12dbf3856151af531cbed8daf70fd0024`.

The matching manifest, source identities, attribution and CC BY 4.0 motion notice
are beside each GLB. Qualified ASL and phone review remain pending; existing avatar
rights remain separate from the imported motion license. See [full update report](README.md).
