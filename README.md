# Latest Text-to-ASL files — 23 October 2026

**647 word/expression entries · 764 English input forms.**

- [Full new word list — all 647 entries](WORD_LIST.md).
- [Today’s Text-to-ASL update, source map, attribution and validation](handover/text_to_asl_20261023/README.md).
- [Latest 647-entry preview APK](apk/AppHear-text-asl-647-preview-20261023.apk) — Git LFS, about 890 MiB; download the raw file.

New animations remain unreviewed; qualified ASL and phone acceptance are pending.
The earlier continuous-ASL handover below is preserved as dated history.

---

# AppHear developer handover

**Latest APK and continuous-ASL files â€” 21 October 2026, Asia/Shanghai.**

- [Latest APK](apk/AppHear-continuous-clean-ui-20261021.apk) â€” open the file and use **Download raw file** (812.7 MiB).
- [UI update report](docs/continuous-asl-clean-ui-20261021.md) — explanation removed and edit action changed to a pencil icon.
- [Developer handover](DEVELOPER_HANDOVER.md) â€” setup, source map, build and known issues.
- [Continuous-ASL Android runtime](android/app/src/main/java/com/example/apphear/continuousasl/).
- [Continuous training/evaluation tools](tools/continuous_asl/).
- [Runtime model and vocabulary](packages/continuous_asl_research_01/).
- [Latest experimental fine-tuning report](experiments/continuous_asl/20261021T080638Z_finetune_vocab01/MISSION_REPORT.md).
- [Exact delivered-file inventory](handover/CONTENT_MANIFEST.json).

APK, models, checkpoints and large binary assets are stored directly at their project paths through **Git LFS**. Use a short checkout path on Windows:

```powershell
git lfs install
git clone https://github.com/Johnmichael030/APPHEAR.git C:\src\APPHEAR
Set-Location C:\src\APPHEAR
git lfs pull
git lfs fsck
python .\tools\handover\verify_checkout.py
```

GitHub's ordinary ZIP download may contain LFS pointers. Clone with Git LFS to obtain actual assets. The repository is public, as explicitly requested. Reading and cloning do not require collaborator access. This UI update excludes detailed local logs and private metadata.

The APK retains the frozen epoch-12 recognizer. Latest research A improved full-sequence DEV WER from 95.67% to 94.64%, but streaming replay stays 95.67% and exact sequences remain 0/80. It was **not promoted**. Recognition and the finite English renderer remain experimental.

Restricted dataset payloads, participant recordings, credentials and environments are excluded. Earlier [encrypted backup and research-upload information](handover/PREVIOUS_REPOSITORY_README.md) is preserved. The older [project README](handover/PROJECT_README.md) is historical; start with DEVELOPER_HANDOVER.md.
