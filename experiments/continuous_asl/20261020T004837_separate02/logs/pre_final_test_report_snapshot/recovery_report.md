# Recovery — PASS with explicit limitations

| Item | Status | Evidence |
|---|---|---|
| Exact repository / prior report | PASS | `D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite`; prior run `D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261019T155428Z_cslr01` exists. Report, continuation, configuration, checkpoints, scripts, tests and logs read. |
| Original preservation | PASS | All 2,932 original files retain SHA256 and historical modification time. |
| Completed prior artifacts | PASS | 401 prior artifact hashes verified, plus original manifest; no prior report/log/checkpoint overwritten. |
| Git identity | PARTIAL | No .git metadata; no current branch/commit can be asserted. File manifests are authoritative here. |
| Python / GPU | PASS | Existing Python 3.11.11 CUDA PyTorch reused; actual input/parameter/loss/backward/optimizer and finite checks passed. No environment/driver upgrade. |
| Storage | PASS | D: initially 178,989,453,312 bytes free; explicit 8 GiB run budget, 2 GiB archive cap, 64 MiB pilot extraction cap. New staging about 1.63 GB; retained regression copy about 1.57 GB. |
| RAM / VRAM | PASS | Memory sampled inside actual training workers, correcting the prior launcher-RSS limitation. Logs contain working set, peak working set, private commit, pagefile-backed peak and CUDA peak. |
| How2Sign | PASS for bounded acquisition | Existing realigned translations verified; current official archive and original annotation acquired, CRC checked; eight clips decoded/paired. Not full-dataset acquisition. |
| Other datasets | PASS/PARTIAL | Eligible ASL Citizen-derived features were found on D:, unlike the prior repository-only search; full raw provenance is not reconstructed from row-only NPZs. NCSLGR original RGB reused by hash. |
| Splits / labels | PASS/PARTIAL | A: 35 train / 5 val signer IDs, zero overlap and official split concordance. B: collection-disjoint, signer independence unverified. No TEST tensors loaded for model work. |
| Prior WER / ONNX | PASS | Recomputed 88% = 6S+16D+0I /25 references, 9 hypothesis tokens. Prior checkpoint/export hashes still match; no completed training resumed. |
| Original missing Keras source | BLOCKED | Frozen phase7 source checkpoint not found in relevant repository/data roots. No substitute manufactured. |
| Running work audit | PARTIAL | NVIDIA initially idle/0 MiB; own jobs ran sequentially and finished. No exhaustive initial CPU download-process inventory is asserted. |

Recovery mismatch list is empty. New phases write only `D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261020T004837_separate02` and new tools under `D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/tools/asl_research_v2`. Old artifacts and source models remain untouched. New root template training_config.yaml was copied by the run initializer but is **not** the configuration for these candidates; their actual configs are under experiment_A and experiment_B_plan.json.
