# Fine-tuning plan

A: restore epoch12 model and optimizer moments; fixed LR1e-4 (ten times below parent), unchanged1500TRAIN/80DEV, frozen vocabulary/normalization, batch4/accumulation2, FP32, clip5, at most12epochs/180seconds. One disposable configured optimizer-step smoke precedes fitting. Save every completed epoch and RNG state; resume only the exact config/manifest. Early-stop by predeclared DEV patience/collapse rules. Report fullTRAIN and current streaming replay only for selected checkpoint.

B: append verified new classes only after permissions, target/feature/timing and family guards pass. Preserve all2289old IDs and parameters at initialization. Mix genuine old/new records; no duplicates counted as new evidence. Block rather than invent labels/reviewer decisions.

C: only if A or B passes the frozen promising screen, compare one retention/imbalance variation; no simultaneous GPU runs or sweep. Decoder comparisons use same cached logits, no English language prior. No promotion without independent/streaming acceptance.
