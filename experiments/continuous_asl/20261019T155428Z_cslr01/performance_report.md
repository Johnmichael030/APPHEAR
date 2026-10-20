# Performance — measured scopes

| Measurement | p50 | p95 | Scope |
|---|---:|---:|---|
| Existing Candidate01 pose + hand extraction | 53.281 ms | 62.685 ms | 944 selected CPU frames, all 37 clips |
| Existing S2 TFLite | 0.399 ms | 0.643 ms | CPU, four threads, invocation only |
| New ONNX CTC | 0.348 ms | 0.618 ms | CPU, two threads, 944 real prefixes |
| New PyTorch CTC CPU | 2.102 ms | 3.764 ms | two threads, invocation only |
| New PyTorch CTC CUDA | 3.410 ms | 8.904 ms | preplaced input + synchronized invocation, transfer excluded |
| CPU RGB-to-gloss replay | 56.096 ms | 58.652 ms | one real 88-frame clip / 30 observations; includes features and new model |

One-clip CPU replay wall time excluding startup: 1.729 s; processing capacity 17.35 selected observations/s. Actual configured observation cadence is capped at 10 Hz. This does **not** pass the 15 FPS product gate. It is saved-video replay without live camera, rendering, SentenceBuilder, lifecycle or Android scheduling. No Android end-to-end FPS, signed-phrase latency or device thermal stability was measured.

Replay startup (perception + ORT) 8.531 s; ONNX-only session construction 27.016 ms (single process/cold sample, OS file caches unspecified). New ONNX 336,478 bytes; checkpoint 980,299 bytes; 76,551 trainable parameters. Layer/shape/prefix checks support an adapter investigation, not mobile runtime compatibility certification. For this small workload, CPU ORT was faster than synchronized PyTorch CUDA; that does not imply GPU training was unnecessary.

Training peak allocated VRAM 70,403,584 bytes; peak reserved 71,303,168. Actual CUDA inputs/logits/loss/gradients and finite memory-probe results are retained. Median epoch loop time 0.155 s, sum 2.432 s; excludes environment/model initialization and includes validation. PowerShell sampled NVIDIA utilization/memory/temperature every two seconds; the job is too short for sustained-load or thermal conclusions.

Actual CPU replay worker PID 19320: peak working set 366,141,440 bytes; private commit 1,405,800,448; peak pagefile-backed usage 1,418,661,888. Peak working set is resident RAM, not total commit. **Training peak RAM was not captured reliably**: PowerShell observed the Windows venv launcher, whose small RSS must not be interpreted as the CUDA worker's memory. This limitation remains explicit rather than triggering another training run for a missing metric.

Main measured bottleneck: landmark extraction, with substantial hand missingness. Recommended next performance experiment is a device-measured cadence/coverage study while retaining complete feature blocks, after a valid accuracy/rejection benchmark exists. No heavier RGB backbone or default model replacement is justified.
