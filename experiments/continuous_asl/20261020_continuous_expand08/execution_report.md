# Executed continuous-ASL expansion and training

**Execution PASS; recognition readiness PARTIAL.**

Canonical corpus: 3122 verified unique records; 3025 training-eligible utterances across separate publisher label pools. Main fit consumed 2119 unique TRAIN utterances and used the exact original9DEV;17 reserved TEST utterances remain untouched.

| Run | TRAIN | Epochs / selected | DEV WER | S / D / I | Fit and reload seconds | Peak PyTorch MiB |
|---|---:|---:|---:|---:|---:|---:|
| Frozen expand03 reference |100|14 /4|85.96%|10/39/0|5.7732|68.08|
| Bounded expand09 pilot |906|20 /13|96.49%|16/39/0|140.5290|74.66|
| Main expand08 |2119|17 /2|100.00%|0/57/0|147.0251|1244.85|

Main checkpoint: `D:\apphearv3claude\AppHear-claude-s2-r3-eval-suite\AppHear-claude-s2-r3-eval-suite\experiments\continuous_asl\20261020_continuous_expand08\models\reference48\checkpoints\epoch_002.pt`
SHA-256: `f7a2b8bff5638671747352e2808f44fc2c60487a78d31a74cc49e978a46e7364`
TRAIN/DEV manifest SHA-256: `adebeb9e2f20179eb59b7f79805ea9226ad6c9c0594e6cba193c771947cdaf40`

The main candidate did not improve the frozen reference on identical9DEV. This is a limited57-reference-token cross-corpus diagnostic, not an architecture-only comparison or signer-independent benchmark. DEV OOV: 6/57. No experimental model is promoted.

Clean diagnostic v2: 2.94% TRAIN WER on8 eligible utterances; loss 17.4526→0.1152;75 optimizer updates; checkpoint reload matched. Numerical v1 is retained but excluded as a gate because one source utterance overlapped another publisher span.

906-pilot final-epoch full TRAIN WER: 51.70%; DEV-selected checkpoint TRAIN WER: 73.49%. Selection stayed on original9DEV. Optimization worked while generalization remained poor.

Main final epoch17 full TRAIN WER: 71.95%; DEV-selected epoch2 full TRAIN WER: 99.9943%. These separately saved, post-selection TRAIN diagnostics did not change checkpoint selection or score TEST. Final-epoch inference took 7.8051s in addition to the147.0251s fit/reload. The later checkpoint learned TRAIN while DEV selected an early blank-output checkpoint. The2739-label vocabulary includes1331 singleton labels and6/57DEV tokens unseen in TRAIN; cross-corpus label and distribution differences limit this comparison, without establishing a unique cause of regression.

Main median epoch: 8.5488s; observed TRAIN throughput including epoch validation/checkpoint work: 245.19 utterances/s and 11362.54 observations/s. CPU model-only p50/p95: 2.1601/2.6940ms per complete saved feature sequence,2threads,180measurements. These exclude perception/capture/UI and do not establish per-frame streaming latency, phone FPS or end-to-end latency.

Progress: 3025/2,000 eligible across documented separate pools; stretch5,000 shortfall 1975. One internally consistent main fit used 2119; its own5,000 shortfall is 2881. No alternative views, feature copies, isolated signs or English translations inflate counts.

Stage counters, exact bounds, vocabulary/normalization, environment/seed, full predictions, GPU samples, failures and hashes are in execution_summary.json, canonical manifest/quality, training_plan.v2.json and models/reference48. Per-epoch wall times include validation/checkpoint work. Command wall times are separate in logs; fit/reload timing also includes explicitly saved post-loop diagnostics where present. CUDA CTC backward warns that fixed seeds do not guarantee bitwise determinism.

The six ASLLRP overlapping utterances were rejected without trimming; four were stopped at metadata before completion and two cached features retained but excluded. Workers were reassigned before primary/tail catch-up, then balanced into disjoint modulo4 shards using measured clip runtimes.854 and then1115 completed feature hashes passed, with no incomplete output found. Interrupted unfinished work was resumed; completed extraction was reused. A pre-optimization smoke monitor error was preserved and fixed in a new smoke_v2 output; the906 main snapshot did not change.

How2Sign: existing archive verification reused; available inspected labels are English translations, so no fabricated gloss training. Prior failed URL `https://drive.google.com/uc?export=download&id=1oVZyTWhHShyqshC2kUrfWnBF8apIR7Z1`, errorHTTP_404 at2026-10-19T16:53:14.5183865Z remains preserved. Cokely official EAF URLs and Gallaudet OSF transcript endpoints returnedHTTP403; no bypass or unlabeled bulk download followed.

ASLLRP/NCSLGR media stay local under research/education-only terms; no redistribution or commercial deployment. Attribution: [ASLLRP resources](https://www.bu.edu/asllrp/ASL-SignBank-and-other-Resources.html), [publisher download documentation](https://www.bu.edu/asllrp/rpt24/asllrpr24.pdf), [terms](https://www.bu.edu/asllrp/dai-terms.html), [official DePaul ELAN conversion](https://asl.cs.depaul.edu/corpus/index.html). Literal source label conventions remain separate; CSV occurrence labels are not English translations.

Streaming qualification remains blocked without verified signer-disjoint evaluation, independently reviewed boundaries and consented nonsign recordings. Reuse the unchanged acceptance_protocol.md: paired WER improvement≥5points with signer-aware uncertainty; negative-exposure upper false-commit rate≤0.1/min; miss upper bound≤10%; eventF1≥0.90 and boundaryp95≤200ms; normal-phone deliveryp95≤1000ms and completed15FPS/5min. These are proposed criteria, not measured passes.

Original frozen results, checkpoints, production thresholds, Android application/GLB assets and prior reports were preserved. No APK was installed. Exact safe validation and completed-run verification commands are in REPRODUCE.ps1. The completed comparison is unchanged; comparison_provenance.json distinguishes its executed version from an unexecuted additive diagnostic edit.
