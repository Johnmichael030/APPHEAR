# Baseline and error analysis

Date:2026-10-21 (Asia/Shanghai). PASS computational baseline; recognition PARTIAL.

This file was written BEFORE any new optimizer update. Source is frozen epoch12 SHA256 14d0e4669674265ea2be8a3161fa6699e4f813daf4b97b384863dce1daa038ff. Architecture:378D input,48 channels, four causal blocks with two kernel3 convolutions each, dilations1/2/4/8, receptive field61;2289 logits. CTC blank0/UNK1; collapse consecutive IDs before removing blanks. Frozen normalization is embedded in model forward and not refitted.

All1500TRAIN and80DEV feature hashes/schema/chronology/embedded literal targets/CTC lengths pass. TRAIN has3publisher signer IDs;80DEV uses two known signers in disjoint recording families. The reserved17TEST set was not accessed. Original manifests are unchanged. Exact new manifest hashes are in A/TRAINING_CONFIGURATION.json.

Full-sequence greedy DEV predictions reproduce the original saved checkpoint predictions exactly:WER95.670103093%,S136/D328/I0,485references,0/80exact. Current streaming-policy replay reproducesS176/D288/I0,WER95.6701031%,0/80exact. These protocols are distinct even where aggregate WER happens to match. Saved logits can support cheap beam experiments without another model pass. Logit confidence summaries and publisher timing metadata are saved; boundaries are not qualified physical-sign ground truth.

The existing128TRAIN monitor is retained. Historical fullTRAIN WER88.4005% is reused from verified selected_train1500.json; do not claim a fresh fullTRAIN baseline evaluation. The192memorization control, from a different checkpoint, still has10.5874%TRAIN vs101.0309%developmentWER. No new accuracy claim follows from pipeline validity.

The current vocabulary has2287literal labels,1182singleton-supported and1808with<5distinctTRAIN utterances. Two label pools remain separate. Local named clips are not admitted by filename; permissions/annotation/split provenance must be verified. New classes will be append-only and need valid labeled evidence.

CUDA is available onNVIDIA GeForce RTX 3050 Laptop GPU,VRAM4.000GiB;Torch2.8.0+cu128. No dependency/global installation was changed. A is a newly authorized warm-start fine-tune, not a repeat of rejected framing/history changes. B remains independently gated on valid new-class data. No production/app/asset/checkpoint/TEST modification is authorized.
