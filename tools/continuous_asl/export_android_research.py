"""Export the frozen full-task checkpoint for opt-in Android draft recognition.

No fit, dataset acquisition, protected TEST access, or checkpoint selection.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools' / 'continuous_asl'))
from ctc_experiment import CausalCTC

RUN = ROOT / 'experiments/continuous_asl/20261020T143609Z_learning_diagnostic01'
PACKAGE = ROOT / 'packages/continuous_asl_research_01'
EVIDENCE = ROOT / 'evidence/continuous_asl_android_20261021'
CHECKPOINT_SHA = '14d0e4669674265ea2be8a3161fa6699e4f813daf4b97b384863dce1daa038ff'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    assert not PACKAGE.exists(), 'Refuse to overwrite an exported model package'
    # Recoverable pre-change source snapshot; existing assets are hash-protected.
    edits = ['android/app/build.gradle.kts',
             'android/app/src/main/java/com/example/apphear/RecognitionActivity.kt',
             'android/app/src/main/java/com/example/apphear/data/settings/RecognitionModeSettings.kt',
             'android/app/src/main/java/com/example/apphear/ui/settings/SettingsContent.kt',
             'android/app/src/main/java/com/example/apphear/fingerspelling/FingerspellCore.kt',
             'android/app/src/main/res/values/strings.xml',
             'context/feature-schema.md', 'context/progress-tracker.md']
    before = {}
    for name in edits:
        source = ROOT / name
        target = EVIDENCE / 'before' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        before[name] = sha(source)
    assets = {str(p.relative_to(ROOT)): sha(p)
              for p in (ROOT/'android/app/src').rglob('*')
              if p.is_file() and 'assets' in p.relative_to(ROOT).parts}
    (EVIDENCE/'preservation_before.json').write_text(json.dumps(
        {'source_before': before, 'existing_assets': assets}, indent=2), encoding='utf-8')
    checkpoint = RUN/'candidate_checkpoints/epoch_012.pt'
    assert sha(checkpoint) == CHECKPOINT_SHA
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    vocabulary = json.loads((RUN/'vocabulary.json').read_text())
    tokens = vocabulary['tokens'] if isinstance(vocabulary, dict) else vocabulary
    assert tokens[:2] == ['<BLANK>', '<UNK>'] and len(tokens) == 2289
    assert saved['vocabulary'] == tokens
    state = saved['model']
    model = CausalCTC(len(tokens), 48, state['mean'].numpy(), state['std'].numpy())
    model.load_state_dict(state, strict=True)
    model.eval()
    torch.set_num_threads(2)
    target = PACKAGE/'android_assets/continuous_asl'
    target.mkdir(parents=True)
    path = target/'causal_gloss_ctc_fp32.onnx'
    torch.onnx.export(model, (torch.zeros(1, 61, 378),), str(path),
                      input_names=['features'], output_names=['logits'],
                      dynamic_axes={'features': {1: 'observations'}, 'logits': {1: 'observations'}},
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(str(path)))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(20261021)
    results = []
    for length in [1, 2, 17, 61, 97]:
        x = rng.normal(size=(1, length, 378)).astype(np.float32)
        with torch.inference_mode():
            expected = model(torch.from_numpy(x)).numpy()
        actual = session.run(['logits'], {'features': x})[0]
        error = float(np.max(np.abs(expected-actual)))
        assert np.allclose(expected, actual, atol=3e-5, rtol=3e-5), error
        assert np.array_equal(expected.argmax(-1), actual.argmax(-1))
        # A 61-observation receptive field gives immediate causal output, not a
        # 61-frame startup wait. Compare each latest output with full-prefix inference.
        stream = np.concatenate([session.run(['logits'], {'features': x[:, max(0, t-60):t+1]})[0][:, -1:]
                                 for t in range(length)], axis=1)
        assert np.allclose(actual, stream, atol=3e-5, rtol=3e-5)
        results.append({'length': length, 'max_abs_error': error,
                        'argmax_equal': True, 'causal_context_parity': True})
    (target/'vocabulary.json').write_text(json.dumps(tokens, ensure_ascii=False)+'\n', encoding='utf-8')
    manifest = {
        'schema_version': 'apphear.continuous_asl.android_research.v1',
        'feature_schema': 'apphear.perception_vnext.segmentation_features.v1',
        'training_feature_schema': 'apphear.continuous_asl.candidate01_causal378.v1',
        'feature_width': 378, 'classes': 2289, 'blank_id': 0, 'unknown_id': 1,
        'input_name': 'features', 'output_name': 'logits', 'context_observations': 61,
        'minimum_interval_ms': 100.0, 'maximum_gap_ms': 1000.0,
        'maximum_draft_tokens': 128, 'maximum_confirmed_characters': 4096,
        'model_file': path.name, 'model_sha256': sha(path),
        'vocabulary_sha256': sha(target/'vocabulary.json'),
        'source_checkpoint_sha256': CHECKPOINT_SHA,
        'source_epoch': 12, 'normalization': 'TRAIN mean/std embedded in graph; clamp -10..10',
        'decoder': 'Greedy CTC; collapse repeated IDs before removing blank; literal labels',
        'heldout_WER': 0.9567010309278351,
        'status': 'EXPERIMENTAL_DRAFT_ONLY_NOT_ACCEPTED_FOR_PRODUCTION',
        'automatic_commit': False, 'threads': 2, 'opset': 17,
        'safety_policy': 'No observed valid hand: no new token; reset temporal/CTC state. Review before use.',
        'policy_validation': 'Operational safeguards, not calibrated accuracy/no-sign thresholds',
        'permissions': 'Local research integration; does not grant redistribution or commercial dataset rights',
    }
    (target/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    # Nonparticipant synthetic fixture for Android/JVM decoder + native model parity.
    fixture = PACKAGE/'parity_assets/continuous_asl'
    fixture.mkdir(parents=True)
    x = rng.normal(size=(1, 17, 378)).astype(np.float32)
    logits = session.run(['logits'], {'features': x})[0]
    (fixture/'synthetic_features.f32').write_bytes(x.tobytes())
    (fixture/'synthetic_logits.f32').write_bytes(logits.tobytes())
    result = {'status': 'PASS', 'synthetic_only': True, 'recognition_accuracy_claim': False,
              'checkpoint_sha256': CHECKPOINT_SHA, 'model_sha256': sha(path),
              'torch': torch.__version__, 'onnxruntime': ort.__version__, 'tests': results}
    (PACKAGE/'export_parity.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    assert sha(checkpoint) == CHECKPOINT_SHA
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
