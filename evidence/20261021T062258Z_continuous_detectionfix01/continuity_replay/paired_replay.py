"""Frozen development replay only: no fitting, TEST, model/vocabulary edits or serving changes."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import onnxruntime as ort

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'experiments/continuous_asl'
TRAIN = BASE / '20261021T112738_nested192_control01/subset_manifest.json'
FOLD = BASE / '20261020T143609Z_learning_diagnostic01/fold_manifest.json'
RESOLVED = FOLD.parent / 'resolved_manifest.json'
PACKAGE = ROOT / 'packages/continuous_asl_research_01/android_assets/continuous_asl'
PARENT = OUT.parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    with (OUT / name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def csvsave(name, rows):
    with (OUT / name).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def family(row):
    match = re.match(r'^(\d{4}-\d{2}-\d{2})_', row['source_group'])
    return row['signer_id'] + '::' + (match[1] if match else 'UNRESOLVED_ALL_UNDATED_MASTERS')


def edits(reference, hypothesis):
    # Same deterministic match/substitute, delete, insert tie order as ctc_experiment.py.
    n, m = len(reference), len(hypothesis)
    distance = np.zeros((n + 1, m + 1), dtype=np.int32)
    distance[:, 0] = np.arange(n + 1)
    distance[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            distance[i, j] = min(distance[i-1, j-1] + (reference[i-1] != hypothesis[j-1]),
                                 distance[i-1, j] + 1, distance[i, j-1] + 1)
    result = dict(substitutions=0, deletions=0, insertions=0, reference_tokens=n)
    i, j = n, m
    while i or j:
        if i and j and distance[i, j] == distance[i-1, j-1] + (reference[i-1] != hypothesis[j-1]):
            result['substitutions'] += int(reference[i-1] != hypothesis[j-1]); i -= 1; j -= 1
        elif i and distance[i, j] == distance[i-1, j] + 1:
            result['deletions'] += 1; i -= 1
        else:
            result['insertions'] += 1; j -= 1
    return result


def main():
    assert not (OUT / 'config.json').exists(), 'EXCLUSIVE_RUN_ALREADY_CONFIGURED'
    frozen = read(FOLD)
    train = read(TRAIN)
    universe = read(RESOLVED)  # Contains only the prior1500 TRAIN and80 internal development rows.
    lookup = {row['sample_id']: row for row in universe}
    held_ids = frozen['compatible_eval_sample_ids']
    held = [lookup[sample] for sample in held_ids]
    assert len(train) == 192 and len(held) == 80
    assert len({r['sample_id'] for r in train}) == 192 and len(set(held_ids)) == 80
    assert {r['sample_id'] for r in train} <= set(frozen['train_sample_ids'])
    assert not ({r['sample_id'] for r in train} & set(frozen['internal_eval_sample_ids']))
    assert not ({family(r) for r in train} & {family(r) for r in held})
    assert {r['signer_id'] for r in held} <= {r['signer_id'] for r in train}
    assert {r['training_pool'] for r in train + held} == {'ASLLRP3_LITERAL_DOMINANT_REALIZED_OCCURRENCE'}
    vocabulary = read(PACKAGE / 'vocabulary.json')
    manifest = read(PACKAGE / 'manifest.json')
    assert vocabulary[:2] == ['<BLANK>', '<UNK>'] and len(vocabulary) == 2289
    assert sha(PACKAGE / 'causal_gloss_ctc_fp32.onnx') == manifest['model_sha256']
    assert sha(PACKAGE / 'vocabulary.json') == manifest['vocabulary_sha256']
    assert manifest['source_epoch'] == 12
    data, inventory = {}, []
    for role, rows in [('TRAIN192', train), ('HELD80', held)]:
        for row in rows:
            feature = Path(row['feature_path'])
            assert sha(feature) == row['feature_sha256']
            with np.load(feature, allow_pickle=False) as packed:
                assert str(packed['schema_version'].item()) == 'apphear.continuous_asl.candidate01_causal378.v1'
                x = packed['X'].copy(); ts = packed['timestamp_ms'].copy()
                metadata_string = str(packed['metadata_json'].item())
                metadata = json.loads(metadata_string)
            assert metadata['sample_id'] == row['sample_id'] and metadata['glosses'] == row['glosses']
            assert x.dtype == np.float32 and x.shape == (len(ts), 378) and np.isfinite(x).all()
            assert np.all(np.isfinite(ts)) and np.all(np.diff(ts) > 0)
            assert all(label in vocabulary[2:] for label in row['glosses'])
            assert len(x) >= len(row['glosses']) + sum(a == b for a,b in zip(row['glosses'], row['glosses'][1:]))
            visible = np.any(x[:,84:126] > 0, axis=1)
            # Every current cache has no >1000-ms source gap; reject a new contract instead of
            # silently deciding how to rederive cached deltas at a long gap.
            assert not np.any(np.diff(ts) > 1000.0), 'UNEXPECTED_LONG_GAP_REQUIRES_FEATURE_REDERIVATION'
            data[row['sample_id']] = (x, ts, visible)
            inventory.append(dict(role=role, sample_id=row['sample_id'], feature_path=str(feature),
                feature_sha256=row['feature_sha256'], labels_sha256=canonical_sha(row['glosses']),
                metadata_sha256=hashlib.sha256(metadata_string.encode()).hexdigest(),
                manifest_row_sha256=canonical_sha(row), observations=len(x)))
    save('TRAIN192_manifest.json', train)
    save('HELD80_manifest.json', held)
    save('input_inventory.json', inventory)
    source_paths = [TRAIN, FOLD, RESOLVED, PACKAGE/'vocabulary.json', PACKAGE/'manifest.json',
        PACKAGE/'causal_gloss_ctc_fp32.onnx', PARENT/'runtime_replay.py',
        PARENT/'runtime_replay_config.json', PARENT/'RUNTIME_REPLAY_RESULTS.json', Path(__file__),
        ROOT/'tools/continuous_asl/ctc_experiment.py']
    sources = {str(p): sha(p) for p in source_paths}
    config = dict(schema='apphear.continuous_asl.paired_missingness_replay.v1',
        model='frozen packaged epoch12; CPU ONNX2threads', source_hashes=sources,
        frozen_TRAIN_manifest_sha256=sha(OUT/'TRAIN192_manifest.json'),
        frozen_HELD_manifest_sha256=sha(OUT/'HELD80_manifest.json'),
        input_inventory_sha256=sha(OUT/'input_inventory.json'),
        policies={'A_reset_nohand': 'No-hand drops pending CTC and resets feature/model history. First visible-run frame215:378 zeroed.',
                  'B_retain_missing_features': 'Append all genuine cached missingness-containing features; do not reset feature/model history on nohand; abstain and reset CTC decoder on nohand.'},
        stable_token_ms=200.0, maximum_tokens=128, maximum_gap_ms=1000.0,
        long_gap_rule='Reset temporal/model/decoder state. All272 caches prechecked to have0 gaps>1000ms; therefore B uses cachedX byte-identically.',
        cached_cadence='Already prepared genuine10Hz observations; intervening Android observations/phone speed not simulated.',
        end_of_utterance_rule='No oracle flush; snapshot after final observation only.',
        inference_implementation='Vectorized causal inference. Only visible-position predictions are consumed. No-hand positions are genuine context only; unused nohand logits may be computed offline. Logical serving calls on nohand remain0.',
        gate=['B WER strictly lower than A on TRAIN192 and HELD80',
              'B insertion count <= A on both partitions',
              'No logical inference calls/published tokens on nohand',
              'RF61 prefix numerical parity passes'],
        parity_indices='longest TRAIN sequence:0,1,30,60,last', parity_atol=1e-4, parity_rtol=3e-5,
        heldout_scope='Previously development-exposed known-signer recording-family holdout, NOT independent or unseen-signer validation.',
        training=False, vocabulary_changed=False, TEST_accessed=False, model_selection=False,
        serving_policy_deployment=False)
    save('config.json', config)  # Frozen before constructing ORT or performing inference.
    frozen_config_sha = sha(OUT/'config.json')
    options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(PACKAGE/'causal_gloss_ctc_fp32.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
    actual_ort_calls = Counter()

    def infer(x, name):
        actual_ort_calls[name] += 1
        return session.run(['logits'], {'features':x[None]})[0][0]

    def predict(x, visible, policy):
        ids = np.zeros(len(x), dtype=np.int64)
        if policy == 'B_retain_missing_features':
            if visible.any(): ids = infer(x, policy).argmax(-1)
        else:
            start = 0
            while start < len(x):
                if not visible[start]: start += 1; continue
                end = start+1
                while end < len(x) and visible[end]: end += 1
                block = x[start:end].copy(); block[0,215:378] = 0.0
                ids[start:end] = infer(block, policy).argmax(-1)
                start = end
        return ids

    def stable(ids, timestamps, visible):
        previous=0; emitted=True; since=0.; result=[]; nohand_emissions=0; logical_nohand_calls=0
        old_time = None
        for index, (timestamp, seen) in enumerate(zip(timestamps, visible)):
            if len(result) >= 128: break
            if old_time is not None and timestamp-old_time > 1000.0:
                previous=0; emitted=True
            old_time=float(timestamp)
            if not seen:
                count=len(result); previous=0; emitted=True
                nohand_emissions += len(result)-count
                continue
            token=int(ids[index])
            if token != previous:
                if not emitted: result.append(vocabulary[previous])
                previous=token; since=float(timestamp); emitted=token==0
            elif not emitted and timestamp-since>=200.0:
                result.append(vocabulary[previous]); emitted=True
        return result, nohand_emissions, logical_nohand_calls

    aggregate = defaultdict(Counter); source_counts = defaultdict(Counter); events=defaultdict(Counter)
    examples=[]; all_predictions=[]; began=time.perf_counter()
    for role, rows in [('TRAIN192', train), ('HELD80', held)]:
        for row in rows:
            x, ts, visible = data[row['sample_id']]
            events[role].update(utterances=1, frames=len(x), visible_frames=int(visible.sum()),
                nohand_frames=int((~visible).sum()), visible_to_nohand=int(np.sum(visible[:-1] & ~visible[1:])),
                nohand_to_visible=int(np.sum(~visible[:-1] & visible[1:])), long_timestamp_gaps=0,
                visible_runs=int(visible[0])+int(np.sum(~visible[:-1] & visible[1:])))
            pair={}
            for policy in config['policies']:
                ids=predict(x,visible,policy)
                hyp, nohand_outputs, nohand_calls = stable(ids,ts,visible)
                assert nohand_outputs==nohand_calls==0
                score=edits(row['glosses'], hyp)
                extra=dict(utterances=1, exact_sequences=int(hyp==row['glosses']), hypothesis_tokens=len(hyp),
                    empty_outputs=int(not hyp), nohand_token_emissions=nohand_outputs,
                    logical_model_calls_on_nohand=nohand_calls, logical_visible_inference_positions=int(visible.sum()),
                    token_limit_hits=int(len(hyp)>=128))
                aggregate[(role,policy)].update(score); aggregate[(role,policy)].update(extra)
                source_counts[(role, policy, row['source_group'], row['signer_id'], family(row))].update(score)
                source_counts[(role, policy, row['source_group'], row['signer_id'], family(row))].update(extra)
                item=dict(role=role, policy=policy, sample_id=row['sample_id'], source_group=row['source_group'],
                    signer_id=row['signer_id'], family=family(row), reference=row['glosses'], hypothesis=hyp,
                    nohand_frames=int((~visible).sum()), **score, exact=int(hyp==row['glosses']))
                all_predictions.append(item); pair[policy]=item
            examples.append(dict(role=role, sample_id=row['sample_id'], source_group=row['source_group'],
                signer_id=row['signer_id'], reference=json.dumps(row['glosses'],ensure_ascii=False),
                A_hypothesis=json.dumps(pair['A_reset_nohand']['hypothesis'],ensure_ascii=False),
                B_hypothesis=json.dumps(pair['B_retain_missing_features']['hypothesis'],ensure_ascii=False),
                A_errors=sum(pair['A_reset_nohand'][k] for k in ['substitutions','deletions','insertions']),
                B_errors=sum(pair['B_retain_missing_features'][k] for k in ['substitutions','deletions','insertions']),
                nohand_frames=int((~visible).sum())))
    inference_seconds=time.perf_counter()-began

    def metric(counter):
        result=dict(counter)
        result['WER']=sum(counter[k] for k in ['substitutions','deletions','insertions'])/counter['reference_tokens']
        result['exact_sequence_fraction']=counter['exact_sequences']/counter['utterances']
        return result

    metrics={role:{policy:metric(aggregate[(role,policy)]) for policy in config['policies']} for role in ['TRAIN192','HELD80']}
    prior=read(PARENT/'RUNTIME_REPLAY_RESULTS.json')['metrics']['nohand_discard_fix']
    for key in ['substitutions','deletions','insertions','reference_tokens','tokens']:
        actual_key='hypothesis_tokens' if key=='tokens' else key
        assert metrics['TRAIN192']['A_reset_nohand'][actual_key]==prior[key], ('ROOT_BASELINE_MISMATCH', key)

    longest=max(train,key=lambda row:len(data[row['sample_id']][0]))
    x,ts,visible=data[longest['sample_id']]
    full=infer(x,'parity_full_B')
    parity=[]
    for index in sorted({0,1,30,60,len(x)-1}):
        if index>=len(x):continue
        prefix=x[max(0,index-60):index+1]
        bounded=infer(prefix,'parity_prefix_B')[-1]
        expected=full[index]
        passed=bool(np.all(np.abs(bounded-expected)<=1e-4+3e-5*np.abs(expected)))
        parity.append(dict(policy='B_retain_missing_features', sample_id=longest['sample_id'], index=index,
            prefix_observations=len(prefix), max_abs_error=float(np.abs(bounded-expected).max()),
            argmax_equal=bool(bounded.argmax()==expected.argmax()), numerical_tolerance_pass=passed,
            visible=bool(visible[index]), inference_scope='Offline numerical test even if nohand; not a serving prediction'))
        assert passed
        if visible[index]:
            start=index
            while start>0 and visible[start-1]:start-=1
            end=index+1
            while end<len(x) and visible[end]:end+=1
            block=x[start:end].copy(); block[0,215:378]=0.0
            block_full=infer(block,'parity_full_A')
            local=index-start
            block_prefix=block[max(0,local-60):local+1]
            bounded=infer(block_prefix,'parity_prefix_A')[-1]; expected=block_full[local]
            passed=bool(np.all(np.abs(bounded-expected)<=1e-4+3e-5*np.abs(expected)))
            parity.append(dict(policy='A_reset_nohand',sample_id=longest['sample_id'],index=index,
                prefix_observations=len(block_prefix),max_abs_error=float(np.abs(bounded-expected).max()),
                argmax_equal=bool(bounded.argmax()==expected.argmax()),numerical_tolerance_pass=passed,
                visible=True,inference_scope='Within the existing contiguous visible model-history block'))
            assert passed
    gate={role:dict(WER_lower=metrics[role]['B_retain_missing_features']['WER']<metrics[role]['A_reset_nohand']['WER'],
        insertions_nonincrease=metrics[role]['B_retain_missing_features']['insertions']<=metrics[role]['A_reset_nohand']['insertions'])
        for role in ['TRAIN192','HELD80']}
    gate_pass=all(all(part.values()) for part in gate.values()) and all(p['numerical_tolerance_pass'] for p in parity)
    result=dict(status='PASS_PAIRED_GATE' if gate_pass else 'BLOCKED_PAIRED_GATE',
        scope='Predeclared development diagnostic; NOT independent evaluation or unseen-signer generalization.',
        metrics=metrics, gate=gate, events={k:dict(v) for k,v in events.items()},
        inference_and_scoring_seconds=inference_seconds, actual_vectorized_ORT_calls=dict(actual_ort_calls),
        numerical_parity_pass=True, config_sha256=frozen_config_sha,
        parent_TRAIN192_A_reproduction='PASS exact edit/token counts',
        deployment='NOT_PERFORMED; root must separately decide after gates and review.', TEST_accessed=False)
    save('RESULTS.json',result); save('PREFIX_PARITY.json',parity); save('PREDICTIONS.json',all_predictions)
    csvsave('METRICS.csv',[dict(role=role,policy=policy,**metrics[role][policy]) for role in metrics for policy in config['policies']])
    csvsave('PER_SOURCE.csv',[dict(role=k[0],policy=k[1],source_group=k[2],signer_id=k[3],family=k[4],**metric(v)) for k,v in sorted(source_counts.items())])
    csvsave('PREDICTION_EXAMPLES.csv',examples)
    assert sha(OUT/'config.json')==frozen_config_sha
    assert all(sha(p)==expected for p,expected in sources.items())
    assert all(sha(i['feature_path'])==i['feature_sha256'] for i in inventory)
    save('INTEGRITY.json',dict(status='PASS',input_features=272,config_sha256=frozen_config_sha,
        source_manifest_model_hashes_unchanged=True,all_feature_hashes_unchanged=True,TEST_accessed=False))
    lines=['# Paired missingness-context replay','','Status: **'+result['status']+'**. Frozen epoch12; no fitting or app changes.','',
        'Both policies discard pending CTC on no hands and publish no token there. A resets feature/model context; B retains genuine missingness-containing cached features. Vocabulary, labels, stable200ms hold,128-token limit and end snapshot are identical. No oracle end flush.','',
        '|Partition|Policy|WER|SUB|DEL|INS|Exact sequences|', '|---|---|---:|---:|---:|---:|---:|']
    for role in metrics:
        for policy in config['policies']:
            m=metrics[role][policy]
            lines.append(f"|{role}|{policy}|{100*m['WER']:.4f}%|{m['substitutions']}|{m['deletions']}|{m['insertions']}|{m['exact_sequences']}/{m['utterances']}|")
    lines += ['', 'Gate requires lower B WER and no insertion-count increase in BOTH partitions. Gate details: `'+json.dumps(gate)+'`. Numerically equivalent RF61 prefixes pass the frozen tolerance on the longest TRAIN sequence. A exactly reproduces parent TRAIN192 edit/token counts.', '',
        'Logical serving inference positions are visible observations only; logical model calls on no hands and published no-hand tokens are0 under both policies. Offline vectorized causal inference computes unused no-hand logits as an optimization, not as app behavior or device timing. All272 caches have0 timestamp gaps>1000ms; a gap would reset history/decoder, and unprovided raw rederivation is rejected rather than silently changing cached features.', '',
        'Interpretation: changes in substitutions/deletions/insertions are descriptive. The80-sequence partition was already development-exposed and the same known signers appear in TRAIN, with guarded recording families held out. This is NOT independent/unseen-signer evidence, no production acceptance and no permission to merge label pools. No negative/no-sign ground truth is available, so false-activation rate cannot be measured here.', '',
        'Preserved: all manifests, checkpoints, labels, vocabulary, normalization, app assets and reserved TEST payloads. See config.json/input_inventory.json for frozen hashes, PER_SOURCE.csv for source metrics, PREDICTION_EXAMPLES.csv/PREDICTIONS.json for literal hypotheses, PREFIX_PARITY.json and INTEGRITY.json.', '',
        f'Actual measured vectorized inference/scoring duration: {inference_seconds:.6f}seconds; no phone latency claim.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!='artifact_hashes.sha256')
    (OUT/'artifact_hashes.sha256').write_text(''.join(sha(p)+'  '+str(p)+'\n' for p in files),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
