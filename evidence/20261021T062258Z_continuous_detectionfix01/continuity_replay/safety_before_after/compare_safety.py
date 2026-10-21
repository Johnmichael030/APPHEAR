"""Previously delivered nohand-flush versus fixed discard, fixed80 development replay only."""
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import zipfile

sys.dont_write_bytecode = True
import numpy as np
import onnxruntime as ort

ROOT=Path.cwd(); OUT=Path(__file__).resolve().parent; PAIRED=OUT.parent; TASK=PAIRED.parent
spec=importlib.util.spec_from_file_location('frozen_paired_metrics',PAIRED/'paired_replay.py')
helpers=importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)
sha=helpers.sha; read=helpers.read; edits=helpers.edits


def save(name,value):
    with (OUT/name).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False); stream.write('\n')


def main():
    assert not (OUT/'config.json').exists(),'EXCLUSIVE_RUN_ALREADY_CONFIGURED'
    source_manifest=PAIRED/'HELD80_manifest.json'; rows=read(source_manifest)
    paired_config=read(PAIRED/'config.json')
    assert sha(source_manifest)==paired_config['frozen_HELD_manifest_sha256']
    assert len(rows)==len({r['sample_id'] for r in rows})==80
    package=ROOT/'packages/continuous_asl_research_01/android_assets/continuous_asl'
    manifest=read(package/'manifest.json'); vocabulary=read(package/'vocabulary.json')
    old_apk=TASK/'before/AppHear-previous.apk'
    with zipfile.ZipFile(old_apk) as archive:
        old_model_sha=hashlib.sha256(archive.read('assets/continuous_asl/causal_gloss_ctc_fp32.onnx')).hexdigest()
        old_vocabulary_sha=hashlib.sha256(archive.read('assets/continuous_asl/vocabulary.json')).hexdigest()
    assert sha(package/'causal_gloss_ctc_fp32.onnx')==old_model_sha==manifest['model_sha256']
    assert sha(package/'vocabulary.json')==old_vocabulary_sha==manifest['vocabulary_sha256']
    before_core=TASK/'before/android/app/src/main/java/com/example/apphear/continuousasl/ContinuousAslCore.kt'
    assert 'decoder.finish()?.let(::append)' in before_core.read_text()
    data={}; inventory=[]
    for row in rows:
        path=Path(row['feature_path']); assert sha(path)==row['feature_sha256']
        with np.load(path,allow_pickle=False) as packed:
            assert str(packed['schema_version'].item())=='apphear.continuous_asl.candidate01_causal378.v1'
            x=packed['X'].copy(); ts=packed['timestamp_ms'].copy(); metadata_string=str(packed['metadata_json'].item())
            metadata=json.loads(metadata_string)
        assert x.dtype==np.float32 and x.shape==(len(ts),378) and np.isfinite(x).all()
        assert np.all(np.diff(ts)>0) and np.all(np.diff(ts)<=1000.0)
        assert metadata['sample_id']==row['sample_id'] and metadata['glosses']==row['glosses']
        visible=np.any(x[:,84:126]>0,axis=1); data[row['sample_id']]=(x,ts,visible)
        inventory.append(dict(sample_id=row['sample_id'],path=str(path),sha256=row['feature_sha256'],
            metadata_sha256=hashlib.sha256(metadata_string.encode()).hexdigest(),labels_sha256=helpers.canonical_sha(row['glosses'])))
    save('HELD80_manifest.json',rows); save('input_inventory.json',inventory)
    paths=[source_manifest,PAIRED/'config.json',PAIRED/'RESULTS.json',PAIRED/'paired_replay.py',
        before_core,package/'manifest.json',package/'vocabulary.json',package/'causal_gloss_ctc_fp32.onnx',Path(__file__)]
    hashes={str(path):sha(path) for path in paths}
    config=dict(schema='apphear.continuous_asl.fixed80_nohand_safety_comparison.v1',
        policies={'OLD_flush':'Old delivered APK: reset context at nohand and publish pending CTC run before reset.',
                  'NEW_discard':'Fixed runtime: same context reset, but discard pending run and publish no token at nohand.'},
        source_hashes=hashes,old_apk_model_sha256=old_model_sha,old_apk_vocabulary_sha256=old_vocabulary_sha,
        frozen_manifest_sha256=sha(OUT/'HELD80_manifest.json'),input_inventory_sha256=sha(OUT/'input_inventory.json'),
        stable_token_ms=200.0,maximum_tokens=128,maximum_gap_ms=1000.0,
        feature_rule='Both use contiguous visible blocks, first frame215:378 zeroed; genuine cache cadence unchanged.',
        end_rule='No oracle end flush; snapshot after last observation.',
        selection='None: descriptive safety comparison frozen before inference; do not tune based on result.',
        scope='Previously development-exposed80 known-signer recording-family held sequences; not independent/live accuracy.',
        pause_resume_events='None in cached utterances; lifecycle correction is separately unit-tested.',
        TEST_accessed=False,training=False,vocabulary_changed=False)
    save('config.json',config); config_hash=sha(OUT/'config.json')
    options=ort.SessionOptions(); options.intra_op_num_threads=2; options.inter_op_num_threads=1
    session=ort.InferenceSession(str(package/'causal_gloss_ctc_fp32.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    aggregate={p:Counter() for p in config['policies']}; examples=[]; predictions=[]; start=time.perf_counter()
    for row in rows:
        x,ts,visible=data[row['sample_id']]; ids=np.zeros(len(x),dtype=np.int64)
        k=0
        while k<len(x):
            if not visible[k]:k+=1;continue
            end=k+1
            while end<len(x) and visible[end]:end+=1
            block=x[k:end].copy(); block[0,215:378]=0.0
            ids[k:end]=session.run(['logits'],{'features':block[None]})[0][0].argmax(-1); k=end
        pair={}
        for policy in config['policies']:
            previous=0; emitted=True; since=0.; hyp=[]; absent_publications=0
            for token,t,seen in zip(ids,ts,visible):
                if len(hyp)>=128:break
                if not seen:
                    if policy=='OLD_flush' and not emitted:
                        hyp.append(vocabulary[previous]); absent_publications+=1
                    previous=0; emitted=True;continue
                token=int(token)
                if token!=previous:
                    if not emitted:hyp.append(vocabulary[previous])
                    previous=token; since=float(t); emitted=token==0
                elif not emitted and t-since>=200.0:
                    hyp.append(vocabulary[previous]); emitted=True
            score=edits(row['glosses'],hyp)
            aggregate[policy].update(score)
            aggregate[policy].update(utterances=1,exact_sequences=int(hyp==row['glosses']),
                empty_outputs=int(not hyp),hypothesis_tokens=len(hyp),nohand_token_publications=absent_publications,
                nohand_model_calls=0)
            item=dict(sample_id=row['sample_id'],source_group=row['source_group'],signer_id=row['signer_id'],
                policy=policy,reference=row['glosses'],hypothesis=hyp,nohand_publications=absent_publications,**score)
            predictions.append(item); pair[policy]=item
        examples.append(dict(sample_id=row['sample_id'],source_group=row['source_group'],
            reference=json.dumps(row['glosses'],ensure_ascii=False),
            old_hypothesis=json.dumps(pair['OLD_flush']['hypothesis'],ensure_ascii=False),
            new_hypothesis=json.dumps(pair['NEW_discard']['hypothesis'],ensure_ascii=False),
            old_nohand_publications=pair['OLD_flush']['nohand_publications'],
            new_nohand_publications=pair['NEW_discard']['nohand_publications'],
            old_errors=sum(pair['OLD_flush'][k] for k in ['substitutions','deletions','insertions']),
            new_errors=sum(pair['NEW_discard'][k] for k in ['substitutions','deletions','insertions'])))
    elapsed=time.perf_counter()-start
    metrics={}
    for policy,c in aggregate.items():
        metrics[policy]=dict(c); metrics[policy]['WER']=sum(c[k] for k in ['substitutions','deletions','insertions'])/c['reference_tokens']
        metrics[policy]['exact_sequence_fraction']=c['exact_sequences']/c['utterances']
    prior=read(PAIRED/'RESULTS.json')['metrics']['HELD80']['A_reset_nohand']
    for key in ['substitutions','deletions','insertions','reference_tokens','hypothesis_tokens','empty_outputs']:
        assert metrics['NEW_discard'][key]==prior[key],('PAIRED_BASELINE_MISMATCH',key)
    assert metrics['NEW_discard']['nohand_token_publications']==0
    result=dict(status='PASS_COMPUTATIONAL_SAFETY_COMPARISON_NOT_ACCURACY_ACCEPTANCE',metrics=metrics,
        measured_inference_and_scoring_seconds=elapsed,config_sha256=config_hash,
        no_TEST=True,source_model_and_labels_unchanged=True,
        interpretation='Suppressing absent-hand publications enforces the declared visibility safety contract; report deletion/WER tradeoff explicitly. Lifecycle tests do not establish quality. No deployment/model selection occurred in this replay.')
    save('RESULTS.json',result); save('PREDICTIONS.json',predictions)
    with (OUT/'PREDICTION_EXAMPLES.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(examples[0])); writer.writeheader();writer.writerows(examples)
    with (OUT/'METRICS.csv').open('x',encoding='utf-8',newline='') as stream:
        metric_rows=[dict(policy=p,**m) for p,m in metrics.items()]
        writer=csv.DictWriter(stream,fieldnames=list(metric_rows[0]));writer.writeheader();writer.writerows(metric_rows)
    assert sha(OUT/'config.json')==config_hash and all(sha(p)==value for p,value in hashes.items())
    assert all(sha(i['path'])==i['sha256'] for i in inventory)
    save('INTEGRITY.json',dict(status='PASS',feature_files=80,inputs_and_model_unchanged=True,
        paired_parent_results_config_unchanged=True,NEW_matches_prior_fixed80_reset_replay=True,TEST_accessed=False))
    lines=['# Old delivered no-hand flush versus new discard — fixed80','','No fitting or tuning; previously development-exposed known-signer recording-family holdout. Cached observations do not reproduce intervening phone frames or establish live/no-sign false-activation rates.','',
        '|Policy|WER|SUB|DEL|INS|No-hand token publications|Exact|Empty outputs|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for policy,m in metrics.items():lines.append(f"|{policy}|{100*m['WER']:.4f}%|{m['substitutions']}|{m['deletions']}|{m['insertions']}|{m['nohand_token_publications']}|{m['exact_sequences']}/80|{m['empty_outputs']}|")
    lines += ['', 'The safety fix is not automatically an accuracy gain. Compare the full edit tradeoff above before describing user-facing recognition quality. Both policies have identical no-hand model-call abstention, weights, vocabulary, reset feature history,200ms settle policy and128-token limit. No oracle end flush was used. The paused-background lifecycle correction is separately tested and does not participate in these uninterrupted cached utterances.', '',
        'The previous delivered APK model/vocabulary bytes match the frozen package exactly. Old source contains the pending-token flush; new replay uses discard. NEW reproduces the paired context-replay A counts exactly. All80 feature hashes and sealed parent config/results hashes remain unchanged; no reserved TEST payload was read.', '',
        f'Actual CPU vectorized inference/scoring duration: {elapsed:.6f}s. See config.json, METRICS.csv, PREDICTION_EXAMPLES.csv and INTEGRITY.json.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'commands_and_results.txt').write_text("Working directory: "+str(ROOT)+"\nExecuted once:\n& '.\\experiments\\continuous_asl\\20261019T155428Z_cslr01\\runtime\\venv\\Scripts\\python.exe' '.\\evidence\\20261021T062258Z_continuous_detectionfix01\\continuity_replay\\safety_before_after\\compare_safety.py'\nExit0. Config frozen before ORT inference; no fitting, tuning or deployment.\n"+json.dumps(result,indent=2)+'\n',encoding='utf-8')
    files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!='artifact_hashes.sha256')
    (OUT/'artifact_hashes.sha256').write_text(''.join(sha(p)+'  '+str(p)+'\n' for p in files),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
