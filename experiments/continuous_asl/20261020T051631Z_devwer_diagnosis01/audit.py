"""Inference-only audit of frozen research artifacts; never opens reserved TEST.

All writes are exclusive and confined to this new diagnostic directory.
The only decoder counterfactual is predeclared beam10/top16, with no sweep.
"""
from __future__ import annotations
import ast
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TREE = ROOT / 'experiments/continuous_asl'
SIX = TREE / '20261020_continuous_expand06'
RUNS = {k: TREE / f'20261020_continuous_expand{n:02d}' for k,n in [('baseline',3),('pilot906',9),('main2119',8)]}
SCHEMA = 'apphear.continuous_asl.candidate01_causal378.v1'
sys.path.insert(0, str(ROOT / 'tools/continuous_asl'))
sys.path.insert(0, str(ROOT))
from ctc_experiment import CausalCTC, batch, decode, edit_counts, evaluate, seeds

INPUTS = {}
CHECKS = []

def sha(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576), b''):
            h.update(chunk)
    INPUTS[str(path)] = h.hexdigest()
    return h.hexdigest()

def read(path):
    path = Path(path)
    sha(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))

def write(name, data):
    path = (HERE/name).resolve()
    if HERE not in path.parents:
        raise ValueError('DIAGNOSTIC_OUTPUT_SCOPE')
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write('\n')

def csv_write(name, rows):
    with (HERE/name).open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})

def check(name, passed, evidence, limitation=''):
    CHECKS.append({'finding':name,'status':'PASS' if passed else 'PARTIAL','passed':bool(passed),'evidence':evidence,'limitation':limitation})

def frozen_snapshot():
    files = set()
    for number in range(3,10):
        base = TREE/f'20261020_continuous_expand{number:02d}'
        index = base/'artifact_manifest.json'
        if index.exists():
            files.add(index)
            files.update(base/r['path'] for r in read(index)['files'])
    # Existing protection includes production assets, prior results, reports.
    baseline = read(TREE/'20261020_continuous_expand04/preservation_baseline.json')
    files.update(Path(r['path']) for r in baseline['metadata_snapshot'])
    files.update(ROOT/p for p in ['tools/continuous_asl/ctc_experiment.py','pipeline/perception_vnext/segmentation/learned_features.py','pipeline/perception_vnext/recognition_schema.py'])
    return {str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(files)}

def levenshtein(a,b):
    previous = list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]

def collapse(path):
    return tuple(v for i,v in enumerate(path) if v!=0 and (i==0 or v!=path[i-1]))

def prefix_beam(logp, width=10, top=16):
    """CTC prefix sum; blank=0; fixed pruning, no language model or penalties."""
    beam={(): (0., -math.inf)}
    for frame in logp:
        candidates=set(np.argsort(frame)[-min(top,len(frame)):].tolist()) | {0}
        nxt=defaultdict(lambda:[-math.inf,-math.inf])
        for prefix,(pb,pnb) in beam.items():
            total=float(np.logaddexp(pb,pnb))
            nxt[prefix][0]=float(np.logaddexp(nxt[prefix][0],total+frame[0]))
            for c in candidates-{0}:
                value=float(frame[c])
                if prefix and c==prefix[-1]:
                    nxt[prefix][1]=float(np.logaddexp(nxt[prefix][1],pnb+value))
                    ext=prefix+(c,)
                    nxt[ext][1]=float(np.logaddexp(nxt[ext][1],pb+value))
                else:
                    ext=prefix+(c,)
                    nxt[ext][1]=float(np.logaddexp(nxt[ext][1],total+value))
        beam=dict(sorted(nxt.items(),key=lambda kv:float(np.logaddexp(*kv[1])),reverse=True)[:width])
    ordered=sorted(beam.items(),key=lambda kv:float(np.logaddexp(*kv[1])),reverse=True)
    return ordered[0][0], float(np.logaddexp(*ordered[0][1]))

def unit_checks():
    vocab=['<BLANK>','<UNK>','A','B']
    cases=[([0,0],[]),([2,2,0,2],['A','A']),([2,3,3],['A','B']),([1,0,1],['<UNK>','<UNK>'])]
    for ids,expected in cases:
        z=torch.full((len(ids),4),-10.)
        z[range(len(ids)),ids]=10.
        check('synthetic_decode_'+str(ids),decode(z,vocab)==expected,{'ids':ids,'expected':expected,'actual':decode(z,vocab)},'Synthetic contract check, not model evaluation.')
    rng=np.random.default_rng(20261020)
    for trial in range(4):
        p=rng.uniform(.05,1,(4,3));p=p/p.sum(1,keepdims=True)
        totals=defaultdict(float)
        for path in itertools.product(range(3),repeat=4):
            totals[collapse(path)]+=math.prod(p[t,v] for t,v in enumerate(path))
        exact=max(totals,key=totals.get)
        actual,lp=prefix_beam(np.log(p),width=1000,top=3)
        check(f'synthetic_beam_exhaustive_{trial}',actual==exact and abs(math.exp(lp)-totals[exact])<1e-12,{'exact_prefix':exact,'beam_prefix':actual,'probability_error':abs(math.exp(lp)-totals[exact])},'Exhaustive synthetic 3^4 paths; real beam is pruned.')
    for a,b in [([],[]),(['IX-3p:i','READ++'],[]),(['A','A'],['A']),(['A'],['B','A'])]:
        e=edit_counts(a,b)
        check('synthetic_wer_'+str(a)+str(b),sum(e[k] for k in ('substitutions','deletions','insertions'))==levenshtein(a,b),e,'Synthetic metric check.')

def get_feature(row, data):
    sample=row['sample_id']
    if sample not in data:
        path=Path(row['feature_path'])
        if sha(path)!=row['feature_sha256']:raise ValueError('FROZEN_FEATURE_HASH')
        with np.load(path,allow_pickle=False) as archive:
            if str(archive['schema_version'].item())!=SCHEMA:raise ValueError('SCHEMA_VERSION')
            x=archive['X'].copy();ts=archive['timestamp_ms'].copy()
        if x.dtype!=np.float32 or x.shape!=(len(ts),378) or not np.isfinite(x).all() or not np.all(np.diff(ts)>0):raise ValueError('FEATURE_CONTRACT')
        data[sample]=(x,ts)
    return data[sample]

def load_model(run, which='best'):
    base=run/'models/reference48'
    pointer=read(base/f'checkpoints/{which}.json')
    path=base/pointer['path']
    if sha(path)!=pointer['sha256']:raise ValueError('CHECKPOINT_SHA')
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['schema']!=SCHEMA:raise ValueError('MODEL_FEATURE_SCHEMA')
    vocab=read(base/'vocabulary.json')['tokens']
    config=read(base/'config.json')
    if state['vocabulary']!=vocab or state['config']!=config:raise ValueError('VOCAB_CONFIG_CHECKPOINT_MISMATCH')
    model=CausalCTC(len(vocab),config['channels'],state['model']['mean'].numpy(),state['model']['std'].numpy())
    model.load_state_dict(state['model']);model.eval()
    return model,state,vocab,pointer

def function_hashes():
    path=ROOT/'tools/continuous_asl/ctc_experiment.py'
    source=path.read_text(encoding='utf-8');sha(path)
    return {n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('decode','evaluate','edit_counts','batch','CausalCTC','CausalBlock')}

def raw_parity(rows,data):
    from pipeline.perception_vnext.recognition_schema import load_raw_sequence
    from pipeline.perception_vnext.segmentation.learned_features import derive_causal_segmentation_features
    selected=[rows['main2119'][0],rows['baseline'][0],next(r for r in rows['pilot906'] if 'source_cache' in r)]
    reports=[]
    for row in selected:
        feature=Path(row['feature_path']);raw_path=Path(str(feature).replace('.features.npz','.raw.npz'))
        if 'source_cache' in row:
            raw_path=Path(str(row['source_cache']).replace('.raw.json','.raw.npz'))
        if not raw_path.exists():
            reports.append({'sample_id':row['sample_id'],'status':'UNKNOWN','reason':'No adjacent saved raw file; extraction not repeated.'});continue
        sha(raw_path);raw=load_raw_sequence(raw_path)
        if 'source_cache' in row:
            mask=(raw.timestamp_ms>=row['start_ms'])&(raw.timestamp_ms<row['end_ms'])
            raw=replace(raw,**{k:getattr(raw,k)[mask] for k in raw.__dataclass_fields__ if k!='metadata'})
        x=derive_causal_segmentation_features(raw);saved,ts=get_feature(row,data)
        mid=max(1,len(x)//2)
        prefix=replace(raw,**{k:getattr(raw,k)[:mid] for k in raw.__dataclass_fields__ if k!='metadata'})
        ok=np.array_equal(x,saved) and np.array_equal(raw.timestamp_ms,ts) and np.array_equal(x[:mid],derive_causal_segmentation_features(prefix))
        reports.append({'sample_id':row['sample_id'],'status':'PASS' if ok else 'PARTIAL','feature_exact_match':bool(np.array_equal(x,saved)),'timestamp_exact_match':bool(np.array_equal(raw.timestamp_ms,ts)),'causal_prefix_match':bool(np.array_equal(x[:mid],derive_causal_segmentation_features(prefix)))})
    check('bounded_cached_raw_to_feature_parity',all(r['status']=='PASS' for r in reports),reports,'Three saved raw fixtures only; no video decoding, acquisition or feature-file writes.')

def pool_audit(canonical,data):
    pools=defaultdict(list)
    for row in canonical:
        if row['canonical_role']=='eligible_train':pools[row['training_pool']].append(row)
    out=[];details={};lengths=[]
    csv_path=SIX/'data/raw/asllrp2025/asllrp_sentence_signs_2025_06_28.csv'
    sha(csv_path)
    with csv_path.open(encoding='utf-8-sig',newline='') as stream:
        csv_source={line:r for line,r in enumerate(csv.DictReader(stream),2)}
    for name,rows in sorted(pools.items()):
        labels=Counter(t for r in rows for t in r['glosses']);ctc_bad=[];bad_events=[];features=[];time_checks=0;event_joins=0
        for row in rows:
            x,ts=get_feature(row,data);u=len(row['glosses']);minimum=u+sum(a==b for a,b in zip(row['glosses'],row['glosses'][1:]))
            if len(x)<minimum:ctc_bad.append(row['sample_id'])
            lengths.append({'pool':name,'sample_id':row['sample_id'],'input_observations':len(x),'target_tokens':u,'minimum_ctc_frames':minimum,'ctc_margin':len(x)-minimum})
            if 'events' in row:
                events=row['events'];ordered=sorted(events,key=lambda e:(e['start_frame_master'],e['source_row']))
                # Verify literal publisher values, including punctuation, hidden labels and whole compounds.
                if events!=ordered or [e['label'] for e in events]!=row['glosses']:bad_events.append(row['sample_id']+':ORDER_OR_LABEL')
                for e in events:
                    src=csv_source[e['source_row']]
                    ok=(src['Occurrence label']==e['label'] and int(src['Video ID number'])==int(e['event_id']) and
                        src['Utterance video filename']==row['utterance_filename'] and
                        int(src['Start frame of the sign video'])==e['start_frame_master'] and int(src['End frame of the sign video'])==e['end_frame_master'])
                    if not ok:bad_events.append(row['sample_id']+':CSV_EVENT_JOIN')
                    if not row['utterance_start_frame_master']<=e['start_frame_master']<=e['end_frame_master']<=row['utterance_end_frame_master']:bad_events.append(row['sample_id']+':EVENT_BOUND')
                    event_joins+=1
                if row['decoded_frames']!=row['utterance_end_frame_master']-row['utterance_start_frame_master']+1:bad_events.append(row['sample_id']+':FRAME_COUNT')
                time_checks+=1
            elif 'sign_times_ms' in row:
                if len(row['sign_times_ms'])!=u or not all(row['start_ms']<=a<=b<=row['end_ms'] for a,b in row['sign_times_ms']):bad_events.append(row['sample_id']+':EAF_BOUND')
                if not (ts[0]>=row['start_ms'] and ts[-1]<row['end_ms']):bad_events.append(row['sample_id']+':OBSERVATION_SPAN')
                time_checks+=1
            features.append([float(x[:,172].mean()),float(x[:,173].mean()),float(x[:,214].mean()),len(x),u])
        stats=np.asarray(features)
        details[name]={'utterances':len(rows),'unique_labels':len(labels),'target_tokens':sum(labels.values()),'singleton_labels':sum(n==1 for n in labels.values()),
            'collections':dict(Counter(r['collection'] for r in rows)),'source_recordings':len({r['source_group'] for r in rows}),
            'known_signer_ids':sorted({str(r['signer_id']) for r in rows if r.get('signer_id') is not None}),
            'annotation_types':dict(Counter(r['annotation_type'] for r in rows)),'schema':SCHEMA,'ctc_invalid':ctc_bad,'annotation_errors':bad_events,
            'publisher_bound_checks':time_checks,'direct_csv_event_joins':event_joins,'unweighted_mean_left_right_hand_and_body_available':stats[:,:3].mean(0).tolist(),
            'observations_p5_p50_p95':np.percentile(stats[:,3],[5,50,95]).tolist(),'target_tokens_p5_p50_p95':np.percentile(stats[:,4],[5,50,95]).tolist(),
            'spatial_representation':'Camera-normalized XY and ROI, not signer/root/handedness invariant; full378 current/missingness/backward delta blocks.',
            'label_counts':dict(labels)}
        is_asl=name.startswith('ASLLRP3')
        out.append({'status':'PARTIAL','pool':name,'eligible_utterances':len(rows),'literal_label_source':'SignStream3 CSV Occurrence label; dominant-realized events, including hidden events' if is_asl else 'NCSLGR2 main gloss; literal case/locus/repetition notation and source HOLD events',
            'feature_schema':SCHEMA,'feature_width':378,'vocabulary_labels':len(labels),'ctc_invalid':len(ctc_bad),'annotation_join_errors':len(bad_events),
            'temporal_alignment':'Inclusive master frames at30000/1001fps; per-event bounds checked' if is_asl else '451 new EAF millisecond segments; older utterance-only labels/uncalibrated individual boundaries',
            'visual_alignment':'UNKNOWN: not independently visually reviewed','signer_independent_split':'BLOCKED: incomplete stable signer IDs',
            'merge_readiness':'BLOCKED_PENDING_VERSIONED_LABEL_SCOPE_MAPPING_AND_REVIEW','merge_reason':'Shared378features do not make dominant-occurrence and older main-gloss annotation semantics identical; separate vocab IDs must be rebuilt in any future version.'})
        check(name+'_internal_contract',not ctc_bad and not bad_events,{'utterances':len(rows),'ctc_invalid':ctc_bad,'annotation_errors':bad_events,'direct_csv_event_joins':event_joins,'publisher_bound_checks':time_checks},'Publisher-coordinate consistency is not independent linguistic/visual alignment.')
    a,b=(set(details[k]['label_counts']) for k in sorted(details))
    write('pool_details.json',{'pools':details,'literal_label_intersection':len(a&b),'literal_union':len(a|b),'label_mapping_or_merging_performed':False})
    csv_write('label_pool_compatibility.csv',out);csv_write('ctc_lengths.csv',lengths)
    return details

def main():
    started=time.perf_counter();seeds(20261020)
    before=frozen_snapshot()
    write('preservation_before.json',{'files':before,'scope':'Metadata of all indexed expand03-09 artifacts, inherited protected metadata and shared source; archives not decoded or re-extracted.'})
    unit_checks();shared=function_hashes();data={};rows={};summaries={};all_predictions=[];equivalence=[];norm_results={}
    canonical=read(SIX/'canonical/manifest.json')
    # Only eligible TRAIN and fixed DEV records are addressed. No reserved-test manifest, features, annotations or predictions are opened.
    pool_details=pool_audit(canonical,data)
    del canonical
    common_dev=None;frozen_models={};states={};vocabs={};logits_saved={};arrays={key:v[0] for key,v in data.items()}
    for name,run in RUNS.items():
        manifest=read(run/'data/processed/train_dev/manifest.json');fit=[r for r in manifest if r['split']=='train'];dev=[r for r in manifest if r['split']=='dev'];rows[name]=fit
        for r in manifest:get_feature(r,data)
        arrays={key:v[0] for key,v in data.items()}
        identity=[(r['sample_id'],r['glosses'],r['feature_sha256']) for r in dev]
        if common_dev is None:common_dev=identity
        check(name+'_fixed_dev_identity',identity==common_dev,{'dev_ids':[r['sample_id'] for r in dev],'tokens':sum(len(r['glosses']) for r in dev)})
        model,state,vocab,pointer=load_model(run);frozen_models[name]=model;states[name]=state;vocabs[name]=vocab
        expected=['<BLANK>','<UNK>']+sorted({t for r in fit for t in r['glosses']})
        check(name+'_vocab_ids',vocab==expected and len(vocab)==len(set(vocab)),{'vocab_size':len(vocab),'blank_token':vocab[0],'unknown_token':vocab[1]})
        check(name+'_checkpoint_manifest',state['manifest_sha256']==sha(run/'data/processed/train_dev/manifest.json'),{'checkpoint_epoch':state['epoch'],'manifest_sha256':state['manifest_sha256']})
        with (run/'models/reference48/training_log.csv').open(newline='') as stream:logs=list(csv.DictReader(stream))
        selection=min(logs,key=lambda r:(float(r['dev_wer']),float(r['dev_loss'])))
        check(name+'_selection',int(selection['epoch'])==pointer['epoch'],{'selected_epoch':pointer['epoch'],'argmin_log_epoch':int(selection['epoch']),'epoch_wer':[float(r['dev_wer']) for r in logs]})
        saved=read(run/'models/reference48/dev_best.json');result=evaluate(model,dev,arrays,vocab,'cpu')
        check(name+'_reproduce_saved_dev',result['predictions']==saved['predictions'],{'wer':result['gloss_wer'],'S':result['substitutions'],'D':result['deletions'],'I':result['insertions'],'saved_loss':saved['loss'],'cpu_loss':result['loss']},'Float loss can differ slightly between CUDA/CPU while literal hypotheses and edit counts match.')
        values=np.concatenate([arrays[r['sample_id']] for r in fit]).astype(np.float64)
        mean=values.mean(0).astype(np.float32);std=values.std(0);std=np.where(std<1e-6,1.,std).astype(np.float32);del values
        check(name+'_train_only_normalization',np.array_equal(mean,state['model']['mean'].numpy()) and np.array_equal(std,state['model']['std'].numpy()),{'mean_max_abs_error':float(np.abs(mean-state['model']['mean'].numpy()).max()),'std_max_abs_error':float(np.abs(std-state['model']['std'].numpy()).max())})
        counts=Counter(t for r in fit for t in r['glosses']);distribution={}
        for population,items in [('train',fit),('dev',dev)]:
            sums=np.zeros(3);n=0;clipped=0;size=0
            for row in items:
                x=arrays[row['sample_id']];sums+=x[:,[172,173,214]].sum(0);n+=len(x);clipped+=int((np.abs((x-mean)/std)>10).sum());size+=x.size
            distribution[population]={'observations':n,'observed_left_right_body':(sums/n).tolist(),'clipped_feature_fraction':clipped/size}
        norm_results[name]=distribution
        leakage={}
        for key in ['sample_id','collection','source_group','feature_sha256']:
            leakage[key]=sorted({str(r[key]) for r in fit if key in r}&{str(r[key]) for r in dev if key in r})
        leakage['raw_sha256']=sorted({r.get('raw_sha256',r.get('source_video_sha256')) for r in fit}&{r.get('raw_sha256',r.get('source_video_sha256')) for r in dev}-{None})
        leakage['same_literal_gloss_sequences']=[r['sample_id'] for r in dev if tuple(r['glosses']) in {tuple(t['glosses']) for t in fit}]
        check(name+'_known_train_dev_leakage',not any(leakage[k] for k in leakage if k!='same_literal_gloss_sequences'),leakage,'No exhaustive perceptual duplicate detector; unknown signer IDs prevent signer-leakage certification. Repeated linguistic prompts are not themselves identical media.')
        logits_saved[name]={};blank_n=0;observations=0;empty=0;beam_counts=Counter();beam_pairs=[]
        with torch.inference_mode():
            for row in dev:
                x=arrays[row['sample_id']];z=model(torch.as_tensor(x)[None])[0];logp=z.log_softmax(-1).numpy();ids=z.argmax(-1).numpy();hyp=decode(z,vocab)
                assert len(z)==len(x)
                logits_saved[name][row['sample_id']]=z.numpy()
                independent=[vocab[t] for t in collapse(ids.tolist())]
                assert independent==hyp and decode(z.log_softmax(-1),vocab)==hyp
                edits=edit_counts(row['glosses'],hyp)
                assert sum(edits[k] for k in ('substitutions','deletions','insertions'))==levenshtein(row['glosses'],hyp)
                blank_n+=int((ids==0).sum());observations+=len(ids);empty+=not hyp
                nonblank=z[:,1:].max(-1).values.numpy();margin=z[:,0].numpy()-nonblank
                beam_ids,beam_logp=prefix_beam(logp,width=10,top=16);beam_hyp=[vocab[t] for t in beam_ids];beam_edits=edit_counts(row['glosses'],beam_hyp);beam_counts.update(beam_edits)
                beam_pairs.append({'sample_id':row['sample_id'],'reference':row['glosses'],'greedy':hyp,'fixed_beam10_top16':beam_hyp,**beam_edits,'beam_log_probability':beam_logp})
                all_predictions.append({'status':'PASS','run':name,'epoch':pointer['epoch'],'sample_id':row['sample_id'],'collection':row['collection'],'source_group':row['source_group'],'signer_id':row.get('signer_id'),
                    'reference':row['glosses'],'hypothesis':hyp,'hypothesis_display':' '.join(hyp) if hyp else '<EMPTY>','input_observations':len(x),'output_timesteps':len(z),
                    'blank_argmax_fraction':float((ids==0).mean()),'mean_blank_probability':float(np.exp(logp[:,0]).mean()),'min_blank_logit_margin':float(margin.min()),
                    'oov_reference_labels':[t for t in row['glosses'] if t not in vocab],'reference_token_ids_for_ctc_loss':[vocab.index(t) if t in vocab else 1 for t in row['glosses']],
                    'raw_argmax_ids':ids.tolist(),'best_nonblank_labels':dict(Counter(vocab[i+1] for i in z[:,1:].argmax(-1).tolist())),**edits})
            # Causal right padding cannot alter earlier logits; test actual frozen weights.
            short=min(dev,key=lambda r:len(arrays[r['sample_id']]));long=max(dev,key=lambda r:len(arrays[r['sample_id']]))
            xb,_,_,_=batch([short,long],arrays,vocab,'cpu');batched=model(xb)[0,:len(arrays[short['sample_id']])];single=torch.as_tensor(logits_saved[name][short['sample_id']])
            prefix_n=max(1,len(single)//2);prefix=model(torch.as_tensor(arrays[short['sample_id']][:prefix_n])[None])[0]
            check(name+'_padding_and_no_truncation',torch.allclose(batched,single,atol=2e-5,rtol=1e-5) and torch.allclose(prefix,single[:prefix_n],atol=2e-5,rtol=1e-5),
                {'padding_max_logit_error':float((batched-single).abs().max()),'prefix_max_logit_error':float((prefix-single[:prefix_n]).abs().max()),'all_dev_input_output_lengths_equal':True},'Causal stride1 TCN retains T; cadence sampling is upstream, not neural downsampling.')
        write(name+'_dev_logits.json',{'schema':SCHEMA,'vocabulary':vocab,'logits':{k:v.tolist() for k,v in logits_saved[name].items()},'scope':'DEV diagnostic only; no threshold or checkpoint tuning.'})
        beam_wer=sum(beam_counts[k] for k in ('substitutions','deletions','insertions'))/beam_counts['reference_tokens']
        write(name+'_decoder_counterfactual.json',{'predeclared':'Change decoder only: prefix beam10/top16, blank0, no LM, no penalty or parameter sweep. Pruned approximation, not new official metrics.','gloss_wer':beam_wer,'counts':dict(beam_counts),'pairs':beam_pairs})
        summaries[name]={'train':len(fit),'dev':len(dev),'selected_epoch':pointer['epoch'],'completed_epochs':len(logs),'dev':{k:v for k,v in result.items() if k!='predictions'},
            'blank_argmax_fraction':blank_n/observations,'empty_hypotheses':empty,'timesteps':observations,'beam10_top16_wer':beam_wer,'checkpoint_sha256':pointer['sha256'],
            'vocab_size':len(vocab),'unique_train_labels':len(counts),'singleton_labels':sum(n==1 for n in counts.values()),'target_tokens':sum(counts.values()),
            'oov_counts':dict(Counter(t for r in dev for t in r['glosses'] if t not in vocab)),'known_dev_signers':sorted({str(r['signer_id']) for r in dev if r.get('signer_id') is not None}),
            'dev_collections':sorted({r['collection'] for r in dev}),'dev_source_groups':sorted({r['source_group'] for r in dev}),
            'config':state['config'],'shared_function_ast_sha256':shared,'leakage':leakage}
        for dimension,status,detail in [('DEV_ID_reference_feature_order','PASS','Exactly identical original nine IDs, literal labels and feature SHA256s.'),('split_policy','PARTIAL','Collection-disjoint custom research split; no verified signer-independent split.'),('decoder_and_metric','PASS','Same imported greedyCTC blank0 decoder and literal-token micro WER; checkpoint predictions independently reproduced.'),('vocabulary','PARTIAL',f'TRAIN-fitted {len(vocab)} tokens; literal label identities are compared, numeric IDs are model-specific; OOV={result["out_of_vocabulary_tokens"]}/57.'),('normalization','PARTIAL','Same formula, each different TRAIN pool has different fitted buffers; exact recomputation passes.'),('training_data_and_budget','PARTIAL',f'{len(fit)} TRAIN, {len(logs)} epochs, selected{pointer["epoch"]}; data/budget differ, so this is not an architecture-only controlled comparison.')]:
            equivalence.append({'status':status,'run':name,'dimension':dimension,'evidence':detail})
    csv_write('prediction_audit.csv',all_predictions);csv_write('comparison_equivalence.csv',equivalence)
    raw_parity(rows,data)
    # Verify previously reported TRAIN WER using CPU inference of exactly the frozen
    # latest epoch, then the DEV-selected epoch. No fit or checkpoint selection.
    main_run=RUNS['main2119'];fit=rows['main2119'];last_model,last_state,vocab,last_pointer=load_model(main_run,'latest')
    train_results={}
    for name,model,epoch,path in [('last_epoch',last_model,last_pointer['epoch'],main_run/'last_checkpoint_train_diagnostic.json'),('selected_epoch',frozen_models['main2119'],states['main2119']['epoch'],main_run/'selected_checkpoint_train_diagnostic.json')]:
        tick=time.perf_counter();value=evaluate(model,fit,arrays,vocab,'cpu');saved=read(path)['result']
        check('main_'+name+'_train_wer',value['predictions']==saved['predictions'] and value['gloss_wer']==saved['gloss_wer'],{'epoch':epoch,'wer':value['gloss_wer'],'saved_wer':saved['gloss_wer'],'reference_tokens':value['reference_tokens'],'S':value['substitutions'],'D':value['deletions'],'I':value['insertions']})
        train_results[name]={'epoch':epoch,'seconds':time.perf_counter()-tick,'scores':{k:v for k,v in value.items() if k!='predictions'}}
    # Literal pool compatibility is not inferred from feature width or a few labels.
    excluded=set(read(SIX/'asllrp2025_catalog/annotation_boundary_audit.v2.json')['reject_sample_ids'])
    eligible_ids={r['sample_id'] for r in rows['main2119']+rows['pilot906']}
    check('six_overlap_exclusions_preserved',len(excluded)==6 and not excluded&eligible_ids,{'excluded_ids':sorted(excluded),'eligible_union':len(eligible_ids)})
    quality=read(SIX/'canonical/quality.json')
    check('frame_mismatch_exclusion_preserved',quality['rejection_counts'].get('PUBLISHER_INCLUSIVE_FRAME_SPAN_MISMATCH:expected=100:actual=99')==1 and 'asllrp3_8d6cbe4ca1fc7986' not in eligible_ids,quality['rejection_counts'])
    after=frozen_snapshot();changed=[p for p,meta in before.items() if after.get(p)!=meta]
    check('frozen_files_preserved',not changed,{'checked_files':len(before),'changed':changed})
    write('preservation_after.json',{'checked_files':len(before),'changed':changed,'status':'PASS' if not changed else 'PARTIAL','training_performed':False,'reserved_test_opened_or_scored':False,'acquisition_performed':False,'production_modified':False})
    write('checks.json',CHECKS)
    write('audit_results.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'runs':summaries,'normalization_and_feature_distributions':norm_results,'main_train_reproduction':train_results,
        'shared_function_ast_sha256':shared,'pool_summary':{k:{a:b for a,b in v.items() if a not in ('label_counts','collections')} for k,v in pool_details.items()},
        'elapsed_seconds':time.perf_counter()-started,'training_performed':False,'test_used':False,'status':'PASS_DIAGNOSTIC_EXECUTION' if all(c['passed'] for c in CHECKS) else 'PARTIAL_REQUIRES_REVIEW'})
    write('input_artifacts.json',INPUTS)
    print(json.dumps({'status':'DIAGNOSTIC_COMPLETE','checks':len(CHECKS),'nonpasses':[c['finding'] for c in CHECKS if not c['passed']],
        'runs':{k:{a:b for a,b in v.items() if a in ('selected_epoch','empty_hypotheses','blank_argmax_fraction','beam10_top16_wer','oov_counts')} for k,v in summaries.items()},'train_reproduction':train_results,'seconds':time.perf_counter()-started}),flush=True)

if __name__=='__main__':main()
