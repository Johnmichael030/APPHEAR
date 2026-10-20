"""Saved-prediction-only CTC diagnosis. No torch/model, feature reconstruction or TEST access."""
from pathlib import Path
from collections import Counter, defaultdict
import csv, hashlib, json, math, re, statistics, time

RUN=Path(__file__).resolve().parent
OLD=RUN.parent/'20261020T143609Z_learning_diagnostic01'
META=RUN.parent/'20261020T135119Z_read_only_failure_audit01'/'feature_and_target_checks.json'
ROOT=RUN.parents[2]
INPUTS={}
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def track(p):
    p=Path(p);s=p.stat();v=dict(sha256=digest(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    if str(p) in INPUTS: assert INPUTS[str(p)]==v
    INPUTS[str(p)]=v
    return p
def read(p): return json.loads(track(p).read_text(encoding='utf-8-sig'))
def save(name,value):
    with (RUN/name).open('x',encoding='utf-8') as f: json.dump(value,f,indent=2);f.write('\n')
def export(name,rows):
    assert rows
    with (RUN/name).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def alignment(ref,hyp):
    # Same DP and traceback ties as tools/continuous_asl/ctc_experiment.py: diagonal, delete, insert.
    n,m=len(ref),len(hyp);d=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):d[i][0]=i
    for j in range(m+1):d[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):d[i][j]=min(d[i-1][j-1]+(ref[i-1]!=hyp[j-1]),d[i-1][j]+1,d[i][j-1]+1)
    i,j=n,m;out=[]
    while i or j:
        if i and j and d[i][j]==d[i-1][j-1]+(ref[i-1]!=hyp[j-1]):
            out.append(('match' if ref[i-1]==hyp[j-1] else 'substitution',ref[i-1],hyp[j-1]));i-=1;j-=1
        elif i and d[i][j]==d[i-1][j]+1:out.append(('deletion',ref[i-1],None));i-=1
        else:out.append(('insertion',None,hyp[j-1]));j-=1
    return list(reversed(out))
def band(n):return '1' if n==1 else '2-4' if n<5 else '5-9' if n<10 else '10-19' if n<20 else '20+'
def quantiles(v):
    v=sorted(v);return dict(min=min(v),p25=v[round(.25*(len(v)-1))],median=statistics.median(v),p75=v[round(.75*(len(v)-1))],max=max(v))
def family(r):
    m=re.match(r'^(\d{4}-\d{2}-\d{2})_',r['source_group'])
    return r['signer_id']+'::'+(m[1] if m else 'UNRESOLVED_ALL_UNDATED_MASTERS')

tick=time.perf_counter()
track(Path(__file__))
for name in ['MISSION_REPORT.md','CONTROL_REVIEW.md','run.log','artifact_hashes.sha256','run_mission.py']:
    track(OLD/name)
cfg=read(OLD/'config.json');decision=read(OLD/'decision.json');fold=read(OLD/'fold_manifest.json')
vocab=read(OLD/'vocabulary.json')['tokens'];track(OLD/'normalization.npz')
track(ROOT/'tools/continuous_asl/ctc_experiment.py')
resolved=read(OLD/'resolved_manifest.json');lookup={r['sample_id']:r for r in resolved}
trainids=fold['train_sample_ids'];devids=fold['compatible_eval_sample_ids']
assert len(trainids)==len(set(trainids))==1500 and len(devids)==len(set(devids))==80
assert not set(trainids)&set(devids)
tr=[lookup[i] for i in trainids];dv=[lookup[i] for i in devids]
assert all(r['internal_role']=='TRAIN' for r in tr)
assert all(r['internal_role']=='HELD_OUT_COMPATIBLE' for r in dv)
assert not {family(r) for r in tr}&{family(r) for r in dv}
support=Counter(t for r in tr for t in set(r['glosses']))
freq=Counter(t for r in tr for t in r['glosses'])
assert vocab==['<BLANK>','<UNK>']+sorted(support)
assert digest(OLD/'vocabulary.json')==cfg['vocabulary_sha256']
assert digest(OLD/'normalization.npz')==cfg['normalization_sha256']
assert digest(ROOT/'tools/continuous_asl/ctc_experiment.py')==cfg['source_model_decoder_sha256']
for rows in [tr,dv]:assert all(t in support and vocab.index(t)>=2 for r in rows for t in r['glosses'])
cache={r['sample_id']:r for r in read(META)}
selected={'TRAIN':read(OLD/'selected_train1500.json'),'HELDOUT80':read(OLD/'selected_primary80.json')}
assert {p['sample_id'] for p in selected['TRAIN']['predictions']}==set(trainids)
assert {p['sample_id'] for p in selected['HELDOUT80']['predictions']}==set(devids)
with track(OLD/'training_metrics.csv').open(encoding='utf-8-sig',newline='') as f:metrics=list(csv.DictReader(f))
with track(OLD/'prediction_examples.csv').open(encoding='utf-8-sig',newline='') as f:examples=list(csv.DictReader(f))
exindex={(r['stage'],r['sample_id']):r for r in examples}
index={}
for line in (OLD/'artifact_hashes.sha256').read_text(encoding='utf-8-sig').splitlines():
    if line.strip():h,p=line.split('  ',1);index[p.replace('\\','/')]=h
verified=[]
for p,v in list(INPUTS.items()):
    pp=Path(p)
    if pp.is_relative_to(OLD):
        rel=pp.relative_to(OLD).as_posix()
        if rel in index:assert v['sha256']==index[rel],rel;verified.append(rel)

perlabel={t:dict(TRAIN=Counter(),HELDOUT80=Counter()) for t in support}
confusions={s:defaultdict(Counter) for s in selected}
hypfreq={s:Counter() for s in selected}
utterance=[];strata={s:defaultdict(Counter) for s in selected}
associations={s:defaultdict(Counter) for s in selected}
source_coverage={}
for scope,data in selected.items():
    counts=Counter();refs=Counter();hyps=Counter();frames=blank=0
    for p in data['predictions']:
        sid=p['sample_id'];r=lookup[sid];cm=cache[sid];ref=p['reference'];hyp=p['hypothesis']
        assert ref==r['glosses'] and p['output_frames']==cm['observations']==r['observations']
        assert all(t in vocab for t in hyp)
        a=alignment(ref,hyp);ct=Counter(x[0] for x in a)
        assert all(ct[k[:-1]]==p[k] for k in ['substitutions','deletions','insertions'])
        assert p['reference_tokens']==len(ref)
        er=exindex[('candidate_TRAIN' if scope=='TRAIN' else 'candidate_primary80',sid)]
        assert json.loads(er['reference'])==ref and json.loads(er['hypothesis'])==hyp
        repeats=sum(x==y for x,y in zip(ref,ref[1:]));minimum=len(ref)+repeats
        assert p['output_frames']>=minimum
        assert cm['feature_transcript_source_join']=='PASS' and cm['cached_raw_feature_recomputation']=='EXACT'
        assert cm['CTC_minimum_steps']==minimum
        for op,t,h in a:
            if t is not None:
                perlabel[t][scope][op]+=1;strata[scope][band(support[t])][op]+=1
            if op=='substitution':confusions[scope][t][h]+=1
        errors=ct['substitution']+ct['deletion']+ct['insertion']
        refs.update(ref);hyps.update(hyp);hypfreq[scope].update(hyp)
        frames+=p['output_frames'];blank+=p['blank_frames']
        for k in ['substitutions','deletions','insertions','reference_tokens']:counts[k]+=p[k]
        flag='event_missing_frame_or_no_hand' if cm['missing_event_observation_count'] or cm['sampled_events_no_hand_count'] else 'neither_event_flag'
        associations[scope][flag].update(utterances=1,errors=errors,reference_tokens=len(ref),hypothesis_tokens=len(hyp))
        utterance.append(dict(scope=scope,sample_id=sid,signer_id=r['signer_id'],source_group=r['source_group'],recording_family=family(r),
            reference_json=json.dumps(ref,ensure_ascii=False),hypothesis_json=json.dumps(hyp,ensure_ascii=False),
            substitutions=ct['substitution'],deletions=ct['deletion'],insertions=ct['insertion'],exact_matches=ct['match'],
            WER=errors/len(ref),reference_tokens=len(ref),hypothesis_tokens=len(hyp),prediction_reference_length_ratio=len(hyp)/len(ref),
            CTC_loss=p['CTC_loss'],argmax_blank_fraction=p['blank_frames']/p['output_frames'],blank_frames=p['blank_frames'],input_frames=p['output_frames'],
            adjacent_repeated_targets=repeats,CTC_minimum_steps=minimum,CTC_slack=p['output_frames']-minimum,
            train_support_min=min(support[t] for t in ref),train_support_median=statistics.median(support[t] for t in ref),
            missing_frame_event_count=cm['missing_event_observation_count'],sampled_no_hand_event_count=cm['sampled_events_no_hand_count'],
            observed_any_hand_fraction=cm['observed_any_hand_fraction'],body_available_fraction=cm['body_available_fraction'],
            normalization_clip_fraction=cm['normalization_clip_fraction'],posterior_blank_probability='UNKNOWN_NOT_SAVED'))
    assert all(counts[k]==data[k] for k in counts)
    assert math.isclose(blank/frames,data['blank_fraction'],abs_tol=1e-12)
    scope_rows=[r for r in tr if scope=='TRAIN'] if scope=='TRAIN' else dv
    source_coverage[scope]=dict(utterances=len(data['predictions']),reference_tokens=sum(refs.values()),hypothesis_tokens=sum(hyps.values()),
        prediction_reference_length_ratio=sum(hyps.values())/sum(refs.values()),argmax_blank_fraction=blank/frames,
        output_frames=frames,blank_frames=blank,nonblank_frames=frames-blank,unique_reference_labels=len(refs),unique_predicted_labels=len(hyps),
        reference_labels_never_predicted=len(set(refs)-set(hyps)),empty_outputs=sum(not p['hypothesis'] for p in data['predictions']),
        top_reference_tokens=refs.most_common(15),top_hypothesis_tokens=hyps.most_common(15),
        signers=sorted({r['signer_id'] for r in scope_rows}),masters=len({r['source_group'] for r in scope_rows}),families=sorted({family(r) for r in scope_rows}),
        source_counts=dict(Counter(r['source_group'] for r in scope_rows)),signer_counts=dict(Counter(r['signer_id'] for r in scope_rows)),
        WER=data['WER'],CTC_loss=data['CTC_loss'],**{k:v for k,v in counts.items() if k!='reference_tokens'})

labelrows=[]
devrefs=Counter(t for r in dv for t in r['glosses'])
devutter=Counter(t for r in dv for t in set(r['glosses']))
for t in sorted(support):
    x=dict(literal_label=t,vocabulary_id=vocab.index(t),TRAIN_distinct_utterance_support=support[t],TRAIN_token_frequency=freq[t],
        support_band=band(support[t]),HELDOUT80_distinct_reference_utterances=devutter[t],HELDOUT80_reference_tokens=devrefs[t])
    for s in selected:
        c=perlabel[t][s];den=sum(c.values())
        x.update({f'{s}_aligned_reference_tokens':den,f'{s}_exact_matches':c['match'],f'{s}_substitutions':c['substitution'],f'{s}_deletions':c['deletion'],
            f'{s}_aligned_exact_token_recall':c['match']/den if den else '',f'{s}_predicted_token_frequency':hypfreq[s][t],
            f'{s}_substitution_destinations_json':json.dumps(confusions[s][t].most_common(),ensure_ascii=False)})
    labelrows.append(x)
export('diagnosis_by_label.csv',labelrows);export('diagnosis_by_utterance.csv',utterance)

trajectory=[]
for m in metrics:
    e=int(m['epoch']);entry=dict(epoch=e,TRAIN_WER=float(m['TRAIN_WER']),TRAIN_CTC_loss=float(m['TRAIN_CTC_loss']),TRAIN_blank_fraction=float(m['TRAIN_blank_fraction']),
        TRAIN_deletions=int(m['TRAIN_deletions']),TRAIN_empty_outputs=int(m['TRAIN_empty_outputs']),TRAIN_token_diversity=int(m['TRAIN_token_diversity']))
    for s,filename in [('TRAIN',f'train1500_epoch_{e:03d}.json'),('HELDOUT80',f'primary80_epoch_{e:03d}.json')]:
        p=OLD/'candidate_predictions'/filename
        if not p.exists():continue
        a=read(p);pred=a['predictions'];rt=sum(len(x['reference']) for x in pred);ht=sum(len(x['hypothesis']) for x in pred)
        entry[s+'_prediction_reference_length_ratio']=ht/rt
        if s=='HELDOUT80':
            entry.update(HELDOUT80_WER=a['WER'],HELDOUT80_CTC_loss=a['CTC_loss'],HELDOUT80_blank_fraction=a['blank_fraction'],
                HELDOUT80_deletions=a['deletions'],HELDOUT80_token_diversity=a['token_diversity'])
    trajectory.append(entry)

stratout={}
for s,ss in strata.items():
    stratout[s]={}
    for b,c in ss.items():
        den=sum(c.values());stratout[s][b]=dict(reference_tokens=den,matches=c['match'],substitutions=c['substitution'],deletions=c['deletion'],recall=c['match']/den)
assocout={s:{g:dict(**dict(c),WER=c['errors']/c['reference_tokens']) for g,c in gs.items()} for s,gs in associations.items()}
summary=dict(status='PASS_NUMERICAL_VALIDITY_PARTIAL_CAUSAL_EXPLANATION',pipeline_valid_for_compact_control=True,
    posterior_saved=False,
    posterior_probability_status='UNKNOWN: full posteriors/logits were not saved; blank numbers are framewise argmax rates only',
    alignment='Exact-token Levenshtein, deterministic diagonal(match/substitute), deletion, insertion traceback; recall/confusions alignment-dependent',
    vocabulary=dict(labels=len(support),classes=len(vocab),blank_id=0,UNK_id=1,all_dev_labels_seen_in_train=True,support_distribution=dict(Counter(band(n) for n in support.values()))),
    scope_metrics=source_coverage,support_strata=stratout,observability_associations=assocout,
    CTC=dict(all_feasible=True,input_frames=quantiles([r['input_frames'] for r in utterance]),target_lengths=quantiles([r['reference_tokens'] for r in utterance]),
        slack=quantiles([r['CTC_slack'] for r in utterance]),utterances_with_adjacent_repeats=sum(r['adjacent_repeated_targets']>0 for r in utterance),
        raw_feature_validation='Reused prior exact cached reconstruction/source-join evidence; no raw array reconstruction now'),
    trajectory=trajectory,verified_against_old_artifact_index=verified,
    causal_status='UNKNOWN: sparse labels, limited representation and optimization/capacity remain competing or interacting explanations; these observational strata cannot identify cause',
    no_TEST_access=True,no_model_inference=True,no_training=True,runtime_seconds=time.perf_counter()-tick)
save('diagnosis_summary.json',summary)
text=[]
text.extend(['# Phase 1 saved-prediction CTC diagnosis','',
'PASS — no critical numerical, token-mapping, manifest or evaluation inconsistency found in the scoped saved evidence. A compact TRAIN-only learning control is justified; recognition quality is not accepted.','',
'Confirmed facts:'])
for s,a in source_coverage.items():
    text.append(f"- {s}: WER {100*a['WER']:.4f}%; CTC {a['CTC_loss']:.6f}; S/D/I {a['substitutions']}/{a['deletions']}/{a['insertions']}; argmax blank {100*a['argmax_blank_fraction']:.4f}%; hypothesis/reference tokens {a['hypothesis_tokens']}/{a['reference_tokens']} ({a['prediction_reference_length_ratio']:.4f}); diversity {a['unique_predicted_labels']}; empty {a['empty_outputs']}/{a['utterances']}.")
text.extend(['',f"TRAIN vocabulary: {len(support)} literal glosses plus blank0/UNK1. TRAIN distinct-utterance support bands: {dict(Counter(band(n) for n in support.values()))}. Held-out labels are all represented in TRAIN; unseen labels cannot explain this held-out failure. No pool merge occurred.",
'', 'The dominant selected error is deletion/short output. This is more precise than a single-token collapse: TRAIN predicts 450 distinct labels, held-out 49. Blank posterior probability is UNKNOWN because only greedy argmax frame counts were saved, not logits or probability arrays.',
'','The full-task trajectory is still learning its TRAIN set within the bound: TRAIN WER falls from 100% at epoch1 to 88.40% at epoch12 and 84.61% at epoch13; TRAIN CTC falls from6.591 to3.728. Held-out CTC worsens from6.627 (epoch1) to7.606 (epoch12) despite a small WER change. These facts suggest underlearning plus poor transfer; they do not establish a single cause or justify unbounded continuation.',
'','Per-label aligned recall and support strata are in diagnosis_by_label.csv and diagnosis_summary.json. TRAIN/held-out rare-label correlations are observational. Source families are disjoint, available signers are known, and family/signer structure is confounded. No unseen-signer conclusion is possible.',
'','All1580 references exactly match resolved literal targets and the saved prediction CSV. The vocabulary equals sorted TRAIN literals after two specials; all targets have IDs>=2. Decoder removes blank0 and collapses repeated IDs before label lookup. No model downsampling; every saved input length meets U+adjacent repeats. Saved normalization/vocabulary/decoder hashes match the prior validated configuration. Prior feature schema/finite/strict-timestamp/hash/annotation checks were reused, not recomputed.',
'','Hypotheses: long-tail support, optimization time/blank attractor, the 48-channel representation/capacity, and missing input observability may interact. A representative compact TRAIN control is the smallest authorized discriminator. It cannot by itself establish transfer or causally separate these explanations. Keep architecture, features, normalization, labels, optimizer and decoder unchanged; keep the corrected accumulation tail. No further fit was executed by this phase.',
'','Preservation: no feature reconstruction, checkpoint/model loading, inference, training, acquisition, or reserved TEST access. Only explicit saved TRAIN/primary-held-out inputs were read. Integrity details are in diagnosis_input_hashes.json.'])
with (RUN/'PHASE1_FINDINGS.md').open('x',encoding='utf-8') as f:f.write('\n'.join(text)+'\n')
after={}
for p,before in INPUTS.items():
    pp=Path(p);s=pp.stat();after[p]=dict(sha256=digest(pp),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
assert INPUTS==after
save('diagnosis_input_hashes.json',dict(status='PASS',scope='Explicit saved TRAIN and primary80 metadata/predictions only; no reserved TEST or feature/checkpoint payload reads',before=INPUTS,after=after,changed=[]))
print(json.dumps(dict(status='PASS',pipeline_valid_for_compact_control=True,summary=str(RUN/'diagnosis_summary.json'),runtime_seconds=time.perf_counter()-tick,TRAIN=source_coverage['TRAIN'],HELDOUT80=source_coverage['HELDOUT80']),indent=2))
