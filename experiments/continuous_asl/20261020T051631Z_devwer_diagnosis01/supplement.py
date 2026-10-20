"""Bounded annotation joins, numeric CTC contract and final-epoch DEV replay."""
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
import torch
from audit import HERE,ROOT,SIX,RUNS,read,sha,write,load_model,get_feature,INPUTS,collapse
from ctc_experiment import evaluate,seeds

def main():
    seeds(20261020);started=time.perf_counter();results=[]
    rng=np.random.default_rng(20261020);p=rng.uniform(.05,1,(4,3));p=p/p.sum(1,keepdims=True)
    for target in [(2,),(2,2),(1,2)]:
        probability=sum(np.prod([p[t,v] for t,v in enumerate(path)]) for path in itertools.product(range(3),repeat=4) if collapse(path)==target)
        loss=torch.nn.CTCLoss(blank=0,reduction='sum',zero_infinity=False)(torch.tensor(np.log(p),dtype=torch.float64)[:,None,:],torch.tensor(target),torch.tensor([4]),torch.tensor([len(target)]))
        error=abs(float(loss)+np.log(probability))
        results.append({'status':'PASS' if error<1e-10 else 'PARTIAL','test':'synthetic_CTC_path_sum','target_ids':target,'pytorch_loss':float(loss),'exact_enumerated_loss':-np.log(probability),'absolute_error':error,'scope':'Synthetic test, not model evaluation; no optimizer/backward.'})
    plan=read(SIX/'depaul_catalog/plan.json');fit=read(RUNS['pilot906']/'data/processed/train_dev/manifest.json')
    fit=[r for r in fit if r['split']=='train'];dev=[r for r in read(RUNS['main2119']/'data/processed/train_dev/manifest.json') if r['split']=='dev']
    source_rows={r['sample_id']:r for r in plan['rows']}
    timed=[r for r in fit if 'sign_times_ms' in r];targeted=timed+[source_rows[r['sample_id']] for r in dev if r['sample_id'] in source_rows]
    parsed={};member_hashes={};joins=[];participant_attrs=[]
    archive_path=SIX/'data/raw/depaul/elanBUcorpus.zip'
    # Targeted label/boundary reads only. Do not call testzip, re-download or decode media.
    with zipfile.ZipFile(archive_path) as archive:
        for row in targeted:
            prov=row['annotation_provenance'];member=prov['member']
            if member not in parsed:
                content=archive.read(member);digest=hashlib.sha256(content).hexdigest();member_hashes[member]=digest
                root=ET.fromstring(content);slots={s.attrib['TIME_SLOT_ID']:int(s.attrib['TIME_VALUE']) for s in root.find('TIME_ORDER')}
                events={};tiers={}
                for tier in root.findall('TIER'):
                    attrs=tier.attrib
                    participant_attrs.extend({k:v for k,v in attrs.items() if k in ('PARTICIPANT','ANNOTATOR')} for _ in [0] if 'PARTICIPANT' in attrs or 'ANNOTATOR' in attrs)
                    current=[]
                    for annotation in tier.findall('ANNOTATION/ALIGNABLE_ANNOTATION'):
                        event={'id':annotation.attrib['ANNOTATION_ID'],'start':slots[annotation.attrib['TIME_SLOT_REF1']],'end':slots[annotation.attrib['TIME_SLOT_REF2']],'tier':attrs['TIER_ID']}
                        # Read gloss target text only; English tier supplies boundaries, never labels.
                        if attrs['TIER_ID']=='main gloss':event['label']=annotation.findtext('ANNOTATION_VALUE')
                        events[event['id']]=event;current.append(event)
                    tiers[attrs['TIER_ID']]=current
                parsed[member]=(root,events,tiers)
            root,events,tiers=parsed[member]
            selected=[events[i] for i in row['annotation_event_ids']]
            span=events[row['utterance_span_event_id']]
            ok=(member_hashes[member]==prov['member_sha256'] and [e.get('label') for e in selected]==row['glosses'] and
                all(e['tier']=='main gloss' for e in selected) and span['tier']=='English translation' and
                span['start']==row['start_ms'] and span['end']==row['end_ms'] and
                [(e['start'],e['end']) for e in selected]==[tuple(v) for v in row['sign_times_ms']])
            joins.append({'sample_id':row['sample_id'],'partition':'fixed_DEV' if row['sample_id'] in {r['sample_id'] for r in dev} else 'TRAIN',
                'member':member,'status':'PASS' if ok else 'PARTIAL','labels_exactly_preserved':bool([e.get('label') for e in selected]==row['glosses']),
                'member_sha256':member_hashes[member],'publisher_time_units':root.find('HEADER').attrib.get('TIME_UNITS')})
    # Verify all 906 frozen TRAIN and nine DEV targets against their original public index row.
    source_index=Path(read(RUNS['baseline']/'dataset_plan.json')['index_path']);sha(source_index)
    with source_index.open(encoding='utf-8-sig',newline='') as stream:
        index=list(csv.DictReader(stream))
    lookup=defaultdict(list)
    for row in index:
        if row['Perspective/Camera id']=='0':lookup[(row['XML file'],row['Video sequence id'],row['main gloss'])].append(row)
    index_joins=[]
    for row in fit+dev:
        key=(row['collection'],str(row['source_group']),' '.join(row['glosses']))
        found=lookup[key]
        index_joins.append({'sample_id':row['sample_id'],'status':'PASS' if len(found)==1 else 'PARTIAL','exact_frontal_collection_media_gloss_matches':len(found)})
    write('annotation_join_checks.json',{'status':'PASS' if all(r['status']=='PASS' for r in joins+index_joins) else 'PARTIAL',
        'EAF_member_count':len(parsed),'EAF_utterance_joins':len(joins),'EAF_joins':joins,'literal_index_joins':index_joins,
        'participant_or_annotator_tier_attributes':participant_attrs,'stable_DEV_signer_id_recovered':False,
        'scope':'Targeted archived annotation members only, no full archive re-verification, no reserved TEST access. EAF authors/annotators are not assumed to be signers.'})
    data={}
    for row in dev:get_feature(row,data)
    arrays={key:v[0] for key,v in data.items()}
    model,state,vocab,pointer=load_model(RUNS['main2119'],'latest');value=evaluate(model,dev,arrays,vocab,'cpu')
    with (RUNS['main2119']/'models/reference48/training_log.csv').open(newline='') as stream:
        log=list(csv.DictReader(stream))[-1]
    parity=all(value[k]==int(log[k]) for k in ('substitutions','deletions','insertions','reference_tokens')) and value['gloss_wer']==float(log['dev_wer'])
    results.append({'status':'PASS' if parity else 'PARTIAL','test':'latest_epoch_DEV_replay','epoch':pointer['epoch'],'checkpoint_sha256':pointer['sha256'],'log_edit_count_parity':parity})
    write('latest_epoch_dev_replay.json',{'epoch':pointer['epoch'],'checkpoint_sha256':pointer['sha256'],'scores':value,
        'scope':'Diagnostic replay only of already logged epoch17 DEV. Does not select another checkpoint or alter frozen metrics. No training/TEST.'})
    unique_dev=Counter(t for r in dev for t in r['glosses'])
    write('supplement_results.json',{'status':'PASS' if all(r['status']=='PASS' for r in results+joins+index_joins) else 'PARTIAL',
        'tests':results,'EAF_joins':len(joins),'EAF_failures':[r for r in joins if r['status']!='PASS'],'index_joins':len(index_joins),
        'index_failures':[r for r in index_joins if r['status']!='PASS'],'DEV_unique_literal_labels':len(unique_dev),'DEV_label_counts':dict(unique_dev),
        'DEV_metadata_signer_ids_available':sum(r.get('signer_id') is not None for r in dev),'EAF_DEV_participant_attributes_available':bool(participant_attrs),
        'elapsed_seconds':time.perf_counter()-started,'training_performed':False})
    write('supplement_input_artifacts.json',INPUTS)
    print(json.dumps({'status':'SUPPLEMENT_COMPLETE','test_results':results,'eaf_joins':len(joins),'eaf_failures':sum(r['status']!='PASS' for r in joins),
        'index_joins':len(index_joins),'index_failures':sum(r['status']!='PASS' for r in index_joins),'latest_DEV_wer':value['gloss_wer'],
        'latest_DEV_empty_hypotheses':sum(not p['hypothesis'] for p in value['predictions'])}),flush=True)

if __name__=='__main__':main()
