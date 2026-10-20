"""Official2025 continuous utterance joins; targets are literal occurrence events.

The publisher CSV describes dominant-hand-realized signs, including two-handed
and hidden events. It omits solely nondominant-hand signs. We retain every CSV
event and explicitly restrict this source's task to that published projection.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import re
from core import RUN, read, write, sha

def main():
    folder=RUN/'asllrp2025_catalog'
    if (folder/'annotation_audit.json').exists():print('ANNOTATION_AUDIT_COMPLETE');return
    path=RUN/'data/raw/asllrp2025/asllrp_sentence_signs_2025_06_28.csv'
    receipt=read(str(path)+'.receipt.json')
    if sha(path)!=receipt['sha256']:raise ValueError('ANNOTATION_HASH')
    fields=['Video ID number','Occurrence label','Start frame of the sign video','End frame of the sign video',
            'Start frame of the containing utterance','End frame of the containing utterance','Utterance video filename',
            'Source collection','Utterance number','Master video filename','Hidden']
    with path.open(encoding='utf-8-sig',newline='') as stream:
        source=list(csv.DictReader(stream))
    if not source or not all(k in source[0] for k in fields):raise ValueError('CSV_SCHEMA')
    groups=defaultdict(list)
    for number,row in enumerate(source,2):
        groups[(row['Source collection'],row['Utterance number'])].append((number,row))
    rows=[];rejected=[]
    for identity,events in groups.items():
        try:
            consistent=['Start frame of the containing utterance','End frame of the containing utterance',
                        'Utterance video filename','Master video filename']
            if any(len({r[k] for _,r in events})!=1 for k in consistent):raise ValueError('INCONSISTENT_MEDIA_OR_UTTERANCE_RANGE')
            first=events[0][1];a=int(first['Start frame of the containing utterance']);b=int(first['End frame of the containing utterance'])
            if a<0 or b<=a:raise ValueError('INVALID_UTTERANCE_RANGE')
            filename=first['Utterance video filename']
            if not re.fullmatch(r'[0-9]+\.mp4',filename):raise ValueError('UNVERIFIED_UTTERANCE_FILENAME')
            sequence=[];seen=set()
            for number,r in events:
                s=int(r['Start frame of the sign video']);e=int(r['End frame of the sign video']);label=r['Occurrence label']
                if not label or label.strip()!=label or '\x00' in label or r['Hidden'] not in ('T','F'):raise ValueError('INVALID_LITERAL_EVENT')
                if not a<=s<=e<=b:raise ValueError('EVENT_OUTSIDE_UTTERANCE')
                if r['Video ID number'] in seen:raise ValueError('DUPLICATE_SIGN_EVENT_ID')
                seen.add(r['Video ID number'])
                sequence.append({'source_row':number,'event_id':r['Video ID number'],'label':label,
                                 'start_frame_master':s,'end_frame_master':e,'hidden':r['Hidden']=='T'})
            sequence.sort(key=lambda r:(r['start_frame_master'],r['end_frame_master'],r['source_row']))
            if len(sequence)<2:raise ValueError('NOT_MULTI_EVENT_CONTINUOUS')
            if any(x['start_frame_master']==y['start_frame_master'] for x,y in zip(sequence,sequence[1:])):
                raise ValueError('SIMULTANEOUS_DOMINANT_EVENTS_NEED_MANUAL_RESOLUTION')
            target=[r['label'] for r in sequence]
            rows.append({'sample_id':'asllrp3_'+hashlib.sha256('|'.join(identity).encode()).hexdigest()[:16],
                 'dataset':'ASLLRP_SIGNSTREAM3_2025','collection':identity[0],'utterance_id':identity[1],
                 'source_group':first['Master video filename'],'signer_id':None,'split':'train',
                 'utterance_filename':filename,'utterance_start_frame_master':a,'utterance_end_frame_master':b,
                 'glosses':target,'events':sequence,'annotation_type':'AUTHENTIC_PUBLISHED_DOMINANT_HAND_OCCURRENCE_EVENTS',
                 'annotation_provenance':{'csv_path':str(path),'csv_sha256':receipt['sha256'],'source_rows':[n for n,_ in events],
                     'target_column':'Occurrence label','order':'verified master frame start; retain hidden events, compounds and full literal labels',
                     'frame_rate_reference':'https://www.bu.edu/asllrp/rpt24/asllrpr24.pdf',
                     'hand_scope_reference':'https://dai.cs.rutgers.edu/asllvd/signbank/Additional-information-6-29-25.pdf',
                     'excluded_nondominant_only_events':'Publisher CSV omits these; never invent missing labels',
                     'English_translation_used_as_gloss':False},
                 'license_status':'PRIVATE_NONCOMMERCIAL_RESEARCH_ONLY_NO_REDISTRIBUTION',
                 'license_reference':'https://www.bu.edu/asllrp/dai-terms.html',
                 'timing_status':'FRAME_RANGES_AWAIT_MATCHING_MEDIA_AND_RATE_VERIFICATION',
                 'signer_independent':False})
        except Exception as e:rejected.append({'collection':identity[0],'utterance_id':identity[1],'reason':str(e)})
    if len({r['utterance_filename'] for r in rows})!=len(rows):raise ValueError('DUPLICATE_MEDIA_FILENAME')
    write(folder/'plan.json',{'rows':rows,'annotation_csv_sha256':receipt['sha256'],
          'literal_label_policy':'Occurrence label unchanged; no cross-version gloss canonicalization. Source-specific dominant-hand realization projection, no nondominant-only hallucination.',
          'split_policy':'New source TRAIN only, original9DEV unchanged, all reserved tests excluded. No source-defined official evaluation partition stated in publisher download documentation.'})
    result={'status':'PASS_ORDERED_DOMINANT_GLOSS_CANDIDATES','csv_rows':len(source),'distinct_source_utterances':len(groups),
            'ordered_candidate_utterances':len(rows),'candidate_media_files':len(rows),'hidden_events_retained':sum(e['hidden'] for r in rows for e in r['events']),
            'rejected':rejected,'label_type':'AUTHENTIC_DOMINANT_HAND_OCCURRENCE_GLOSSES_NOT_ENGLISH_TRANSLATIONS',
            'publisher_source':'https://www.bu.edu/asllrp/ASL-SignBank-and-other-Resources.html',
            'download_directory':'https://dai.cs.rutgers.edu/asllvd/signbank/asllrp_videos/',
            'official_archive_url':'https://dai.cs.rutgers.edu/asllvd/signbank/asllrp_videos/batch_utterance_video_v3_1.zip',
            'hand_scope':'Dominant-realized signs (1h and2h), all hidden rows retained; not an exhaustive two-hand parallel annotation task.',
            'usable_until_download_decode_ctc_checks':0,'training_source_only':True,'script_sha256':sha(__file__)}
    write(folder/'annotation_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rejected'}),flush=True)
    print(json.dumps(dict(Counter(r['reason'] for r in rejected))),flush=True)

if __name__=='__main__':main()
