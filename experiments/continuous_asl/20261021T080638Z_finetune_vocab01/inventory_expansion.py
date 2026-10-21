"""Metadata-only source inventory; no participant video decode, feature extraction or new label claim."""
from collections import Counter,defaultdict
from pathlib import Path
import json
import fine_tune as ft
ROOT=ft.ROOT;RUN=ft.RUN;local=Path('D:/ASL VIDEOS DATASETS');vocab=ft.read(RUN/'vocabulary.json')['tokens'];known=set(vocab[2:]);proposal=[]
labels=ft.read(local/'records/AppHearDataset/labels.json')['labels'];sessions=ft.read(local/'records/AppHearDataset/metadata/sessions.json')['sessions']
signer_meta=ft.read(local/'records/AppHearDataset/metadata/signers.json')
local_counts=[];local_glosses={}
for directory in sorted((local/'records/AppHearDataset/samples/signs').iterdir()):
 if not directory.is_dir():continue
 entries=[ft.read(p) for p in directory.glob('*.json')];glosses={m['gloss'] for m in entries};assert len(glosses)==1
 gloss=next(iter(glosses));assert all(m['schema_version']=='3.0' and m['schema_stage']=='E' for m in entries)
 signers={m['signer_id'] for m in entries};session_ids={m['session_id'] for m in entries}
 count=sum((directory/m['video_file']).exists() for m in entries)
 row=dict(candidate_gloss=gloss,intended_meaning=entries[0]['word']+' (recorder prompt; not independently verified translation)',currently_supported=gloss in known,training_example_count=0,distinct_signer_count=len(signers),recording_session_count=len(session_ids),source_dataset='Local ASLRecorder StageE isolated signing',annotation_status='PROMPT_METADATA_UNREVIEWED; schema3.0 is NOT Candidate01_378',permission_status='CONSENT_RECORD_NOT_ESTABLISHED',priority=1 if gloss not in known else 2,proposed_action='Obtain permissions and qualified target-contract review; create independent new-class validation; then re-extract genuine378features',exclusion_reason='No consent/review records; one session per sign/one signer; no new-label validation; feature schema incompatible',potential_labeled_utterances=count,source_identity=str(directory),download_bytes=0)
 proposal.append(row);local_counts.append(dict(label=gloss,metadata_records=len(entries),MP4_files=count,signers=len(signers),sessions=len(session_ids),feature_schema='3.0',eligible_training_records=0));local_glosses[gloss]=entries[0]['word']
ritdir=ROOT/'experiments/continuous_asl/20261020T104054Z_expand_eval01/rit2025_catalog';rit=ft.read(ritdir/'media_manifest.json');excluded=set(ft.read(ritdir/'annotation_boundary_audit.v2.json')['reject_sample_ids'])
groups=defaultdict(list)
for r in rit:
 if r['sample_id'] in excluded:continue
 for gloss in set(r['glosses']):
  if gloss not in known:groups[gloss].append(r)
for gloss,rows in sorted(groups.items(),key=lambda kv:(-len(kv[1]),kv[0])):
 signers={r['signer_id'] for r in rows if r.get('signer_id') and 'UNKNOWN' not in r['signer_id']}
 proposal.append(dict(candidate_gloss=gloss,intended_meaning='Published literal occurrence label; English meaning not inferred from gloss spelling',currently_supported=False,training_example_count=0,distinct_signer_count=len(signers),recording_session_count=len({r['collection'] for r in rows}),source_dataset='RIT SignStream3 November2025 continuous utterances',annotation_status='AUTHENTIC_PUBLISHED_OCCURRENCE; native-frame/crop-origin contract unresolved',permission_status='PRIVATE_NONCOMMERCIAL_RESEARCH_ONLY_NO_REDISTRIBUTION',priority=1 if gloss in local_glosses else 2,proposed_action='Resolve publisher60fps/two-extra-frame crop origin; preserve full literal labels and exclusions; verify new split/features before B',exclusion_reason='All278media have+2frames vsinclusive publisher span; no guessed offset/trim; independent validation not frozen',potential_labeled_utterances=len({r['sample_id'] for r in rows}),source_identity=';'.join(sorted({r['collection'] for r in rows})),download_bytes=0))
# Existing-class support is a distinct replay/retention workstream, not expansion.
train=ft.read(RUN/'manifests/TRAIN1500.json');support=Counter(t for r in train for t in set(r['glosses']))
for gloss in sorted((t for t,n in support.items() if n<5),key=lambda t:(t not in local_glosses,-support[t],t))[:40]:
 rows=[r for r in train if gloss in r['glosses']]
 proposal.append(dict(candidate_gloss=gloss,intended_meaning='Existing literal model target; retain its established meaning',currently_supported=True,training_example_count=support[gloss],distinct_signer_count=len({r['signer_id'] for r in rows}),recording_session_count=len({ft.family(r) for r in rows}),source_dataset='Frozen ASLLRP3 TRAIN1500',annotation_status='PUBLISHED_LITERAL_TARGET; existing feature/CTC checksPASS',permission_status='PRIVATE_NONCOMMERCIAL_RESEARCH_ONLY_NO_REDISTRIBUTION',priority=2,proposed_action='Retain old ID; collect verified repeated support; suitable existing-class replay',exclusion_reason='',potential_labeled_utterances=support[gloss],source_identity='Frozen TRAIN1500 only',download_bytes=0))
proposal.sort(key=lambda r:(r['priority'],r['currently_supported'],-r['potential_labeled_utterances'],r['candidate_gloss'],r['source_dataset']))
ft.writecsv(RUN/'VOCABULARY_EXPANSION_PLAN.csv',proposal)
summary=dict(status='BLOCKED_NEW_CLASS_ADMISSION',proposed_rows=len(proposal),distinct_proposed_new_labels=len({r['candidate_gloss'] for r in proposal if not r['currently_supported']}),local_records=local_counts,local_recorded_utterances=sum(r['MP4_files'] for r in local_counts),local_unique_signer_metadata=len(signer_meta['signers']),local_consented_verified_records=0,local_new_labels=[r['label'] for r in local_counts if r['label'] not in known],RIT_media_records=len(rit),RIT_nonoverlap_records=len([r for r in rit if r['sample_id'] not in excluded]),RIT_new_literal_labels=len(groups),RIT_eligible_records=0,top_level_named_MP4_files=len(list(local.glob('*.mp4'))),top_level_named_MP4_bytes=sum(p.stat().st_size for p in local.glob('*.mp4')),top_level_training_rights='UNKNOWN; names are not verified ASL annotations',DSP='Publisher about-datasets.pdf explicitly excludes DawnSignPress download; archive index visibility is NOT authorization. No DSP payload acquired.',How2Sign='Existing English translations are not gloss sequences; no repeat acquisition.',ASLLRP3='Existing2119eligibleallalreadyassigned1500TRAIN/206internal-held/413externalDEV. No held family/signer reassignment for expansion.',NCSLGR2='Different main-gloss contract retained separately; no union',new_data_download_bytes=0,media_decoded=0,actual_new_classes_added=0)
ft.save(RUN/'DATA_INVENTORY.json',summary)
(RUN/'B').mkdir(exist_ok=True)
ft.save(RUN/'B/TRAINING_CONFIGURATION.json',dict(experiment='B_append_only_vocabulary_finetune',status='BLOCKED_NEW_CLASS_DATA',source_checkpoint=str(ft.SOURCE),source_checkpoint_sha256=ft.sha(ft.SOURCE),existing_classes=2289,blank_id=0,unknown_id=1,feature_width=378,supported_new_classes=[],admission_manifest_required='B/APPROVED_DATA_MANIFEST.json',minimum_distinct_training_utterances_per_new_class=3,minimum_independent_validation_utterances_per_new_class=1,policy='Preserve all old IDs/weights; initialize only appended output rows; restore compatible optimizer moments; genuine old-class replay plus reviewed compatible new examples',normalization='frozen source parameters unless separately controlled',max_epochs=12,max_wall_seconds=180,learning_rate=.0001,batch_size=4,gradient_accumulation=2,production_promoted=False,TEST_used=False))
ft.save(RUN/'C/TRAINING_CONFIGURATION.json',dict(experiment='C_retention_or_imbalance_control',status='NOT_RUN_PREDECLARED_PROMISING_GATE_FAILED',reason='AgreedyDEVgain1.0309pp<2pprequired; streamingWERunchanged. Bdataadmissionblocked.',changes_applied=False,TEST_used=False))
print(json.dumps({k:v for k,v in summary.items() if k!='local_records'},indent=2))
