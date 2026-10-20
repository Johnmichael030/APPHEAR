"""Versioned broader ASL word-gloss target policy, including authentic fs-word tokens."""
from collections import Counter, defaultdict
import csv
import hashlib
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit
from core import RUN, SOURCE, read, write


def plan():
    output = RUN / 'full_index_v2/plan.json'
    if output.exists():
        print('FULL_INDEX_V2_PLAN_EXISTS_NO_REWRITE')
        return
    old = read(SOURCE / 'dataset_plan.json')
    source = list(csv.DictReader(Path(old['index_path']).open(encoding='utf-8-sig', newline='')))
    previous = {r['sample_id']: r for r in read(SOURCE / 'expanded_catalog_v2/manifest.json')}
    structured = read(RUN / 'signstream_candidate_plan.json')
    annotated = {r['sample_id']: r for r in structured['eligible_new'] + structured['existing_annotations_recovered']}
    uses = defaultdict(set)
    for row in source:
        uses[row['Compressed MOV file']].add((row['XML file'], row['Occurs in utterance id']))
    lex = re.compile(r'(?:\(2h\))?[A-Za-z][A-Za-z0-9_:+().-]*\+*$')
    rows, audit, seen = [], [], set()
    for number, row in enumerate(source, 1):
        key = (row['XML file'], row['Occurs in utterance id'])
        sample = 'ncslgr_' + hashlib.sha256('|'.join(key).encode()).hexdigest()[:16]
        tokens = row['main gloss'].split()
        url = row['Compressed MOV file']
        parts = urlsplit(url)
        decision = 'ADMITTED'
        if row['Perspective/Camera id'] != '0':
            decision = 'OTHER_CAMERA_NOT_NEW_UTTERANCE'
        elif len(uses[url]) != 1:
            decision = 'SHARED_MEDIA_REQUIRES_AUTHENTIC_CALIBRATED_UTTERANCE_TIMING'
        elif parts.hostname != 'csr.bu.edu' or not parts.path.endswith('.mov'):
            decision = 'NOT_VERIFIED_OFFICIAL_MOV'
        elif len(tokens) < 2:
            decision = 'NOT_CONTINUOUS_MULTI_GLOSS'
        elif sample not in annotated and not all(lex.fullmatch(t) and not t.lower().startswith('cl:') for t in tokens):
            decision = 'STRUCTURED_EVENT_BOUNDARIES_REQUIRED_FOR_COMPLEX_LABEL_STRINGS'
        elif key in seen:
            decision = 'DUPLICATE_INDEX_IDENTITY'
        audit.append({'source_row': number, 'collection': key[0], 'utterance_id': key[1], 'source_url': url,
                      'gloss_text': row['main gloss'], 'decision': decision})
        if decision != 'ADMITTED':
            continue
        seen.add(key)
        url = urlunsplit(('https', parts.netloc, parts.path, parts.query, ''))
        if sample in previous:
            entry = {**previous[sample], 'new_raw_required': False, 'source_catalog_role': previous[sample]['split']}
        else:
            entry = {**annotated.get(sample, {}), 'sample_id': sample, 'dataset': 'NCSLGR',
                     'collection': key[0], 'utterance_id': key[1], 'source_group': row['Video sequence id'],
                     'source_url': url, 'source_url_original': row['Compressed MOV file'], 'gloss_text': row['main gloss'],
                     'glosses': annotated[sample]['glosses'] if sample in annotated else tokens,
                     'annotation_type': 'AUTHENTIC_STRUCTURED_MAIN_GLOSS_EVENTS_MATCHING_OFFICIAL_INDEX' if sample in annotated else 'AUTHENTIC_INDEX_MAIN_GLOSS_TOKEN_SEQUENCE_INCLUDING_FS_WORDS',
                     'annotation_provenance': {**annotated.get(sample, {}).get('annotation_provenance', {}), 'official_index': old['index_path'], 'index_sha256': old['index_sha256'], 'source_row': number},
                     'new_raw_required': True,
                     'video_path': annotated[sample]['video_path'] if sample in annotated else str(RUN / 'data/raw/ncslgr_index_v2/videos' / (sample + '.mov')),
                     'signer_id': annotated.get(sample, {}).get('signer_id'),
                     'sign_times_original': annotated.get(sample, {}).get('sign_times_original'),
                     'time_units': annotated.get(sample, {}).get('time_units', 'UNAVAILABLE_IN_INDEX'),
                     'license_status': 'BU_PRIVATE_NONCOMMERCIAL_RESEARCH_TERMS_REVIEWED',
                     'license_reference': 'https://www.bu.edu/asllrp/dai-terms.html'}
            if key[0] in ('ncslgr10e.xml', 'ncslgr10m.xml'):
                entry['split'] = 'PRIOR_DEV_QUARANTINE'
            elif key[0] == 'ncslgr10p.xml':
                entry['split'] = 'NEW_DEV_COLLECTION_QUARANTINE'
            elif key[0] == 'ncslgr10q.xml' or annotated.get(sample, {}).get('split') == 'test_reserved':
                entry['split'] = 'NEW_TEST_RESERVED_UNTOUCHED'
            else:
                entry['split'] = 'TRAIN_POOL_NEW_FIT'
        rows.append(entry)
    write(output, {'schema': 'apphear.continuous_asl.full_index.v2', 'rows': rows,
                   'target_policy': 'Literal authentic main-gloss tokens, including fs-WORD as a complete source word target. Do not expand fingerspelling into letters. Multiword/quoted labels require matching structured event evidence. No English-caption substitution.',
                   'policy_change_from_expand03': 'Prior conservative filter excluded all fs-word targets. Their source-word boundaries are unambiguous in this admitted subset; sequence CTC needs no invented per-sign alignment.',
                   'source_index_sha256': old['index_sha256'], 'previous_unique_raw': len(previous),
                   'new_raw_required': sum(r['new_raw_required'] for r in rows),
                   'role_counts': dict(Counter(r['split'] for r in rows)),
                   'exclusion_counts': dict(Counter(r['decision'] for r in audit if r['decision'] != 'ADMITTED')),
                   'reserve_policy': 'Reserve new10q and previously declared new10d test samples before fitting; unique source groups/hashes disjoint. These share historical source collections, so no signer/collection-independent generalization claim.',
                   'frozen_dev': 'Exactly original9DEV; new6same-collection samples quarantined, not added to selection.'})
    path = RUN / 'full_index_v2/index_inventory.csv'
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    print(__import__('json').dumps({'total': len(rows), 'new_required': sum(r['new_raw_required'] for r in rows), 'roles': dict(Counter(r['split'] for r in rows))}))


if __name__ == '__main__':
    plan()
