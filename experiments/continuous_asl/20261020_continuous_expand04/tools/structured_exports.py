"""Conservative SignStream-v5 parser. Preserve source cells; reject ambiguity."""
from collections import Counter, defaultdict
import csv
import hashlib
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit
from core import RUN, SOURCE, read, write, sha


def parse_export(path):
    # Original exports are ASCII/Mac-era files. Accept ASCII only until an encoding is verified.
    text = Path(path).read_bytes().decode('ascii')
    lines = text.splitlines()
    if not lines or lines[0] != 'SIGNSTREAM EXPORT DATA FILE':
        raise ValueError('NOT_SIGNSTREAM_EXPORT')
    fields = defaultdict(list)
    key = None
    for line in lines[1:]:
        if not line:
            continue
        cells = line.split('\t')
        if cells[0] == '':
            if key is None:
                raise ValueError('CONTINUATION_WITHOUT_FIELD')
            fields[key][-1].extend(cells[1:])
        else:
            key = cells[0]
            fields[key].append(cells[1:])
    def single(name):
        if len(fields[name]) != 1 or len(fields[name][0]) != 1:
            raise ValueError('EXPECTED_SINGLE_FIELD:' + name)
        return fields[name][0][0]
    if single('Export data format version:') != '5' or single('Number of participant panes:') != '1':
        raise ValueError('UNSUPPORTED_VERSION_OR_MULTIPLE_PARTICIPANTS')
    gloss_cells = fields['main gloss']
    if len(gloss_cells) != 1 or len(gloss_cells[0]) % 3:
        raise ValueError('INVALID_GLOSS_TRIPLES')
    tokens, times = [], []
    cells = gloss_cells[0]
    for offset in range(0, len(cells), 3):
        token, start, end = cells[offset:offset + 3]
        if not token or token in ('|>', '<|', '>', '<') or not start.isdecimal() or not end.isdecimal():
            raise ValueError('NONSTANDARD_EVENT_OR_TIME_NO_REPAIR')
        start, end = int(start), int(end)
        if start > end or (times and start < times[-1][0]):
            raise ValueError('UNORDERED_SOURCE_EVENTS')
        tokens.append(token)
        times.append([start, end])
    if len(tokens) < 2:
        raise ValueError('NOT_CONTINUOUS_MULTI_EVENT')
    videos = [cells[0].split(':')[-1] for cells in fields['Video'] if len(cells) == 1 and cells[0]]
    return {'database': single('Database:'), 'export_utterance_id': single('Utterance ID:'),
            'participant_source_name': single('Participant:'), 'glosses': tokens, 'sign_times_original': times,
            'source_utterance_bounds_original': [int(single('Start frame:')), int(single('End frame:'))],
            'video_basenames': videos, 'annotation_path': str(Path(path).resolve()), 'annotation_sha256': sha(path),
            'annotation_format': 'SIGNSTREAM_TEXT_EXPORT_VERSION5_CA2002',
            'time_units': 'LEGACY_SOURCE_FRAME_COORDINATES_UNCALIBRATED_TO_CURRENT_CONTAINER; preserve integers; no conversion or boundary evaluation'}


def plan():
    if (RUN / 'signstream_candidate_plan.json').exists():
        print('STRUCTURED_PLAN_EXISTS_NO_REWRITE')
        return
    original = read(SOURCE / 'dataset_plan.json')
    index = list(csv.DictReader(Path(original['index_path']).open(encoding='utf-8-sig', newline='')))
    by_media = defaultdict(list)
    uses = defaultdict(set)
    for number, row in enumerate(index, 1):
        uses[row['Compressed MOV file']].add((row['XML file'], row['Occurs in utterance id']))
        if row['Perspective/Camera id'] == '0':
            by_media[row['Video file name in XML file']].append((number, row))
    prior = read(SOURCE / 'expanded_catalog_v2/manifest.json')
    previous_ids = {r['sample_id'] for r in prior}
    previous_urls = {r['source_url'] for r in prior}
    rows, excluded, existing_annotations = [], [], []
    for path in sorted((RUN / 'data/raw/bu_signstream/exports').glob('*.txt')):
        try:
            annotation = parse_export(path)
            master = [p for p in annotation['video_basenames'] if '_master_' in p]
            if len(master) != 1:
                raise ValueError('NO_UNAMBIGUOUS_MASTER_MEDIA')
            matched = [(n, r) for n, r in by_media[master[0]] if r['XML file'] == annotation['database'] + '.xml']
            if len(matched) != 1:
                raise ValueError('MEDIA_COLLECTION_NOT_UNIQUE_IN_OFFICIAL_INDEX')
            number, source = matched[0]
            url = source['Compressed MOV file']
            if len(uses[url]) != 1:
                raise ValueError('SHARED_MEDIA_NEEDS_CALIBRATED_UTTERANCE_BOUNDARIES')
            if ' '.join(annotation['glosses']) != source['main gloss']:
                raise ValueError('ANNOTATION_VERSION_GLOSS_CONFLICT_NO_SUBSTITUTION')
            identity = source['XML file'] + '|' + source['Occurs in utterance id']
            sample = 'ncslgr_' + hashlib.sha256(identity.encode()).hexdigest()[:16]
            parts = urlsplit(url)
            url = urlunsplit(('https', parts.netloc, parts.path, parts.query, ''))
            if parts.hostname != 'csr.bu.edu' or not parts.path.endswith('.mov'):
                raise ValueError('NOT_VERIFIED_INDEX_MOV')
            row = {**annotation, 'sample_id': sample, 'collection': source['XML file'],
                   'utterance_id': source['Occurs in utterance id'], 'source_group': source['Video sequence id'],
                   'source_url': url, 'source_url_original': source['Compressed MOV file'],
                   'gloss_text': source['main gloss'], 'dataset': 'NCSLGR',
                   'signer_id': 'ncslgr_participant_' + hashlib.sha256(annotation['participant_source_name'].encode()).hexdigest()[:16],
                   'annotation_type': 'AUTHENTIC_STRUCTURED_MAIN_GLOSS_EVENTS_MATCHING_OFFICIAL_INDEX',
                   'annotation_provenance': {'index_path': original['index_path'], 'index_sha256': original['index_sha256'], 'source_row': number,
                                             'export_source_url': read(str(path) + '.receipt.json')['source_url'],
                                             'ID_note': 'Legacy export ID and current index ID preserved separately; join uses exact media basename+collection+gloss, never inferred offset.'},
                   'license_status': 'BU_PRIVATE_NONCOMMERCIAL_RESEARCH_TERMS_REUSED',
                   'license_reference': 'https://www.bu.edu/asllrp/dai-terms.html',
                   'video_path': str(RUN / 'data/raw/ncslgr_structured/videos' / (sample + '.mov'))}
            if sample in previous_ids or url in previous_urls:
                existing_annotations.append(row)
                continue
            if source['XML file'] in ('ncslgr10e.xml', 'ncslgr10m.xml', 'ncslgr10p.xml', 'ncslgr10q.xml'):
                raise ValueError('PROTECTED_PRIOR_DEV_TEST_COLLECTION')
            # New recordings in collection10d are reserved before new fitting; no signer-disjoint claim.
            row['split'] = 'test_reserved' if source['XML file'] == 'ncslgr10d.xml' else 'train_addition'
            rows.append(row)
        except Exception as error:
            excluded.append({'path': str(path), 'reason': str(error)})
    for key in ('sample_id', 'source_url'):
        if len({r[key] for r in rows}) != len(rows):
            raise ValueError('DUPLICATE_NEW_CANDIDATE:' + key)
    write(RUN / 'signstream_candidate_plan.json', {
        'status': 'PASS', 'eligible_new': rows, 'new_count': len(rows), 'role_counts': dict(Counter(r['split'] for r in rows)),
        'existing_annotations_recovered': existing_annotations, 'existing_count_not_new': len(existing_annotations),
        'excluded': excluded, 'rejected_count': len(excluded), 'index_sha256': original['index_sha256'],
        'timing_policy': 'Preserve original coordinates; use only unique one-utterance source files for CTC; no unverified time conversion.',
        'annotation_identity_policy': 'Exact master media basename, collection and full gloss text; index2011 and legacy2002 identifiers are distinct.'})
    print(__import__('json').dumps({'new': len(rows), 'roles': dict(Counter(r['split'] for r in rows)), 'existing': len(existing_annotations), 'rejected': len(excluded), 'reasons': dict(Counter(r['reason'] for r in excluded))}))


if __name__ == '__main__':
    plan()
