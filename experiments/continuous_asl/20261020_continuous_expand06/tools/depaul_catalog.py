"""Strict official ELAN media/gloss/time joins, including multi-utterance recordings."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, unquote
import xml.etree.ElementTree as ET
import zipfile

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[2]
THREE = ROOT / 'experiments/continuous_asl/20261020_continuous_expand03'
FOUR = ROOT / 'experiments/continuous_asl/20261020_continuous_expand04'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def tier_events(root, name, slots):
    tiers = [t for t in root.findall('TIER') if t.attrib.get('TIER_ID') == name]
    if len(tiers) != 1:
        raise ValueError('MISSING_OR_MULTIPLE_TIER:' + name)
    result = []
    for wrapper in tiers[0].findall('ANNOTATION'):
        item = wrapper.find('ALIGNABLE_ANNOTATION')
        if item is None:
            raise ValueError('REFERENCE_ANNOTATION_NEEDS_SEPARATE_RESOLUTION')
        a, b = slots[item.attrib['TIME_SLOT_REF1']], slots[item.attrib['TIME_SLOT_REF2']]
        value = item.findtext('ANNOTATION_VALUE')
        if not value or a < 0 or b < a:
            raise ValueError('INVALID_EVENT_VALUE_OR_TIME')
        result.append({'id': item.attrib['ANNOTATION_ID'], 'start_ms': a, 'end_ms': b, 'value': value})
    return result


def main():
    output = RUN / 'depaul_catalog/plan.json'
    if output.exists():
        print('DEPAUL_PLAN_EXISTS_NO_REWRITE')
        return
    original = read(THREE / 'dataset_plan.json')
    index = list(csv.DictReader(Path(original['index_path']).open(encoding='utf-8-sig', newline='')))
    lookup = defaultdict(list)
    for number, row in enumerate(index, 1):
        if row['Perspective/Camera id'] == '0':
            lookup[Path(urlsplit(row['Compressed MOV file']).path).name.casefold()].append((number, row))
    previous = read(FOUR / 'full_index_v2/preparation/manifest.json')
    known_ids = {r['sample_id'] for r in previous}
    by_id = {r['sample_id']: r for r in previous}
    path = RUN / 'data/raw/depaul/elanBUcorpus.zip'
    receipt = read(str(path) + '.receipt.json')
    rows, exclusions, file_audit, joins, conflicts = [], [], [], [], []
    seen = set()
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > 5000 or sum(i.file_size for i in entries) > 134217728 or max(i.file_size for i in entries) > 8388608:
            raise ValueError('ANNOTATION_EXPANSION_BUDGET')
        bad = archive.testzip()
        if bad:
            raise ValueError('ANNOTATION_ARCHIVE_CRC:' + bad)
        for info in entries:
            if not info.filename.endswith('.eaf'):
                continue
            if '/DSP ' in info.filename:
                exclusions.append({'member': info.filename, 'reason': 'DAWNSIGNPRESS_SEPARATE_RIGHTS_NOT_CLEARED'})
                continue
            if any(token in Path(info.filename).name for token in ('ali_', 'roadtrip2_')):
                exclusions.append({'member': info.filename, 'reason': 'PUBLISHER_DOCUMENTED_SYNC_OR_MISSING_ANNOTATION_PROBLEM'})
                continue
            try:
                content = archive.read(info)
                root = ET.fromstring(content)
                header = root.find('HEADER')
                if header is None or header.attrib.get('TIME_UNITS') != 'milliseconds':
                    raise ValueError('UNVERIFIED_TIME_UNITS')
                media = header.findall('MEDIA_DESCRIPTOR')
                if any(int(m.attrib.get('TIME_ORIGIN', 0)) != 0 for m in media):
                    raise ValueError('NONZERO_MEDIA_ORIGIN_NEEDS_DOCUMENTED_ADAPTER')
                media_matches = []
                for descriptor in media:
                    basename = Path(unquote(urlsplit(descriptor.attrib['MEDIA_URL']).path)).name.casefold()
                    if basename in lookup:
                        media_matches.append((descriptor, lookup[basename]))
                if len(media_matches) != 1:
                    raise ValueError('NO_UNAMBIGUOUS_LINKED_FRONTAL_INDEX_MEDIA')
                descriptor, matches = media_matches[0]
                collections = {r['XML file'] for _, r in matches}
                urls = {r['Compressed MOV file'] for _, r in matches}
                if len(collections) != 1 or len(urls) != 1:
                    raise ValueError('AMBIGUOUS_COLLECTION_OR_SOURCE_URL')
                collection = next(iter(collections))
                slots = {s.attrib['TIME_SLOT_ID']: int(s.attrib['TIME_VALUE']) for s in root.find('TIME_ORDER')}
                if len(slots) != len(root.find('TIME_ORDER')):
                    raise ValueError('DUPLICATE_TIME_SLOT_ID')
                glosses = tier_events(root, 'main gloss', slots)
                # This tier's publisher-supplied spans locate utterances. Its English text is never a target.
                utterances = tier_events(root, 'English translation', slots)
                movie_url = next(iter(urls))
                parts = urlsplit(movie_url)
                movie_url = urlunsplit(('https', parts.netloc, parts.path, parts.query, ''))
                if parts.hostname != 'csr.bu.edu':
                    raise ValueError('NOT_OFFICIAL_NCSLGR_VIDEO_HOST')
                for utterance in utterances:
                    try:
                        a, b = utterance['start_ms'], utterance['end_ms']
                        if b <= a:
                            raise ValueError('EMPTY_UTTERANCE_SPAN')
                        crossing = [g for g in glosses if g['start_ms'] < b and g['end_ms'] > a and not (a <= g['start_ms'] and g['end_ms'] <= b)]
                        if crossing:
                            raise ValueError('GLOSS_CROSSES_UTTERANCE_BOUNDARY_NO_CLIPPING')
                        events = [g for g in glosses if a <= g['start_ms'] and g['end_ms'] <= b]
                        if len(events) < 2:
                            raise ValueError('NOT_MULTI_GLOSS_CONTINUOUS')
                        if any(x['start_ms'] > y['start_ms'] for x, y in zip(events, events[1:])):
                            raise ValueError('GLOSS_SOURCE_ORDER_NOT_TEMPORAL')
                        target = [g['value'] for g in events]
                        text = ' '.join(target)
                        exact = [(n, r) for n, r in matches if r['main gloss'] == text]
                        if len(exact) != 1:
                            raise ValueError('FULL_GLOSS_NOT_UNIQUELY_MATCHED_TO_OFFICIAL_INDEX')
                        number, indexed = exact[0]
                        identity = collection + '|' + indexed['Occurs in utterance id']
                        sample = 'ncslgr_' + hashlib.sha256(identity.encode()).hexdigest()[:16]
                        if sample in seen:
                            raise ValueError('DUPLICATE_ANNOTATION_IDENTITY')
                        seen.add(sample)
                        row = {'sample_id': sample, 'dataset': 'NCSLGR', 'collection': collection,
                               'utterance_id': indexed['Occurs in utterance id'], 'source_group': indexed['Video sequence id'],
                               'source_url': movie_url, 'glosses': target, 'gloss_text': text,
                               'start_ms': a, 'end_ms': b, 'sign_times_ms': [[g['start_ms'], g['end_ms']] for g in events],
                               'annotation_event_ids': [g['id'] for g in events], 'utterance_span_event_id': utterance['id'],
                               'annotation_type': 'AUTHENTIC_ELAN_MAIN_GLOSS_EVENTS_EXACTLY_MATCHED_TO_OFFICIAL_INDEX',
                               'annotation_provenance': {'archive_path': str(path), 'archive_sha256': receipt['sha256'],
                                    'member': info.filename, 'member_sha256': hashlib.sha256(content).hexdigest(),
                                    'eaf_version': root.attrib.get('VERSION'), 'time_units': 'milliseconds',
                                    'media_descriptor': descriptor.attrib, 'index_path': original['index_path'],
                                    'index_sha256': original['index_sha256'], 'source_row': number,
                                    'English_translation_used_as_gloss': False, 'span_source': 'Publisher-aligned utterance span from English translation tier; target values exclusively main gloss'},
                               'existing_verified_utterance': sample in known_ids, 'requires_temporal_slice': len(matches) > 1,
                               'source_file_utterance_count': len(matches), 'signer_id': None,
                               'license_status': 'BU_RESEARCH_TERMS_WITH_OFFICIAL_DEPAUL_CONVERSION_ATTRIBUTION',
                               'license_reference': 'https://www.bu.edu/asllrp/dai-terms.html'}
                        if sample in known_ids:
                            joins.append({'sample_id': sample, 'glosses_equal': target == by_id[sample]['glosses'],
                                          'start_ms': a, 'end_ms': b, 'member': info.filename})
                        if collection in ('ncslgr10e.xml', 'ncslgr10m.xml', 'ncslgr10p.xml', 'ncslgr10q.xml'):
                            row['role'] = 'PROTECTED_PRIOR_DEV_TEST_COLLECTION'
                        else:
                            row['role'] = 'NEW_RESEARCH_TRAIN_CANDIDATE' if sample not in known_ids else 'REUSED_VERIFIED'
                        rows.append(row)
                    except Exception as error:
                        conflicts.append({'member': info.filename, 'span_id': utterance['id'], 'reason': str(error)})
                file_audit.append({'member': info.filename, 'collection': collection, 'source_url': movie_url,
                                   'eaf_utterances': len(utterances), 'eaf_gloss_events': len(glosses)})
            except Exception as error:
                exclusions.append({'member': info.filename, 'reason': str(error)})
    unique_files = defaultdict(list)
    for row in rows:
        if not row['existing_verified_utterance']:
            unique_files[row['source_url']].append(row)
    data = {'status': 'PASS_ANNOTATION_CATALOG_ONLY_MEDIA_PENDING', 'archive_integrity': {'members': len(entries), 'expanded_bytes': sum(i.file_size for i in entries), 'crc_pass': True},
            'annotation_matched_candidates': len(rows), 'existing_joins': len(joins),
            'new_annotation_matched_utterances': sum(not r['existing_verified_utterance'] for r in rows),
            'unique_new_media_urls': len(unique_files), 'rows': rows, 'member_audit': file_audit,
            'exclusions': exclusions, 'span_conflicts': conflicts, 'existing_label_checks': joins,
            'new_media_priority': [{'source_url': url, 'utterances': len(samples), 'collections': sorted({r['collection'] for r in samples})} for url, samples in sorted(unique_files.items(), key=lambda kv: -len(kv[1]))],
            'not_counted_as_usable_until_media_and_features_verified': True,
            'minimum_target': 2000, 'stretch_target': 5000}
    write(output, data)
    print(json.dumps({k: data[k] for k in ('annotation_matched_candidates', 'existing_joins', 'new_annotation_matched_utterances', 'unique_new_media_urls')}))
    print(json.dumps({'member_exclusions': dict(Counter(r['reason'] for r in exclusions)),
                      'span_conflicts': dict(Counter(r['reason'] for r in conflicts)), 'priority': data['new_media_priority'][:8]}))


if __name__ == '__main__':
    main()
