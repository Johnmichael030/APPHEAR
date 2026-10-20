"""Complete the failed CSV serialization from saved JSON; no feature recomputation."""
import csv
import json
from core import RUN, read, write, sha


def main():
    base = RUN / 'full_index_v2/preparation'
    target = base / 'manifest.completed.csv'
    if target.exists():
        print('COMPLETED_CSV_EXISTS_NO_REWRITE')
        return
    rows = read(base / 'manifest.json')
    quality = read(base / 'quality.json')
    if len(rows) != quality['processed'] or len(rows) != 552 or quality['failures']:
        raise ValueError('SAVED_FEATURE_PREPARATION_INCOMPLETE')
    for row in rows:
        if sha(row['feature_path']) != row['feature_sha256']:
            raise ValueError('SAVED_FEATURE_HASH')
    fields = ('sample_id', 'collection', 'utterance_id', 'source_group', 'signer_id', 'split', 'source_url',
              'annotation_type', 'glosses', 'raw_sha256', 'bytes', 'feature_path', 'feature_sha256',
              'observations', 'new_raw_required', 'unique_valid', 'exclusion_reason')
    with target.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row[k]) if k == 'glosses' else row.get(k) for k in fields})
    with target.open(encoding='utf-8', newline='') as stream:
        if len(list(csv.DictReader(stream))) != len(rows):
            raise ValueError('CSV_ROW_COUNT')
    write(base / 'csv_export_recovery.json', {'status': 'PASS', 'error': "NameError: name 'csv' is not defined",
          'failed_command': 'python -B tools/prepare_index_v2.py --wait-for-downloads (calls preparation_port.py)',
          'failed_port_sha256': sha(RUN / 'tools/preparation_port.py'),
          'original_log': str(RUN / 'logs/prepare_index_v2.txt'), 'retained_failed_csv': str(base / 'manifest.csv'),
          'recovery_command': 'python -B tools/complete_csv_recovery.py',
          'completed_csv': str(target), 'completed_csv_sha256': sha(target), 'rows': len(rows),
          'json_manifest_sha256': sha(base / 'manifest.json'), 'video_decode_repeated': False, 'feature_extraction_repeated': False,
          'cause': 'Function port reused source preparation but omitted csv from adapter header; error was confined to final optional CSV serialization. Recovery explicitly imports csv and validates saved rows/features.'})
    print(json.dumps({'status': 'PASS_CSV_RECOVERY', 'rows': len(rows), 'reprocessed_videos': 0}))


if __name__ == '__main__':
    main()
