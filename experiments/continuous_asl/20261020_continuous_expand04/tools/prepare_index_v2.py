"""Port the verified causal378 preparation function and reuse all223 prior features."""
import ast
import json
from pathlib import Path
import sys
from core import RUN, ROOT, SOURCE, read, write, sha


def main(wait_for_downloads=False):
    folder = RUN / 'full_index_v2/preparation'
    if (folder / 'manifest.json').exists():
        print('PREPARATION_ALREADY_COMPLETE_NO_REPEAT')
        return
    incoming = read(RUN / 'full_index_v2/plan.json')
    missing = [r['sample_id'] for r in incoming['rows'] if not Path(r['video_path']).is_file() or not Path(str(r['video_path']) + '.receipt.json').is_file()]
    if missing and not wait_for_downloads:
        raise ValueError('FINISH_BOUNDED_ACQUISITION_FIRST:' + str(missing[:10]) + ':count=' + str(len(missing)))
    if not (folder / 'plan.json').exists():
        rows = []
        for original in incoming['rows']:
            row = {**original, 'new_in_catalog_v2': original['new_raw_required']}
            if not original['new_raw_required']:
                row['cached_record'] = {**original, 'source_video_sha256': original['raw_sha256'], 'cache_origin': 'FROZEN_EXPAND03_VERIFIED_CATALOG223'}
            rows.append(row)
        write(folder / 'plan.json', {'rows': rows, 'source_plan_sha256': sha(RUN / 'full_index_v2/plan.json')})
    # Exactly the existing decoder, Candidate01, raw-v1 and causal378 implementation.
    path = SOURCE / 'tools/expand_catalog.py'
    text = path.read_text(encoding='utf-8')
    parsed = ast.parse(text)
    function = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == 'prepare')
    body = ast.get_source_segment(text, function)
    # Source plan counts are dynamic. This metadata field describes the already frozen181/9 run, not new fitting.
    body = body.replace("'frozen_v1_training_counts':{'train':100,'dev':9}", "'previous_frozen_training_counts':{'train':181,'dev':9}")
    anchor = "video=Path(row['video_path']);receipt=read(str(video)+'.receipt.json')"
    replacement = """video=Path(row['video_path'])
                pending=time.monotonic()
                while not video.is_file() or not Path(str(video)+'.receipt.json').is_file():
                    failure=RUN/'logs'/('download_indexv2_'+sample+'_attempt1.json')
                    if failure.exists():raise ValueError('DOWNLOAD_FAILED_RETAINED_EVIDENCE:'+str(failure))
                    if time.monotonic()-pending>120:raise ValueError('DOWNLOAD_NOT_READY_WITHIN120SECONDS')
                    time.sleep(0.5)
                receipt=read(str(video)+'.receipt.json')"""
    if anchor not in body:
        raise ValueError('PREPARATION_PORT_ANCHOR_CHANGED')
    body = body.replace(anchor, replacement)
    header = '''from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from core import RUN, ROOT, OLD, SCHEMA, read, write, sha
CAT=RUN/'full_index_v2/preparation'
OUT=RUN/'data/processed/index_v2'
'''
    port = RUN / 'tools/preparation_port.py'
    source = header + '\n' + body + "\n\nif __name__=='__main__':prepare()\n"
    if port.exists():
        if port.read_text(encoding='utf-8') != source:
            raise ValueError('PREPARATION_PORT_CHANGED')
    else:
        port.write_text(source, encoding='utf-8')
        write(folder / 'code_provenance.json', {'source': str(path), 'source_sha256': sha(path), 'port_sha256': sha(port),
                                              'change': 'New plan/output paths, reuse223 caches, metadata frozen counts181/9; bounded file-only wait for in-flight approved downloads; feature math and observation policy unchanged.'})
    import subprocess
    subprocess.run([sys.executable, '-B', str(port)], check=True, cwd=ROOT)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-for-downloads', action='store_true')
    main(parser.parse_args().wait_for_downloads)
