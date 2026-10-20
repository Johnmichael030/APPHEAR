"""Metadata-only reconstruction of the original custom DEV eligibility filter."""
from collections import defaultdict
import csv
import hashlib
import re
from urllib.parse import urlsplit
from audit import HERE,ROOT,RUNS,read,sha,write,INPUTS

def main():
    original=RUNS['baseline'];plan=read(original/'dataset_plan.json')
    with open(plan['index_path'],encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    sha(plan['index_path']);sha(original/'tools/core.py')
    uses=defaultdict(set)
    for row in rows:uses[row['Compressed MOV file']].add((row['XML file'],row['Occurs in utterance id'],row['main gloss']))
    lexical=re.compile(r'(?:\(2h\))?[A-Za-z][A-Za-z0-9_:+().-]*\+*$')
    candidates=[];seen=set()
    for row in rows:
        if row['XML file']!='ncslgr10p.xml':continue
        tokens=row['main gloss'].split();parts=urlsplit(row['Compressed MOV file']);identity=(row['XML file'],row['Occurs in utterance id'])
        if row['Perspective/Camera id']!='0' or len(uses[row['Compressed MOV file']])!=1:continue
        if parts.hostname!='csr.bu.edu' or not parts.path.endswith('.mov'):continue
        if not 2<=len(tokens)<=12 or not all(lexical.fullmatch(t) and not any(s in t.lower() for s in ('cl:','fs-','fingerspell')) for t in tokens):continue
        if identity in seen:continue
        seen.add(identity);candidates.append('ncslgr_'+hashlib.sha256('|'.join(identity).encode()).hexdigest()[:16])
    actual=[r['sample_id'] for r in read(original/'data/processed/train_dev/manifest.json') if r['split']=='dev']
    provenance=read(RUNS['main2119']/'code_provenance.json')
    shared=sha(ROOT/'tools/continuous_asl/ctc_experiment.py')
    checks=[{'status':'PASS' if set(candidates)==set(actual) and len(candidates)==9 else 'PARTIAL','test':'original_DEV_filter_reconstruction','reconstructed_ids':candidates,'frozen_ids':actual},
        {'status':'PASS' if shared==provenance['model_decoder_sha256'] else 'PARTIAL','test':'shared_model_decoder_source_matches_fitting_provenance','current_sha256':shared,'recorded_sha256':provenance['model_decoder_sha256']}]
    write('source_contract_checks.json',{'status':'PASS' if all(c['status']=='PASS' for c in checks) else 'PARTIAL','checks':checks,
        'DEV_policy':'Collection ncslgr10p.xml; frontal0, unambiguous whole-utterance movie,2..12initial lexical tokens, original exclusions of quoted/complex/classifier/fingerspelled labels. Later TRAIN policy broadened but DEV remained fixed.',
        'training_performed':False,'test_examples_addressed':False})
    write('source_contract_input_artifacts.json',INPUTS)
    print('PASS_SOURCE_CONTRACT_CHECKS' if all(c['status']=='PASS' for c in checks) else 'PARTIAL_SOURCE_CONTRACT_CHECKS',flush=True)

if __name__=='__main__':main()
