"""Preserve numerical v1 result; replace the globally ineligible pilot record."""
from core import RUN,SIX,read,write,sha

def main():
    destination=RUN/'tools/overfit_v2.py'
    if destination.exists():print('V2_DIAGNOSTIC_TOOL_EXISTS');return
    old_manifest=read(RUN/'overfit_diagnostic/manifest.json')
    blocked=set(read(SIX/'asllrp2025_catalog/annotation_boundary_audit.v2.json')['reject_sample_ids'])
    invalid=[r['sample_id'] for r in old_manifest if r['sample_id'] in blocked]
    if not invalid:raise ValueError('NO_DEMONSTRATED_ELIGIBILITY_DEFECT')
    source_path=RUN/'tools/overfit.py';source=source_path.read_text(encoding='utf-8')
    source=source.replace("RUN/'overfit_diagnostic'","RUN/'overfit_diagnostic_v2'")
    before="asl=[read(SIX/'data/processed/asllrp2025'/(s+'.row.json')) for s in pilot['accepted']]"
    after="""blocked=set(read(SIX/'asllrp2025_catalog/annotation_boundary_audit.v2.json')['reject_sample_ids'])
    selected={}
    for candidate in read(SIX/'asllrp2025_catalog/media_manifest.json'):
        path=SIX/'data/processed/asllrp2025'/(candidate['sample_id']+'.row.json')
        source_collection=candidate['collection'].split('_')[0]
        if candidate['sample_id'] not in blocked and path.exists() and source_collection not in selected:
            selected[source_collection]=read(path)
        if len(selected)==3:break
    asl=list(selected.values())"""
    if before not in source:raise ValueError('ELIGIBLE_REPLACEMENT_ANCHOR')
    source=source.replace(before,after)
    source=source.replace('3 source-collection-diverse ASLLRP pilot clips.','3 source-collection-diverse ASLLRP completed clips excluding all global overlap IDs.')
    with destination.open('x',encoding='utf-8') as f:f.write(source)
    write(RUN/'overfit_gate_adjudication.v2.json',{'v1_status':'NUMERICAL_LEARNABILITY_PASSED_BUT_INVALID_FOR_ELIGIBILITY_GATE',
          'reason':'Global inclusive-master-frame audit found overlap in one selected pilot utterance. No labels or boundaries edited.',
          'excluded_diagnostic_ids':invalid,'original_result_sha256':sha(RUN/'overfit_diagnostic/result.json'),
          'authoritative_main_gate':'overfit_diagnostic_v2/result.json must PASS',
          'v2_code_sha256':sha(destination),'source_code_sha256':sha(source_path),
          'changes':'New output directory and eligible source-diverse ASLLRP replacement; identical model/normalization/optimizer/bounds/criteria.'})

if __name__=='__main__':main()
