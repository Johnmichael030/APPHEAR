"""Read-only verification of diagnostic outputs and recorded frozen inputs."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def main():
    checked=0
    for line in (HERE/'artifact_hashes.sha256').read_text(encoding='utf-8').splitlines():
        if not line or line.startswith('#'):continue
        expected,name=line.split('  ',1)
        path=(HERE/name).resolve()
        if HERE not in path.parents or not path.is_file() or digest(path)!=expected:raise ValueError('DIAGNOSTIC_ARTIFACT_CHANGED:'+name)
        checked+=1
    inputs={}
    for name in ('input_artifacts.json','supplement_input_artifacts.json','source_contract_input_artifacts.json'):
        inputs.update(json.loads((HERE/name).read_text(encoding='utf-8')))
    for path,expected in inputs.items():
        if digest(Path(path))!=expected:raise ValueError('FROZEN_INPUT_CHANGED:'+path)
    before=json.loads((HERE/'preservation_before.json').read_text(encoding='utf-8'))['files']
    for name,expected in before.items():
        stat=Path(name).stat()
        if [stat.st_size,stat.st_mtime_ns]!=expected:raise ValueError('FROZEN_METADATA_CHANGED:'+name)
    print(json.dumps({'status':'PASS_READ_ONLY_DIAGNOSTIC_VERIFICATION','diagnostic_artifacts_hashed':checked,
        'frozen_input_hashes':len(inputs),'frozen_metadata_checks':len(before),'training_performed':False,'test_inference_performed':False},indent=2))

if __name__=='__main__':main()
