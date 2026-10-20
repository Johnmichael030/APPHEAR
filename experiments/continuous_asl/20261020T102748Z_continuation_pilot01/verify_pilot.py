"""Read-only pilot verification; no training or inference."""
import hashlib,json
from pathlib import Path
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[2]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for c in iter(lambda:f.read(1048576),b''):h.update(c)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def main():
 count=0
 for line in (RUN/'artifact_hashes.sha256').read_text().splitlines():
  if not line or line.startswith('#'):continue
  expected,name=line.split('  ',1);path=(RUN/name).resolve()
  if RUN not in path.parents or sha(path)!=expected:raise ValueError('PILOT_ARTIFACT_HASH:'+name)
  count+=1
 status=read(RUN/'training_status.json');config=read(RUN/'config.json')
 if sha(config['manifest_path'])!=status['manifest_sha256'] or sha(config['parent_checkpoint'])!=config['parent_checkpoint_sha256']:raise ValueError('FROZEN_INPUT_HASH')
 if sha(RUN/status['best_checkpoint']['path'])!=status['best_checkpoint']['sha256'] or not status['used_cuda'] or not status['checkpoint_reload_predictions_match'] or status['test_accessed'] or status['production_promoted']:raise ValueError('PILOT_EXECUTION_CONTRACT')
 print(json.dumps({'status':'PASS_READ_ONLY_PILOT_VERIFICATION','artifacts_hashed':count,'additional_epochs':status['completed_additional_epochs'],
   'train':status['unique_train_consumed'],'dev':9,'test_used':False,'training_performed_by_verifier':False,'production_promoted':False},indent=2))
if __name__=='__main__':main()
