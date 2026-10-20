"""Read-only checks for complete lists, report counts, inputs and frozen experiment chain."""
from pathlib import Path
import hashlib,json,subprocess,sys,re
from html.parser import HTMLParser
R=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
n=0
for line in (R/'artifact_hashes.sha256').read_text().splitlines():
 expected,name=line.split('  ',1);assert sha(R/name)==expected,name;n+=1
for p,expected in read(R/'input_hashes.json').items():assert sha(p)==expected,p
labels=read(R/'GLOSS_LABELS_AND_COUNTS.json');counts=read(R/'COUNTS.json');assert len(labels)==3370
trained=[r for r in labels if r['ever_used_as_TRAIN_target']];assert len(trained)==3256
for name,expected in [('ALL_ELIGIBLE_GLOSS_LABELS.txt',3370),('ALL_TRAINED_GLOSS_LABELS.txt',3256),('ASLLRP3_FULL_GLOSS_LIST.txt',2739),('NCSLGR2_FULL_GLOSS_LIST.txt',1062)]:
 rows=[s for s in (R/name).read_text(encoding='utf-8').splitlines() if re.match(r'^\d+\. ',s)];assert len(rows)==expected,name
inventory=[json.loads(s) for s in (R/'UTTERANCE_DATA_INVENTORY.jsonl').read_text(encoding='utf-8').splitlines()];assert len(inventory)==len({r['sample_id'] for r in inventory})==3121
assert sum(r['ever_used_in_verified_continuous_fit'] for r in inventory)==3025
assert sum(r['gloss_tokens'] for r in inventory)==23002
class ReportParser(HTMLParser):
 def __init__(self):super().__init__();self.wordrows=0;self.links=[]
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if tag=='tr' and a.get('class')=='gloss-row':self.wordrows+=1
  if tag=='a' and a.get('href'):self.links.append(a['href'])
p=ReportParser();p.feed((R/'FULL_TRAINING_REPORT.html').read_text(encoding='utf-8'));assert p.wordrows==3370
for name in p.links:assert (R/name).is_file(),name
for row in read(R/'TRAINING_RUNS.json'):
 if row['checkpoint_hash_verified']:assert sha(row['checkpoint_path'])==row['checkpoint_sha256']
subprocess.run([sys.executable,'-B',str(R.parent/'20261020T133148Z_source_transfer_pilot01/verify.py')],check=True)
print(json.dumps({'status':'PASS_COMPLETE_GLOSS_AND_TRAINING_REPORT','report_files':n,'eligible_gloss_spellings':3370,'trained_gloss_spellings':3256,'eligible_utterances':3121,'actually_fitted_utterances':3025,'new_training_performed':False,'TEST_contents_exported':False}))
