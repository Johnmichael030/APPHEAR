"""Read-only permission discovery. Never opens participant video or feature arrays."""
from pathlib import Path
import json,re,zipfile,xml.etree.ElementTree as ET
from collections import Counter
import fine_tune as ft

root=Path('D:/ASL VIDEOS DATASETS')
files=list(root.rglob('*'));files=[p for p in files if p.is_file()]
named=[str(p) for p in files if re.search(r'consent|permission|licen[sc]e|agreement|release|verified|validation',p.name,re.I)]
keyhits=Counter();review_values=Counter();docs=[]
def visit(x):
 if isinstance(x,dict):
  for k,v in x.items():
   if re.search(r'consent|permission|licen[sc]e|review|verified',k,re.I):
    keyhits[k]+=1
    if k=='reviewer_notes':review_values['empty' if not v else 'nonempty']+=1
   visit(v)
 elif isinstance(x,list):
  for v in x:visit(v)
for p in files:
 if p.suffix.lower()=='.json':
  try:visit(ft.read(p))
  except (ValueError,UnicodeError):pass
 if p.suffix.lower()=='.docx':
  with zipfile.ZipFile(p) as z:
   tree=ET.fromstring(z.read('word/document.xml'))
  ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
  paragraphs=[''.join(t.text or '' for t in q.findall('.//w:t',ns)) for q in tree.findall('.//w:p',ns)]
  hits=[dict(paragraph=i+1,text=t) for i,t in enumerate(paragraphs) if re.search(r'consent|permission|ethical|reviewer|annotation|informed',t,re.I)]
  docs.append(dict(path=str(p),sha256=ft.sha(p),paragraphs=len(paragraphs),permission_related_paragraphs=hits))
ft.save(ft.RUN/'LOCAL_PERMISSION_SCAN.json',dict(user_supplied_path=str(root),file_count=len(files),named_candidate_records=named,JSON_key_counts=dict(keyhits),reviewer_notes_values=dict(review_values),documents=docs,participant_media_opened=False,status='REQUIRES_HUMAN_EVIDENCE_INTERPRETATION'))
print(json.dumps(dict(files=len(files),named_candidate_records=named,key_counts=dict(keyhits),reviewer_notes_values=dict(review_values),docx_hits=[dict(path=r['path'],paragraphs=r['permission_related_paragraphs']) for r in docs]),ensure_ascii=False))
