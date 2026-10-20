"""Resolve a publisher URL list only by its unique exact EAF media basename."""
from collections import defaultdict
from pathlib import PurePosixPath
import hashlib,json,urllib.parse
from core import RUN,OLD,ROOT,read,write,sha,fetch

def main():
 original=read(OLD/'depaul_catalog/plan.json');groups=defaultdict(list)
 for row in original['rows']:
  if row['role']=='NEW_RESEARCH_TRAIN_CANDIDATE' and ';' in row['source_url']:groups[row['source_url']].append(row)
 rows=[];evidence=[]
 known={r['sample_id'] for r in read(OLD/'canonical/manifest.json')}
 for field,items in groups.items():
  descriptor_names={PurePosixPath(urllib.parse.unquote(urllib.parse.urlsplit(r['annotation_provenance']['media_descriptor']['MEDIA_URL']).path)).name.casefold() for r in items}
  if len(descriptor_names)!=1:raise ValueError('MULTIPLE_EAF_MEDIA_IDENTITIES')
  wanted=next(iter(descriptor_names));options=field.split(';')
  matches=[u for u in options if PurePosixPath(urllib.parse.urlsplit(u).path).name.casefold()==wanted]
  if len(matches)!=1:raise ValueError('AMBIGUOUS_PUBLISHER_URL_OPTIONS')
  parts=urllib.parse.urlsplit(matches[0])
  if parts.hostname!='csr.bu.edu' or not parts.path.endswith('_small_0.mov'):raise ValueError('NONOFFICIAL_OR_NONFRONTAL_SOURCE')
  url=urllib.parse.urlunsplit(('https',parts.netloc,parts.path,parts.query,''))
  ordered=sorted(items,key=lambda r:(r['start_ms'],r['end_ms']))
  if any(a['end_ms']>b['start_ms'] for a,b in zip(ordered,ordered[1:])):raise ValueError('OVERLAPPING_SOURCE_UTTERANCE_SPANS')
  if known&{r['sample_id'] for r in items}:raise ValueError('ALREADY_PREPARED_ID')
  key=hashlib.sha256(url.encode()).hexdigest()[:20]
  for row in items:rows.append({**row,'source_url':url,'source_url_original':row['source_url'],'publisher_url_resolution':{'original_field':field,'all_publisher_options':options,'EAF_basename':wanted,'matching_original_URL':matches[0],'method':'Exactly one listed URL basename matches the authentic EAF MEDIA_DESCRIPTOR. No guessed media path, offset or label repair.'}})
  evidence.append({'source_url':url,'EAF_basename':wanted,'utterances':len(items),'url_components':options,'matching_components':len(matches),'raw_output':f'data/raw/ncslgr_media/{key}.mov'})
 if not (RUN/'depaul_catalog/plan.json').exists():write(RUN/'depaul_catalog/plan.json',{'rows':rows,'old_plan_sha256':sha(OLD/'depaul_catalog/plan.json'),'resolution_evidence':evidence,'target_label_changes':False,'timestamp_changes':False,'status':'PASS_UNIQUE_PUBLISHER_MEDIA_MATCH'})
 for item in evidence:fetch(item['source_url'],PurePosixPath(item['source_url']).name,cap=536870912,target_relative=item['raw_output'])
 source=OLD/'tools/prepare_timed.py';target=RUN/'prepare_new_ncslgr.py'
 if not target.exists():
  with target.open('x',encoding='utf-8') as f:f.write(source.read_text(encoding='utf-8'))
  write(RUN/'ncslgr_adapter_provenance.json',{'source_path':str(source),'source_sha256':sha(source),'copied_path':str(target),'copied_sha256':sha(target),'source_algorithm_changed':False,'new_input_plan_only':True,'all_previous_exclusions_preserved':True})
 print(json.dumps({'status':'NEW_URL_LIST_JOINS_RESOLVED','candidate_utterances':len(rows),'new_sources':len(evidence)}),flush=True)

if __name__=='__main__':main()
