"""Reuse the exercised literal-event and media gates for a separate RIT pool."""
from core import RUN,OLD,write,sha

def main():
 specs=[('asllrp_catalog.py','rit_catalog.py',[
  ("RUN/'asllrp2025_catalog'","RUN/'rit2025_catalog'"),
  ("RUN/'data/raw/asllrp2025/asllrp_sentence_signs_2025_06_28.csv'","RUN/'sources/rit_sentence_signs_2025_11_01.csv'"),
  ("'asllrp3_'","'rit3_'"),("'ASLLRP_SIGNSTREAM3_2025'","'RIT_SENTENCES_SIGNSTREAM3_2025'"),
  ('https://dai.cs.rutgers.edu/asllvd/signbank/asllrp_videos/batch_utterance_video_v3_1.zip','https://dai.cs.rutgers.edu/asllvd/signbank/rit3/batch_utterance_video_v1_1.zip')]),
  ('audit_asllrp_boundaries.py','audit_rit_boundaries.py',[('asllrp2025_catalog','rit2025_catalog')]),
  ('asllrp_archive.py','rit_archive.py',[('asllrp2025_catalog','rit2025_catalog'),
   ("RUN/'data/raw/asllrp2025/batch_utterance_video_v3_1.zip'","RUN/'sources/rit_batch_utterance_video_v1_1.zip'"),
   ("RUN/'data/raw/asllrp2025/utterances'","RUN/'data/raw/rit_utterances'")]),
  ('prepare_asllrp.py','prepare_rit.py',[
   ('asllrp2025_catalog','rit2025_catalog'),("RUN/'data/processed/asllrp2025'","RUN/'data/processed/rit2025'"),
   ("rows=read(RUN/'rit2025_catalog/media_manifest.json')","rows=read(RUN/'rit2025_catalog/media_manifest.json');excluded=set(read(RUN/'rit2025_catalog/annotation_boundary_audit.v2.json')['reject_sample_ids']);rows=[r for r in rows if r['sample_id'] not in excluded]"),
   ("if not 0<=shard<shards<=4:raise ValueError('SHARD_CONFIGURATION')","if not 0<=shard<shards<=4:raise ValueError('SHARD_CONFIGURATION')"),
   ("ROOT=RUN.parents[2]","ROOT=RUN.parents[2]")])]
 provenance=[]
 for source_name,target_name,replacements in specs:
  source=OLD/'tools'/source_name;target=RUN/target_name
  if target.exists():continue
  text=source.read_text(encoding='utf-8')
  for before,after in replacements:
   if before not in text:
    if before==after:continue
    raise ValueError('ADAPTER_LITERAL_NOT_FOUND:'+before)
   text=text.replace(before,after)
  with target.open('x',encoding='utf-8') as f:f.write(text)
  provenance.append({'source':str(source),'source_sha256':sha(source),'new_path':str(target),'new_sha256':sha(target),'changes':replacements})
 if provenance:write(RUN/'rit_adapter_provenance.json',{'files':provenance,'math_changed':False,'purpose':'Separate official RIT continuous sentences; existing exact frame/schema/CTC gates, no trimming/label normalization.'})

if __name__=='__main__':main()
