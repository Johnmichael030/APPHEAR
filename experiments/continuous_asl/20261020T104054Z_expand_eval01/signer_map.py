"""Verify publisher signer IDs through the ordinary public utterance-search form."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import html,json,re,urllib.parse
from core import RUN,OLD,fetch,read,write,sha
from intake_metadata import Fields

def parse(path):
 text=path.read_text(encoding='utf-8',errors='strict');parser=Fields();parser.feed(text)
 identities=[]
 for content in re.findall(r'<td\s+class="helpBod datasource"[^>]*>(.*?)</td>',text,re.S):
  identity=html.unescape(re.sub(r'<[^>]*>','',content)).strip()
  if not re.fullmatch(r'.+-U-[0-9]+',identity):raise ValueError('PUBLISHER_UTTERANCE_ID_FORMAT:'+identity)
  collection,utterance=identity.rsplit('-U-',1);identities.append((collection,utterance))
 links={}
 for link in parser.links:
  href=link.get('href','');url=urllib.parse.urljoin('https://dai.cs.rutgers.edu/dai/s/utterancesearch',href);query=urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
  if 'page_num' in query and 'utterancesearchresult' in url:links[int(query['page_num'][0])]=url
 return identities,links

def main():
 output=RUN/'signer_metadata.json'
 if output.exists():print('SIGNER_METADATA_COMPLETE_NO_REPEAT');return
 parser=Fields();parser.feed((RUN/'sources/utterancesearch.html').read_text(encoding='utf-8'));form=parser.forms[0]
 pairs=[(r['name'],r.get('value','')) for r in form['inputs'] if r.get('name') and r.get('name')!='participant' and (r.get('type') not in ('checkbox','radio') or 'checked' in r)]
 urlbase=urllib.parse.urljoin('https://dai.cs.rutgers.edu/dai/s/utterancesearch',form['attributes']['action'])
 mapping={};reports=[]
 for participant,name in [('4','Jonathan'),('2','Cory'),('3','Rachel'),('5','Ben')]:
  url=urlbase+'?'+urllib.parse.urlencode(pairs+[('participant',participant)])
  page1=fetch(url,f'utterances_participant{participant}_page1.html')
  first,links=parse(page1)
  # Only pagination destinations actually present in the official response are followed.
  pages={1:page1};todo=dict(links);todo.pop(1,None)
  while todo:
   batch=sorted((n,u) for n,u in todo.items() if n not in pages)
   if not batch:break
   if len(pages)+len(batch)>100:raise ValueError('METADATA_PAGE_BUDGET')
   def get(item):
    n,u=item
    if urllib.parse.parse_qs(urllib.parse.urlsplit(u).query).get('participant')!=[participant]:raise ValueError('PAGINATION_SIGNER_FILTER_CHANGED')
    return n,fetch(u,f'utterances_participant{participant}_page{n}.html')
   with ThreadPoolExecutor(max_workers=3) as executor:
    for n,path in executor.map(get,batch):pages[n]=path
   todo={}
   for path in pages.values():
    _,more=parse(path);todo.update({n:u for n,u in more.items() if n not in pages})
  identities=[]
  for n,path in sorted(pages.items()):
   values,_=parse(path);identities.extend(values)
   for collection,utterance in values:
    key=collection+'|'+utterance
    entry={'publisher_participant_id':participant,'publisher_participant_name':name,'signer_id':'asllrp_dai_participant_'+participant,'collection':collection,'utterance_id':utterance,
           'source_page':str(path),'source_page_sha256':sha(path),'query_receipt':str(path)+'.receipt.json','method':'Official public participant-filtered utterance search, exhaustive published pagination.'}
    if key in mapping and mapping[key]['publisher_participant_id']!=participant:raise ValueError('AMBIGUOUS_PUBLISHER_SIGNER')
    mapping[key]=entry
  if len(identities)!=len(set(identities)):raise ValueError('REPEATED_UTTERANCE_ACROSS_PAGES')
  reports.append({'participant':participant,'name':name,'utterances':len(identities),'pages':len(pages),'collections':sorted({c for c,u in identities})})
  print(json.dumps(reports[-1]),flush=True)
 canonical=read(OLD/'canonical/manifest.json');eligible=[r for r in canonical if r['canonical_role']=='eligible_train' and r['training_pool'].startswith('ASLLRP3')]
 missing=[r['sample_id'] for r in eligible if r['collection']+'|'+r['utterance_id'] not in mapping]
 joined={r['sample_id']:mapping[r['collection']+'|'+r['utterance_id']] for r in eligible if r['collection']+'|'+r['utterance_id'] in mapping}
 write(output,{'status':'PASS_PUBLISHER_SIGNER_JOINS' if not missing else 'PARTIAL_UNMAPPED_SIGNERS','publisher_queries':reports,'joined_eligible_records':joined,'missing':missing,
  'coverage':dict(Counter(r['signer_id'] for r in joined.values())),'original_manifests_modified':False,'signer_scope':'Publisher database participant IDs within this ASLLRP corpus; no facial inference or undocumented cross-corpus identity merge.'})
 print(json.dumps({'status':'SIGNER_JOIN_COMPLETE','joined':len(joined),'missing':len(missing)}),flush=True)

if __name__=='__main__':main()
