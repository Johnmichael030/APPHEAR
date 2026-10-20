"""Bound the two primary workers to indices below the tail-owned range."""
import ast
from core import RUN,write,sha

def main():
 source_path=RUN/'tools/prepare_asllrp.py';source=source_path.read_text(encoding='utf-8');tree=ast.parse(source)
 node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');body=ast.get_source_segment(source,node)
 anchor='rows=[r for i,r in enumerate(rows) if i%shards==shard]'
 if anchor not in body:raise ValueError('FRONT_RANGE_ANCHOR')
 body=body.replace(anchor,'rows=[r for i,r in enumerate(rows) if i<1300 and i%shards==shard]')
 body=body.replace("f'worker_{shard}_of_{shards}_{time.time_ns()}.json'","f'front1300_{shard}_of_{shards}_{time.time_ns()}.json'")
 header='''from dataclasses import replace
import hashlib,json,os,shutil,time
from pathlib import Path
import numpy as np
from core import RUN,ROOT,SCHEMA,read,write,sha
from prepare_asllrp import memory_available,check_media_contract
'''
 code=header+'\n'+body+'''\n\nif __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);a=p.parse_args();main(a.shard,2,False)
'''
 destination=RUN/'tools/prepare_asllrp_front.py'
 ast.parse(code)
 with destination.open('x',encoding='utf-8') as f:f.write(code)
 write(RUN/'asllrp2025_catalog/front_worker_provenance.json',{'source_sha256':sha(source_path),'port_sha256':sha(destination),
       'ranges':{'front_shard0':'indices<1300,even','front_shard1':'indices<1300,odd','tail1':'1300<=index<1700','tail2':'1700<=index<2126'},
       'reason':'Bound primary workers before they catch up to tail ranges. Existing completed row/feature hashes reused; no feature math or sampling changes.',
       'maximum_concurrent_runtime_workers':4})

if __name__=='__main__':main()
