"""Port identical extraction for disjoint future ranges; keep running code frozen."""
import ast
import hashlib
from core import RUN,read,write,sha

def main():
    source_path=RUN/'tools/prepare_asllrp.py';source=source_path.read_text(encoding='utf-8')
    tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    body=ast.get_source_segment(source,node)
    before='def main(shard=0,shards=1,pilot=False):'
    body=body.replace(before,'def main(start_index,stop_index):\n    shard=0;shards=1;pilot=False')
    body=body.replace("rows=read(RUN/'asllrp2025_catalog/media_manifest.json')", "rows=read(RUN/'asllrp2025_catalog/media_manifest.json')\n    allrows=list(rows)\n    positions={r['sample_id']:i for i,r in enumerate(allrows)}")
    anchor='rows=[r for i,r in enumerate(rows) if i%shards==shard]'
    if anchor not in body:raise ValueError('TAIL_SHARD_ANCHOR')
    body=body.replace(anchor,'rows=[r for i,r in enumerate(rows) if start_index<=i<stop_index]')
    anchor="            if memory_available()<1073741824 or shutil.disk_usage(RUN).free<10737418240:"
    if anchor not in body:raise ValueError('TAIL_RESOURCE_ANCHOR')
    body=body.replace(anchor,"""            frontier=next((i for i,r in enumerate(allrows) if not (output/(r['sample_id']+'.row.json')).exists() and not (output/(r['sample_id']+'.failure.json')).exists()),len(allrows))
            if positions[sample]-frontier<400:
                resource_stop='GRACEFUL_STOP_PRIMARY_FRONTIER_WITHIN400_SAMPLES_NO_DUPLICATE_WORK';break
"""+anchor)
    body=body.replace("f'worker_{shard}_of_{shards}_{time.time_ns()}.json'","f'tail_{start_index}_{stop_index}_{time.time_ns()}.json'")
    header='''from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from core import RUN,ROOT,SCHEMA,read,write,sha
from prepare_asllrp import memory_available,check_media_contract
'''
    destination=RUN/'tools/prepare_asllrp_tail.py'
    code=header+'\n'+body+'''\n\nif __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--start',type=int,required=True);p.add_argument('--stop',type=int,required=True)
    a=p.parse_args();main(a.start,a.stop)
'''
    with destination.open('x',encoding='utf-8') as f:f.write(code)
    write(RUN/'asllrp2025_catalog/tail_worker_provenance.json',{'source_sha256':sha(source_path),'port_sha256':sha(destination),
          'changes':'Only future disjoint index ranges and graceful stop at400-sample primary frontier; feature math/timing/CTC and sampling unchanged.',
          'ranges':[[1300,1700],[1700,2126]],'maximum_concurrent_runtime_workers':4,
          'available_physical_RAM_GiB_before_extra_workers':4.6549072265625,'logical_CPU_count':16,'minimum_RAM_headroom_GiB':1})

if __name__=='__main__':main()
