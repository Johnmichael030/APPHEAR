"""Integrity-check the official archive; extract only exact approved utterances."""
from collections import Counter, defaultdict
from pathlib import PurePosixPath
import json
import os
import shutil
import stat
import zipfile
from core import RUN, read, write, sha

def main():
    folder=RUN/'asllrp2025_catalog'
    if (folder/'media_manifest.json').exists():print('ARCHIVE_ALREADY_VERIFIED_NO_REPEAT');return
    archive_path=RUN/'data/raw/asllrp2025/batch_utterance_video_v3_1.zip'
    receipt=read(str(archive_path)+'.receipt.json')
    if sha(archive_path)!=receipt['sha256']:raise ValueError('ARCHIVE_HASH')
    plan=read(folder/'plan.json');wanted={r['utterance_filename']:r for r in plan['rows']}
    with zipfile.ZipFile(archive_path) as archive:
        infos=archive.infolist()
        if len(infos)>10000 or sum(i.file_size for i in infos)>4294967296:raise ValueError('EXPANSION_BUDGET')
        lookup=defaultdict(list)
        for info in infos:
            path=PurePosixPath(info.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in info.filename or stat.S_ISLNK(info.external_attr>>16):raise ValueError('UNSAFE_ZIP_MEMBER')
            if info.file_size>33554432:raise ValueError('INDIVIDUAL_UTTERANCE_SIZE_CAP')
            lookup[path.name].append(info)
        if (folder/'archive_integrity.json').exists():
            if read(folder/'archive_integrity.json')['archive_sha256']!=receipt['sha256']:raise ValueError('CRC_CACHE_ARCHIVE_CHANGED')
        else:
            bad=archive.testzip()
            if bad:raise ValueError('CRC_FAILURE:'+bad)
            write(folder/'archive_integrity.json',{'status':'PASS_CRC_ALL_MEMBERS','archive_sha256':receipt['sha256'],
                  'members':len(infos),'expanded_bytes':sum(i.file_size for i in infos),'script_sha256':sha(__file__)})
        result=[];missing=[]
        for filename,row in wanted.items():
            matches=lookup[filename]
            if len(matches)!=1:
                missing.append({'sample_id':row['sample_id'],'filename':filename,'reason':'MISSING_OR_AMBIGUOUS_ARCHIVE_MEMBER','matches':len(matches)});continue
            info=matches[0];target=RUN/'data/raw/asllrp2025/utterances'/filename
            target.parent.mkdir(parents=True,exist_ok=True)
            extraction_receipt=target.with_suffix('.mp4.extraction.json')
            if target.exists():
                if not extraction_receipt.exists():raise ValueError('EXISTING_UNMANIFESTED_EXTRACTION')
                prior=read(extraction_receipt)
                if prior['archive_sha256']!=receipt['sha256'] or prior['member']!=info.filename or sha(target)!=prior['sha256']:raise ValueError('EXTRACTION_REUSE_HASH')
                file_hash=prior['sha256']
            else:
                with archive.open(info) as source,target.open('xb') as dest:shutil.copyfileobj(source,dest,1048576)
                if target.stat().st_size!=info.file_size:raise ValueError('EXTRACTION_SIZE')
                file_hash=sha(target)
                write(extraction_receipt,{'archive_sha256':receipt['sha256'],'member':info.filename,'sha256':file_hash,'bytes':info.file_size})
            result.append({**row,'video_path':str(target),'raw_sha256':file_hash,'bytes':info.file_size,
                           'media_provenance':{'official_archive_url':receipt['source_url'],'archive_sha256':receipt['sha256'],
                             'member':info.filename,'crc32':info.CRC},'media_status':'CRC_VERIFIED_NOT_YET_DECODED'})
    write(folder/'media_manifest.json',result)
    write(folder/'media_join_quality.json',{'status':'PASS' if not missing else 'PARTIAL','annotation_candidates':len(wanted),
          'unique_annotation_media_joins':len(result),'missing':missing,'unique_raw_hashes':len({r['raw_sha256'] for r in result}),
          'duplicate_raw_hashes':len(result)-len({r['raw_sha256'] for r in result})})
    print(json.dumps({'joined_media':len(result),'missing':len(missing)}),flush=True)

if __name__=='__main__':main()
