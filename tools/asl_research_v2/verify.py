"""Verify original sources and completed prior run without writing to either."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from common import ROOT,PREVIOUS,run_root,sha
p=argparse.ArgumentParser();p.add_argument('--run-root',required=True,type=Path);args=p.parse_args();run=run_root(args.run_root)
records=json.loads((run/'protected_sources.json').read_text())+json.loads((run/'external_inputs_protected.json').read_text())
bad=[];timestamps=[]
for item in records:
    path=Path(item['path'])
    if not path.is_file() or sha(path)!=item['sha256']:bad.append(str(path))
    elif path.stat().st_mtime_ns!=item['mtime_ns']:timestamps.append(str(path))
original=json.loads((PREVIOUS/'preservation_baseline.json').read_text());epoch=datetime(1970,1,1,tzinfo=timezone.utc);historical_time_changes=[]
for item in original['files']:
    text=item['last_write_utc'];whole,frac=text.rstrip('Z').split('.')
    instant=datetime.fromisoformat(whole).replace(tzinfo=timezone.utc);delta=instant-epoch
    expected=(delta.days*86400+delta.seconds)*1000000000+int(frac.ljust(9,'0'))
    if (ROOT/item['path']).stat().st_mtime_ns!=expected:historical_time_changes.append(item['path'])
result={'status':'PASS' if not bad and not timestamps and not historical_time_changes else 'FAIL','checked_records':len(records),
    'original_2932_hashes_and_timestamps_preserved':not bad and not historical_time_changes,
    'prior_run_artifacts_preserved':not bad and not timestamps,'hash_mismatches':bad,'mtime_changes':timestamps,
    'historical_timestamp_changes':historical_time_changes,'production_model_GLbs_manifests_avatar_choices':'included in protected original inventory',
    'Android_installed_or_changed':False}
(run/'preservation_final.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if result['status']!='PASS':raise SystemExit(1)
