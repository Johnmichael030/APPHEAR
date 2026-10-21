"""Run tests in a copy with writes, subprocesses and network access confined."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
p=argparse.ArgumentParser();p.add_argument('--run-root',required=True,type=Path);p.add_argument('--tag',default='full_regression_final');p.add_argument('--targets',nargs='*',default=['tests']);args=p.parse_args();run=args.run_root.resolve();sandbox=run/'regressions/sandbox_repo'
os.environ['KERAS_HOME']=str(run/'cache/regression_keras');os.environ['MPLCONFIGDIR']=str(run/'cache/regression_matplotlib');os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TMP']=str(run/'regressions/tmp');os.environ['TEMP']=os.environ['TMP'];Path(os.environ['TMP']).mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(sandbox));os.chdir(sandbox)
def allowed(value):
    if isinstance(value,int):return True
    try:
        resolved=str(Path(os.fsdecode(value)).resolve())
        if resolved.startswith('\\\\?\\'):resolved=resolved[4:]
        Path(resolved).relative_to(run);return True
    except (ValueError,TypeError):return False
def audit(event,args):
    if event=='open':
        mode=args[1];flags=args[2];mutating=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        if mutating and not allowed(args[0]):raise PermissionError('RESEARCH_TEST_WRITE_OUTSIDE_RUN')
    elif event in ('os.remove','os.rmdir','os.mkdir','os.chmod','os.utime') and not allowed(args[0]):raise PermissionError('RESEARCH_TEST_MUTATION_OUTSIDE_RUN')
    elif event in ('os.rename','os.replace') and (not allowed(args[0]) or not allowed(args[1])):raise PermissionError('RESEARCH_TEST_MOVE_OUTSIDE_RUN')
    elif event in ('os.link','os.symlink','subprocess.Popen','os.system','socket.connect'):raise PermissionError('RESEARCH_TEST_EXTERNAL_EXECUTION_OR_NETWORK_DISABLED')
sys.addaudithook(audit)
import pytest
start=time.perf_counter();code=pytest.main(args.targets+['-q','--tb=short','--disable-warnings','--continue-on-collection-errors','-p','no:cacheprovider','-o','log_file='+str(run/'logs/pytest_internal.log'),'--basetemp',str(run/'regressions'/('pytest_tmp_'+args.tag)),
    '--junitxml',str(run/'logs'/(args.tag+'.xml'))])
(run/'logs'/(args.tag+'_status.json')).write_text(json.dumps({'exit_code':int(code),'seconds':time.perf_counter()-start,'scope':'full available targets in fresh copied repository; no external writes/processes/network',
    'original_repository_modified':False},indent=2))
raise SystemExit(int(code))
