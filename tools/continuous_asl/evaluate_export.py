"""Measured diagnostic evaluation/export. Does not deploy or calibrate rejection."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from ctc_experiment import CausalCTC,decode,evaluate,load_data
from prepare_features import SCHEMA,checked_run,sha
from pipeline.perception_vnext.recognition_schema import RawSequence,load_raw_sequence
from pipeline.perception_vnext.segmentation.learned_features import derive_causal_segmentation_features

def fresh_json(path: Path, value: object) -> None:
    with path.open('x',encoding='utf-8') as f: json.dump(value,f,indent=2);f.write('\n')

def perturbed_data(run: Path, rows: list[dict], kind: str) -> dict:
    data={}
    for row in rows:
        raw=load_raw_sequence((run/row['feature_path']).with_name(row['sample_id']+'.raw.npz'))
        arrays={k:getattr(raw,k).copy() for k in RawSequence.__dataclass_fields__ if k!='metadata'}
        if kind=='drop_every_third_observation':
            keep=np.arange(len(raw.frame_index))%3!=2
            arrays={k:v[keep] for k,v in arrays.items()}
        elif kind=='both_hands_unobserved':
            arrays['hand_joint_valid'][:]=False;arrays['hand_observed'][:]=False
            arrays['hand_proposal_present'][:]=False;arrays['hand_xy'][:]=np.nan
            arrays['hand_presence'][:]=np.nan;arrays['hand_roi_center_side'][:]=np.nan
            arrays['hand_proposal_support'][:]=0
        else: raise ValueError('STRESS_KIND')
        value=RawSequence(metadata=raw.metadata,**arrays)
        data[row['sample_id']]=derive_causal_segmentation_features(value)
    return data

def aligned_pairs(ref: list[str],hyp: list[str]) -> list[tuple[str,str]]:
    n,m=len(ref),len(hyp);d=np.zeros((n+1,m+1),np.int32);d[:,0]=np.arange(n+1);d[0,:]=np.arange(m+1)
    for i in range(1,n+1):
        for j in range(1,m+1): d[i,j]=min(d[i-1,j-1]+(ref[i-1]!=hyp[j-1]),d[i-1,j]+1,d[i,j-1]+1)
    i,j=n,m;pairs=[]
    while i or j:
        if i and j and d[i,j]==d[i-1,j-1]+(ref[i-1]!=hyp[j-1]): pairs.append((ref[i-1],hyp[j-1]));i-=1;j-=1
        elif i and d[i,j]==d[i-1,j]+1: pairs.append((ref[i-1],'<DELETION>'));i-=1
        else: pairs.append(('<INSERTION>',hyp[j-1]));j-=1
    return pairs[::-1]

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);args=p.parse_args();run=checked_run(args.run_root)
    torch.set_num_threads(2);rows,data=load_data(run); val=[r for r in rows if r['split']=='val']
    pointer=json.loads((run/'checkpoints/best.json').read_text());checkpoint=(run/pointer['path']).resolve();checkpoint.relative_to(run/'checkpoints')
    if sha(checkpoint)!=pointer['sha256']: raise ValueError('CHECKPOINT_HASH')
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ckpt['schema']!=SCHEMA or ckpt['manifest_sha256']!=sha(run/'data/processed/full/manifest.json'): raise ValueError('CHECKPOINT_CONTRACT')
    vocab=json.loads((run/'vocabulary.json').read_text())['tokens'];state=ckpt['model']
    model=CausalCTC(len(vocab),ckpt['config']['channels'],state['mean'].numpy(),state['std'].numpy()).eval();model.load_state_dict(state)
    result=evaluate(model,val,data,vocab,'cpu');result['checkpoint']=pointer;result['untrained_baseline']=json.loads((run/'logs/untrained_baseline.json').read_text())
    result['stress_tests']={}
    for kind in ('drop_every_third_observation','both_hands_unobserved'):
        stress=evaluate(model,val,perturbed_data(run,val,kind),vocab,'cpu')
        result['stress_tests'][kind]=stress
    result.update(nonsign_evaluation='BLOCKED_USER_CONFIRMED_NO_ADDITIONAL_LABELED_DATA',
        boundary_evaluation='BLOCKED_NO_TIMED_SIGN_BOUNDARIES',cross_signer_evaluation='BLOCKED_SIGNER_IDS_UNKNOWN',
        isolated_accuracy_and_macro_f1='NOT_APPLICABLE_CONTINUOUS_SEQUENCE_TASK',production_comparison='BLOCKED_INCOMPATIBLE_TASK_AND_NO_PRODUCT_BENCHMARK',
        release_decision='DO_NOT_PROMOTE_OR_INTEGRATE')
    fresh_json(run/'evaluation_results.json',result)
    with (run/'sequence_errors.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=('sample_id','reference','hypothesis','substitutions','deletions','insertions','reference_tokens'));w.writeheader()
        for row in result['predictions']: w.writerow({**row,'reference':' '.join(row['reference']),'hypothesis':' '.join(row['hypothesis'])})
    pairs=Counter(pair for row in result['predictions'] for pair in aligned_pairs(row['reference'],row['hypothesis']))
    with (run/'token_alignment_confusion.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(('reference_token','predicted_token','alignment_count'))
        for (ref,hyp),count in sorted(pairs.items()): w.writerow((ref,hyp,count))
    # Standard ONNX operators, dynamic B/T, full normalization embedded.
    import onnx
    import onnxruntime as ort
    export=run/'export';export.mkdir()
    example=torch.from_numpy(data[val[0]['sample_id']][None])
    path=export/'causal_gloss_ctc_fp32.onnx'
    torch.onnx.export(model,example,str(path),opset_version=17,dynamo=False,
        input_names=['causal378'],output_names=['gloss_logits'],
        dynamic_axes={'causal378':{0:'batch',1:'time'},'gloss_logits':{0:'batch',1:'time'}})
    graph=onnx.load(str(path));onnx.checker.check_model(graph,full_check=True)
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    tick=time.perf_counter();session=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider']);cold_ms=(time.perf_counter()-tick)*1000
    max_error=0.;sum_error=0.;elements=0;frame_mismatches=0;sequence_mismatches=0;parity_rows=[];prefix_max=0.
    gpu_model=None;gpu_max=0.;gpu_sum=0.;gpu_elements=0;gpu_frames=0
    if torch.cuda.is_available():
        gpu_model=CausalCTC(len(vocab),ckpt['config']['channels'],state['mean'].numpy(),state['std'].numpy()).cuda().eval();gpu_model.load_state_dict(state)
    with torch.inference_mode():
        for row in rows:
            x=data[row['sample_id']][None];ref=model(torch.from_numpy(x)).numpy();actual=session.run(None,{'causal378':x})[0]
            err=np.abs(ref-actual);max_error=max(max_error,float(err.max()));sum_error+=float(err.sum());elements+=err.size
            frame_mismatches+=int(np.count_nonzero(ref.argmax(-1)!=actual.argmax(-1)))
            mismatch=decode(torch.from_numpy(ref[0]),vocab)!=decode(torch.from_numpy(actual[0]),vocab);sequence_mismatches+=int(mismatch)
            # Every prefix length exercised; bounded history and full history agree at last output.
            for i in range(len(x[0])):
                prefix=session.run(None,{'causal378':x[:,max(0,i-60):i+1]})[0]
                prefix_max=max(prefix_max,float(np.abs(prefix[0,-1]-actual[0,i]).max()))
            if gpu_model is not None:
                g=gpu_model(torch.from_numpy(x).cuda()).cpu().numpy();e=np.abs(ref-g)
                gpu_max=max(gpu_max,float(e.max()));gpu_sum+=float(e.sum());gpu_elements+=e.size
                gpu_frames+=int(np.count_nonzero(ref.argmax(-1)!=g.argmax(-1)))
            parity_rows.append({'sample_id':row['sample_id'],'length':len(x[0]),'max_abs_logit_error':float(err.max()),'decoded_mismatch':mismatch})
    timings=[];gpu_times=[];torch_cpu_times=[]
    with torch.inference_mode():
        # Latency runs use real chronological prefixes, excluding graph/session construction.
        for row in rows:
            x=data[row['sample_id']]
            for i in range(len(x)):
                prefix=x[max(0,i-60):i+1][None]
                tick=time.perf_counter();session.run(None,{'causal378':prefix});timings.append((time.perf_counter()-tick)*1000)
                tensor=torch.from_numpy(prefix);tick=time.perf_counter();model(tensor);torch_cpu_times.append((time.perf_counter()-tick)*1000)
                if gpu_model is not None:
                    tensor=tensor.cuda();torch.cuda.synchronize();tick=time.perf_counter();gpu_model(tensor);torch.cuda.synchronize();gpu_times.append((time.perf_counter()-tick)*1000)
    passed=max_error<=1e-4 and frame_mismatches==0 and sequence_mismatches==0 and prefix_max<=1e-4
    parity={'status':'PASS' if passed else 'FAILED','checkpoint':pointer,'onnx_sha256':sha(path),'onnx_bytes':path.stat().st_size,
        'opset':17,'operator_types':sorted({n.op_type for n in graph.graph.node}),'providers':session.get_providers(),
        'real_clips_checked':len(rows),'input_shape':'float32 [B,T,378]','output_shape':f'float32 [B,T,{len(vocab)}]',
        'maximum_abs_logit_error':max_error,'mean_abs_logit_error':sum_error/elements,
        'frame_argmax_mismatches':frame_mismatches,'decoded_sequence_mismatches':sequence_mismatches,
        'bounded_history_vs_full_max_abs_error':prefix_max,'per_clip':parity_rows,
        'pytorch_CPU_vs_CUDA':{'max_abs_logit_error':gpu_max,'mean_abs_logit_error':gpu_sum/gpu_elements if gpu_elements else None,'frame_argmax_mismatches':gpu_frames},
        'android_runtime':'NOT_RUN_NO_DEVICE_OR_ADAPTER','onnx_cuda_EP':'UNAVAILABLE_NOT_INSTALLED'}
    fresh_json(run/'model_parity.json',parity)
    def stats(values: list[float]) -> dict:
        return {'samples':len(values),'p50_ms':float(np.percentile(values,50)),'p95_ms':float(np.percentile(values,95)),
            'sum_ms':sum(values),'scope':'real prefixes, warmed runtime, model invocation only'} if values else {'status':'NOT_RUN'}
    performance={'onnx_CPU':stats(timings),'torch_CPU':stats(torch_cpu_times),'torch_CUDA':stats(gpu_times),
        'onnx_session_cold_start_ms':cold_ms,'onnx_bytes':path.stat().st_size,
        'checkpoint_bytes':checkpoint.stat().st_size,'model_only_is_not_end_to_end':True,
        'end_to_end_android_FPS':'NOT_MEASURED','streaming_replay_load':'sequential saved clips, no live camera, no rendering or SentenceBuilder',
        'parameter_count':sum(p.numel() for p in model.parameters())}
    fresh_json(run/'performance_results.json',performance)
    fresh_json(export/'contract.json',{'schema_version':SCHEMA,'model_sha256':sha(path),'input':'float32 B,T,378 chronological genuine observations',
        'output':'unnormalized per-observation gloss logits, greedy CTC blank collapse','vocabulary':vocab,
        'vocabulary_sha256':sha(run/'vocabulary.json'),'normalization':'embedded TRAIN mean/std, clip [-10,10]',
        'history_observations':61,'maximum_cadence_hz':10,'runtime_default_enabled':False,
        'fallback':'continue original AppHear path; experimental asset never selected',
        'nonsign_rejection_supported':False,'individual_boundaries_supported':False,'production_clearance':False,
        'license':'Private noncommercial research only; distribution rights NOT asserted'})
    print(json.dumps({'evaluation_gloss_wer':result['gloss_wer'],'parity_status':parity['status'],
        'max_abs_logit_error':max_error,'onnx_bytes':path.stat().st_size,'performance':performance}),flush=True)
    if not passed: raise SystemExit(1)
if __name__=='__main__': main()
