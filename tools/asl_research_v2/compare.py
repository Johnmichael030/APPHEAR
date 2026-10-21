"""Paired validation comparisons and parity for a promising isolated candidate."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from common import PREVIOUS,run_root,write,sha
from warmstart_a import LegacyTCN
from train_a import load,batch

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True,type=Path);args=p.parse_args();run=run_root(args.run_root);a=run/'experiment_A';warm=a/'warmstart'
    torch.set_num_threads(2);data=load(run,'val');y=data['y'];ids=data['participant_ids']
    base=json.loads((warm/'baseline_fp32.json').read_text());candidate=json.loads((warm/'validation_best.json').read_text())
    bp=np.asarray(base['predictions']);cp=np.asarray(candidate['predictions']);rng=np.random.default_rng(20261020);signers=np.unique(ids);bootstrap=[]
    for _ in range(2000):
        selected=rng.choice(signers,len(signers),replace=True);idx=np.concatenate([np.flatnonzero(ids==s) for s in selected]);bootstrap.append(float((cp[idx]==y[idx]).mean()-(bp[idx]==y[idx]).mean()))
    delta=float((cp==y).mean()-(bp==y).mean());interval=np.percentile(bootstrap,[2.5,97.5]).tolist()
    probs=np.load(warm/'validation_best_probabilities.npy',allow_pickle=False);confidence=probs.max(1);curve=[]
    for quantile in (0,.25,.5,.75,.9):
        threshold=float(np.quantile(confidence,quantile));keep=confidence>=threshold
        curve.append({'validation_confidence_threshold':threshold,'coverage':float(keep.mean()),'accepted_top1':float((cp[keep]==y[keep]).mean()),
            'positive_sign_rejection_rate':float((~keep).mean()),'false_activations':'NOT_MEASURED_NO_NEGATIVES','threshold_selected_for_runtime':False})
    # Missing-hand proxy in legacy258: a full 21x3 side block equals zero.
    coverage=[]
    for i in range(0,len(y),128):
        x=data['X'][i:i+128];mask=data['mask'][i:i+128]
        absent=(np.abs(x[...,:63]).max(-1)<1e-8)&(np.abs(x[...,63:126]).max(-1)<1e-8)
        coverage.extend(((absent&mask).sum(1)/mask.sum(1)).tolist())
    coverage=np.asarray(coverage);groups=[]
    for lo,hi in ((0,.25),(.25,.5),(.5,.75),(.75,1.0001)):
        keep=(coverage>=lo)&(coverage<hi)
        if keep.any():groups.append({'both_hands_absent_fraction_range':[lo,min(hi,1.)],'rows':int(keep.sum()),
            'baseline_top1':float((bp[keep]==y[keep]).mean()),'candidate_top1':float((cp[keep]==y[keep]).mean())})
    analysis={'matched_baseline':'FP32 original Keras transferred with parity','top1_delta':delta,'macro_f1_delta':candidate['macro_f1']-base['macro_f1'],
        'paired_signer_bootstrap_top1_delta_95pct':interval,'bootstrap_repetitions':2000,'validation_signers':len(signers),
        'uncertainty_limit':'Only five signer clusters; checkpoint selection used this validation set; no independent final test claim',
        'confidence_coverage_curve_positive_only':curve,'missing_hand_proxy_groups':groups,
        'supported_classes':236,'new_fingerspelling_candidate':'BLOCKED_NO_VERIFIED_LETTER_SEQUENCE_TRAINING_DATA',
        'false_activation_improvement':'NOT_MEASURED','production_promotion':False}
    write(run/'paired_comparison.json',analysis)
    with (run/'comparison_results.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=('task','candidate','metric','value','comparison','status'));w.writeheader()
        for name,path in (('frozen_legacy_INT8',a/'baseline.json'),('frozen_legacy_FP32',warm/'baseline_fp32.json'),('scratch_5epoch',a/'validation_best.json'),('warmstart_clip_objective',warm/'validation_best.json')):
            result=json.loads(path.read_text())
            for metric in ('top1','macro_f1'):w.writerow(dict(task='isolated_gloss',candidate=name,metric=metric,value=result[metric],comparison='same 3264 validation rows; five held-out signer IDs',status='DIAGNOSTIC_VALIDATION'))
        for name,file in (('prior_10Hz','verified_10hz_baseline.json'),('prior_model_on_native','frozen_model_native_cadence.json'),('native_retrained','validation_best.json')):
            result=json.loads((run/'experiment_B'/file).read_text());w.writerow(dict(task='continuous_gloss',candidate=name,metric='gloss_WER',value=result['gloss_wer'],comparison='same seven complete reference clips, signer IDs unknown',status='DIAGNOSTIC'))
    for name,result in (('baseline',base),('candidate',candidate)):
        np.savetxt(warm/f'{name}_confusion_matrix.csv',np.asarray(result['confusion_matrix']),fmt='%d',delimiter=',')
    # Only a candidate improving the matched validation metrics warrants an export probe.
    if delta<=0 or candidate['macro_f1']<=base['macro_f1']:
        write(run/'export_results.json',{'status':'NOT_RUN','reason':'No improvement against matched baseline'});return
    pointer=json.loads((warm/'checkpoints/best.json').read_text());checkpoint=warm/pointer['path']
    if sha(checkpoint)!=pointer['sha256']:raise ValueError('BEST_CHECKPOINT_HASH')
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False);contract=json.loads((warm/'source_contract.json').read_text());model=LegacyTCN(contract['configuration']['0_ln1']['epsilon']).eval();model.load_state_dict(ck['model'])
    import onnx
    import onnxruntime as ort
    out=warm/'export';out.mkdir();path=out/'isolated_legacy258_clip_fp32.onnx'
    x,m,_=batch(data,np.asarray([0]),'cpu')
    torch.onnx.export(model,(x,m),str(path),dynamo=False,opset_version=17,input_names=['features','frame_mask'],output_names=['class_probabilities'],
        dynamic_axes={'features':{0:'batch',1:'time'},'frame_mask':{0:'batch',1:'time'},'class_probabilities':{0:'batch'}})
    onnx.checker.check_model(str(path),full_check=True);options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    tick=time.perf_counter();session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider']);cold=(time.perf_counter()-tick)*1000
    maximum=0.;total_error=0.;elements=0;changes=0
    with torch.inference_mode():
        for start in range(0,len(y),32):
            x,m,_=batch(data,np.arange(start,min(start+32,len(y))),'cpu');expected=model(x,m).numpy();actual=session.run(None,{'features':x.numpy(),'frame_mask':m.numpy()})[0]
            error=np.abs(expected-actual);maximum=max(maximum,float(error.max()));total_error+=float(error.sum());elements+=error.size;changes+=int(np.count_nonzero(expected.argmax(-1)!=actual.argmax(-1)))
    cpu=[];gpu=[];gpu_model=LegacyTCN(contract['configuration']['0_ln1']['epsilon']).cuda().eval();gpu_model.load_state_dict(ck['model'])
    with torch.inference_mode():
        for i in range(110):
            x,m,_=batch(data,np.asarray([i]),'cpu');tick=time.perf_counter();session.run(None,{'features':x.numpy(),'frame_mask':m.numpy()});elapsed=(time.perf_counter()-tick)*1000
            gx,gm=x.cuda(),m.cuda();torch.cuda.synchronize();gt=time.perf_counter();gpu_model(gx,gm);torch.cuda.synchronize();ge=(time.perf_counter()-gt)*1000
            if i>=10:cpu.append(elapsed);gpu.append(ge)
    # Causal per-frame RF test uses synthetic inputs solely as an engineering probe.
    with torch.inference_mode():
        probe=torch.randn(1,160,258);a1=model.per_frame(probe)[:,-1];a2=model.per_frame(probe[:,-61:])[:,-1];rf_error=float((a1-a2).abs().max())
        empty=torch.zeros(1,1,258);empty_mask=torch.zeros(1,1,dtype=torch.bool);empty_output=model(empty,empty_mask)
    report={'status':'PASS' if maximum<1e-4 and changes==0 else 'FAIL','onnx_path':str(path),'onnx_sha256':sha(path),'bytes':path.stat().st_size,
        'real_validation_rows':len(y),'max_abs_probability_error':maximum,'mean_abs_probability_error':total_error/elements,'decoded_class_changes':changes,
        'CPU_ORT_p50_ms':float(np.percentile(cpu,50)),'CPU_ORT_p95_ms':float(np.percentile(cpu,95)),
        'CUDA_PyTorch_p50_ms':float(np.percentile(gpu,50)),'CUDA_PyTorch_p95_ms':float(np.percentile(gpu,95)),
        'cold_ORT_session_ms':cold,'61_observation_receptive_field_max_error':rf_error,'synthetic_long_sequence_length':160,
        'empty_mask_output_finite':bool(torch.isfinite(empty_output).all()),'empty_mask_runtime_policy':'REJECT_INPUT; all-zero output must never be argmax-committed',
        'input':'float32 [B,T,258] plus bool frame_mask; legacy schema3.0','output':'236 clip probabilities in unchanged vocabulary order',
        'Android':'NOT_INTEGRATED; current RecognitionV2 350 input incompatible','non_sign_rejection':'NOT_CALIBRATED'}
    write(run/'export_results.json',report);print(json.dumps({k:report[k] for k in ('status','max_abs_probability_error','decoded_class_changes','bytes','CPU_ORT_p50_ms','CPU_ORT_p95_ms')}))
if __name__=='__main__':main()
