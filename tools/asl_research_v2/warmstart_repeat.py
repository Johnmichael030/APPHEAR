"""Parity-gated legacy TCN fine-tune using aggregation-aligned clip supervision."""
import argparse
import csv
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import run_root,write,sha
from train_a import load,batch,metrics,atomic_checkpoint
from replay_profile import memory

class LegacyBlock(nn.Module):
    def __init__(self,inputs: int,dilation: int,epsilon: float) -> None:
        super().__init__();self.pad=2*dilation;self.conv1=nn.Conv1d(inputs,64,3,dilation=dilation);self.conv2=nn.Conv1d(64,64,3,dilation=dilation)
        self.ln1=nn.LayerNorm(64,eps=epsilon);self.ln2=nn.LayerNorm(64,eps=epsilon);self.drop=nn.Dropout(.1);self.proj=nn.Conv1d(inputs,64,1) if inputs!=64 else nn.Identity()
    def forward(self,x):
        y=self.conv1(F.pad(x,(self.pad,0)));y=self.drop(F.relu(self.ln1(y.transpose(1,2)))).transpose(1,2)
        y=self.conv2(F.pad(y,(self.pad,0)));y=self.drop(F.relu(self.ln2(y.transpose(1,2)))).transpose(1,2)
        return self.proj(x)+y

class LegacyTCN(nn.Module):
    def __init__(self,epsilon: float) -> None:
        super().__init__();self.blocks=nn.ModuleList([LegacyBlock(258 if i==0 else 64,d,epsilon) for i,d in enumerate((1,2,4,8))]);self.head=nn.Linear(64,236)
    def per_frame(self,x):
        x=x.transpose(1,2)
        for block in self.blocks:x=block(x)
        return self.head(x.transpose(1,2)).softmax(-1)
    def forward(self,x,mask):
        p=self.per_frame(x);weight=mask.unsqueeze(-1).to(p.dtype)
        return (p*weight).sum(1)/weight.sum(1).clamp_min(1.)

def evaluate(model,data,device):
    model.eval();probs=[]
    with torch.inference_mode():
        for start in range(0,len(data['y']),32):
            x,m,y=batch(data,np.arange(start,min(start+32,len(data['y']))),device);probs.append(model(x,m).cpu().numpy())
    prob=np.concatenate(probs);return metrics(data['y'],prob.argmax(-1),data['participant_ids']),prob

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True,type=Path);p.add_argument('--resume',action='store_true');args=p.parse_args();run=run_root(args.run_root);out=run/'experiment_A/warmstart_repeat'
    seed=20261021;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('WARMSTART_CUDA_REQUIRED')
    contract=json.loads((out/'source_contract.json').read_text());eps=contract['configuration']['0_ln1']['epsilon'];model=LegacyTCN(eps)
    with np.load(out/'source_weights.npz',allow_pickle=False) as weights,torch.no_grad():
        for i,block in enumerate(model.blocks):
            for name in ('conv1','conv2','ln1','ln2'):
                layer=getattr(block,name);w=weights[f'{i}_{name}_0'];b=weights[f'{i}_{name}_1']
                layer.weight.copy_(torch.from_numpy(w.transpose(2,1,0).copy() if name.startswith('conv') else w));layer.bias.copy_(torch.from_numpy(b))
            if i==0:
                block.proj.weight.copy_(torch.from_numpy(weights['0_proj_0'].transpose(2,1,0).copy()));block.proj.bias.copy_(torch.from_numpy(weights['0_proj_1']))
        model.head.weight.copy_(torch.from_numpy(weights['head_0'].T.copy()));model.head.bias.copy_(torch.from_numpy(weights['head_1']))
    model.eval()
    with np.load(out/'transfer_fixture.npz',allow_pickle=False) as f,torch.inference_mode():
        actual=model.per_frame(torch.from_numpy(f['X'])).numpy();error=np.abs(actual-f['expected_probabilities']);mismatches=int(np.count_nonzero(actual.argmax(-1)!=f['expected_probabilities'].argmax(-1)))
    if error.max()>1e-4 or mismatches:raise RuntimeError('KERAS_TORCH_TRANSFER_PARITY_FAILED')
    if not args.resume:write(out/'transfer_parity.json',{'status':'PASS','max_abs_probability_error':float(error.max()),'mean_abs_probability_error':float(error.mean()),'argmax_mismatches':mismatches})
    model=model.cuda();train=load(run,'train');val=load(run,'val');opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001)
    config={'seed':seed,'max_epochs':5,'batch':4,'accumulation':8,'learning_rate':.0001,'loss':'NLL of mean real-frame probability, matching clip decision',
        'architecture':'exact legacy Keras port, parity-gated','source_model_sha256':contract['sha256'],'train_rows':len(train['y']),'val_rows':len(val['y']),
        'hypothesis':'aggregation-aligned clip objective may improve the frozen per-frame-supervised model','augmentation':'none','production_enabled':False,'wall_budget_seconds':600}
    checkpoints=out/'checkpoints';checkpoints.mkdir(exist_ok=True);best=(-1.,-1.);start_epoch=1
    if args.resume:
        pointer=json.loads((checkpoints/'latest.json').read_text());path=out/pointer['path']
        if sha(path)!=pointer['sha256']:raise ValueError('RESUME_HASH')
        c=torch.load(path,map_location='cuda',weights_only=False)
        if c['config']!=config:raise ValueError('RESUME_CONFIG')
        model.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);best=tuple(c['best']);start_epoch=c['epoch']+1
        random.setstate(c['python_rng']);np.random.set_state(c['numpy_rng']);torch.set_rng_state(c['torch_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in c['cuda_rng']])
        if start_epoch>5:print('ALREADY_COMPLETE');return
    else:
        write(out/'config.json',config);baseline,prob=evaluate(model,val,'cuda:0');write(out/'baseline_fp32.json',baseline);np.save(out/'baseline_fp32_probabilities.npy',prob)
    started=time.perf_counter()
    for epoch in range(start_epoch,6):
        model.train();tick=time.perf_counter();torch.cuda.reset_peak_memory_stats();order=np.random.permutation(len(train['y']));loss_sum=0.;seen=0;chunks=list(range(0,len(order),4));opt.zero_grad(set_to_none=True)
        for step,start in enumerate(chunks):
            idx=order[start:start+4];x,m,y=batch(train,idx,'cuda:0');prob=model(x,m);loss=F.nll_loss(prob.clamp_min(1e-8).log(),y)
            if not bool(torch.isfinite(loss)):raise RuntimeError('NONFINITE')
            group=min(8,len(chunks)-(step//8)*8);(loss/group).backward();loss_sum+=float(loss.detach())*len(idx);seen+=len(idx)
            if (step+1)%8==0 or step+1==len(chunks):
                norm=nn.utils.clip_grad_norm_(model.parameters(),5.)
                if not bool(torch.isfinite(norm)):raise RuntimeError('NONFINITE_GRAD')
                opt.step();opt.zero_grad(set_to_none=True)
            if time.perf_counter()-started>600:raise RuntimeError('TIME_BUDGET_RESUME_FROM_LAST_EPOCH')
        result,probs=evaluate(model,val,'cuda:0');score=(result['macro_f1'],result['top1']);improved=score>best
        if improved:best=score
        record={'epoch':epoch,'train_loss':loss_sum/seen,'val_top1':result['top1'],'val_macro_f1':result['macro_f1'],'epoch_seconds':time.perf_counter()-tick,
            'peak_vram_allocated':torch.cuda.max_memory_allocated(),**memory()}
        with (out/'training_log.csv').open('a',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(record))
            if epoch==1:w.writeheader()
            w.writerow(record)
        path=checkpoints/f'epoch_{epoch:03d}.pt';atomic_checkpoint(path,{'epoch':epoch,'config':config,'best':best,'model':model.state_dict(),'optimizer':opt.state_dict(),
            'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()})
        pointer={'path':str(path.relative_to(out)),'sha256':sha(path),'epoch':epoch};(checkpoints/'latest.json').write_text(json.dumps(pointer))
        if improved:(checkpoints/'best.json').write_text(json.dumps(pointer));(out/'validation_best.json').write_text(json.dumps(result));np.save(out/'validation_best_probabilities.npy',probs)
        print(json.dumps(record),flush=True)
    write(out/'training_status.json',{'status':'MAX_EPOCHS','completed_epochs':5,'used_cuda':True,'best_score':best,'seconds':time.perf_counter()-started,'memory':memory()})
if __name__=='__main__':main()

