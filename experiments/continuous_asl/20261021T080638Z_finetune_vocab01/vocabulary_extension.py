"""Append-only experimental CTC head extension. No data admission by filename or English alias."""
import copy,io,math
import numpy as np
import torch
import fine_tune as ft

def extend(model,old_vocabulary,approved,seed=20261021,synthetic_test=False):
 assert old_vocabulary[:2]==['<BLANK>','<UNK>']
 if not approved:raise ValueError('NO_APPROVED_NEW_CLASSES')
 labels=[r['candidate_gloss'] for r in approved]
 assert len(labels)==len(set(labels)) and not set(labels)&set(old_vocabulary),'DUPLICATE_OR_EXISTING_LABEL'
 assert all(isinstance(t,str) and t and not t.startswith('<') for t in labels),'INVALID_LITERAL_LABEL'
 if not synthetic_test:
  for r in approved:
   if r['training_example_count']<3 or r['validation_example_count']<1 or r['permission_status']!='VERIFIED_RESEARCH_USE' or r['annotation_status']!='VERIFIED_LITERAL_TARGET_CONTRACT' or r['split_status']!='PASS_FAMILY_GUARD':
    raise ValueError('NEW_CLASS_ADMISSION_BLOCKED:'+r['candidate_gloss'])
 vocabulary=list(old_vocabulary)+sorted(labels);old=model.state_dict();n=len(old_vocabulary)
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(seed)
  new=ft.CausalCTC(len(vocabulary),old['input.weight'].shape[0],old['mean'].cpu().numpy(),old['std'].cpu().numpy())
  state=new.state_dict()
  for key,value in old.items():
   if key.startswith('output.'):state[key][:n].copy_(value.cpu())
   else:state[key].copy_(value.cpu())
  # Kaiming-uniform rows, deterministic local RNG; old logits/weights remain intact.
  generator=torch.Generator().manual_seed(seed)
  state['output.weight'][n:].uniform_(-1/math.sqrt(state['output.weight'].shape[1]),1/math.sqrt(state['output.weight'].shape[1]),generator=generator)
  state['output.bias'][n:].fill_(float(old['output.bias'][2:].median()))
  new.load_state_dict(state,strict=True)
 assert vocabulary[:n]==old_vocabulary
 for key,value in old.items():assert torch.equal(value.cpu(),new.state_dict()[key][:n] if key.startswith('output.') else new.state_dict()[key])
 return new,vocabulary

def expanded_optimizer_state(old_state,old_model,new_model):
 result=copy.deepcopy(old_state);old_params=list(old_model.parameters());new_params=list(new_model.parameters())
 ids=[i for group in result['param_groups'] for i in group['params']]
 assert len(ids)==len(old_params)==len(new_params)
 for index,before,after in zip(ids,old_params,new_params):
  for name,value in list(result['state'].get(index,{}).items()):
   if torch.is_tensor(value) and value.ndim and tuple(value.shape)==tuple(before.shape) and before.shape!=after.shape:
    assert after.shape[0]>before.shape[0] and after.shape[1:]==before.shape[1:]
    padded=torch.zeros(after.shape,dtype=value.dtype,device=value.device);padded[:before.shape[0]]=value
    result['state'][index][name]=padded
 return result

def tests():
 output=ft.RUN/'B/EXPANSION_IMPLEMENTATION_TESTS.json'
 if output.exists():print('EXPANSION_TESTS_ALREADY_COMPLETE');return
 vocab=ft.read(ft.RUN/'vocabulary.json')['tokens'];state=torch.load(ft.SOURCE,map_location='cpu',weights_only=False)
 old=ft.CausalCTC(len(vocab),48,state['model']['mean'].numpy(),state['model']['std'].numpy());old.load_state_dict(state['model']);old.eval()
 records=[dict(candidate_gloss=f'SYNTHETIC_UNIT_LABEL_{i}',training_example_count=0) for i in range(3)]
 new,newv=extend(old,vocab,records,synthetic_test=True);new.eval()
 x=torch.randn(1,17,378,generator=torch.Generator().manual_seed(17))
 with torch.no_grad():a=old(x);b=new(x)
 difference=float((a-b[:,:,:len(vocab)]).abs().max());assert torch.allclose(a,b[:,:,:len(vocab)],atol=3e-5,rtol=3e-5)
 buffer=io.BytesIO();torch.save(dict(model=new.state_dict(),vocabulary=newv),buffer);buffer.seek(0);roundtrip=torch.load(buffer,map_location='cpu',weights_only=False)
 assert roundtrip['vocabulary']==newv and torch.equal(roundtrip['model']['output.weight'],new.state_dict()['output.weight'])
 opt=torch.optim.AdamW(new.parameters(),lr=.0001);opt.load_state_dict(expanded_optimizer_state(state['optimizer'],old,new))
 target=torch.tensor([len(newv)-1]);loss=torch.nn.CTCLoss(blank=0,zero_infinity=False)(new(x).log_softmax(-1).transpose(0,1),target,torch.tensor([17]),torch.tensor([1]));loss.backward()
 assert torch.isfinite(loss) and new.output.weight.grad[-1].abs().sum()>0
 blocked=False
 try:extend(old,vocab,records)
 except (ValueError,KeyError):blocked=True
 assert blocked
 ft.save(output,dict(status='PASS_SYNTHETIC_ENGINEERING_ONLY',old_class_IDs_unchanged=len(vocab),old_parameter_tensors_exact=True,synthetic_added_classes=3,actual_new_classes_added=0,max_old_logit_error=difference,softmax_note='Old logits are preserved at initialization; adding logits necessarily changes the softmax denominator.',checkpoint_serialization='in-memory roundtrip only; no synthetic checkpoint distributed',expanded_optimizer_moments=True,new_row_CTC_gradient_finite=True,invalid_admission_rejected=True,source_checkpoint_sha256=ft.sha(ft.SOURCE),trained_new_signs=False))
 print('PASS: old2289IDs/weights, expanded head/moments, finite new-rowCTCgradient, serialization and admission rejection. Synthetic only.')
if __name__=='__main__':tests()
