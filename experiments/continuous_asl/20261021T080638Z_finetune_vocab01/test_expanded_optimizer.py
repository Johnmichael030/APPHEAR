"""Synthetic engineering smoke only. No participant data or learned new sign."""
import torch
import fine_tune as ft
from vocabulary_extension import extend, expanded_optimizer_state

torch.set_num_threads(2)
v = ft.read(ft.RUN/'vocabulary.json')['tokens']
s = torch.load(ft.SOURCE, map_location='cpu', weights_only=False)
m = ft.CausalCTC(len(v),48,s['model']['mean'].numpy(),s['model']['std'].numpy())
m.load_state_dict(s['model'])
n,nv = extend(m,v,[{'candidate_gloss':'SYNTHETIC_OPTIMIZER_TEST'}],synthetic_test=True)
o = torch.optim.AdamW(n.parameters(),lr=.0001)
o.load_state_dict(expanded_optimizer_state(s['optimizer'],m,n))
before = n.output.weight[-1].detach().clone()
x = torch.randn(1,8,378,generator=torch.Generator().manual_seed(21))
loss = torch.nn.CTCLoss(blank=0)(n(x).log_softmax(-1).transpose(0,1),torch.tensor([len(nv)-1]),torch.tensor([8]),torch.tensor([1]))
loss.backward(); o.step()
assert torch.isfinite(loss) and not torch.equal(before,n.output.weight[-1])
ft.save(ft.RUN/'B/SYNTHETIC_OPTIMIZER_SMOKE.json',dict(status='PASS_SYNTHETIC_ONLY',loss=float(loss.detach()),expanded_AdamW_step=True,actual_new_classes=0,participant_data_used=False,checkpoint_saved=False))
ft.writecsv(ft.RUN/'MAPPING_DIFF.csv',[dict(old_id=i,new_id=i,old_label=t,new_label=t,status='UNCHANGED_REAL_MODEL') for i,t in enumerate(v)])
print('PASS: expanded AdamW step; synthetic only; no checkpoint saved; all2289real IDs unchanged.')
