"""Plot measured epoch CSV; can use an existing read-only matplotlib installation."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);args=p.parse_args()
run=args.run_root.resolve();run.relative_to(Path(__file__).resolve().parents[2]/'experiments/continuous_asl')
os.environ['MPLCONFIGDIR']=str(run/'cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
rows=list(csv.DictReader((run/'training_log.csv').open()))
epochs=[int(r['epoch']) for r in rows]
fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
axes[0].plot(epochs,[float(r['train_ctc_loss']) for r in rows],label='Train')
axes[0].plot(epochs,[float(r['val_ctc_loss']) for r in rows],label='Validation')
axes[0].set(xlabel='Completed epoch',ylabel='CTC loss',title='Loss');axes[0].legend(frameon=False)
axes[1].plot(epochs,[100*float(r['val_gloss_wer']) for r in rows],color='#b14b32',marker='o',markersize=3)
best=json.loads((run/'checkpoints/best.json').read_text())['epoch']
axes[1].axvline(best,color='#666666',linestyle='--',linewidth=1,label=f'Selected epoch {best}')
axes[1].set(xlabel='Completed epoch',ylabel='Gloss WER (%)',title='Validation sequence error');axes[1].legend(frameon=False)
for ax in axes:ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
fig.suptitle('NCSLGR pilot: 30 train / 7 validation clips\nCollection-disjoint diagnostic; signer identities unknown',fontsize=12)
for ext in ('png','svg'):
    target=run/f'training_curves.{ext}'
    if target.exists():raise FileExistsError('CURVE_EXISTS')
    fig.savefig(target,dpi=160)
with (run/'logs/plot_environment.json').open('x') as f:json.dump({'python':sys.executable,'matplotlib':matplotlib.__version__,'environment_modified':False},f,indent=2)
plt.close(fig)
