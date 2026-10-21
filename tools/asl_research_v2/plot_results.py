"""Plot measured validation results only, using an existing matplotlib installation."""
import argparse
import csv
import json
import os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run-root',required=True,type=Path);args=p.parse_args();run=args.run_root.resolve()
os.environ['MPLCONFIGDIR']=str(run/'cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
baseline=json.loads((run/'experiment_A/warmstart/baseline_fp32.json').read_text())['top1']*100
for folder,label in (('warmstart','Fine-tune seed 20261020'),('warmstart_repeat','Fine-tune seed 20261021')):
    rows=list(csv.DictReader((run/'experiment_A'/folder/'training_log.csv').open()))
    axes[0].plot([int(r['epoch']) for r in rows],[100*float(r['val_top1']) for r in rows],marker='o',markersize=4,label=label)
axes[0].axhline(baseline,color='#777',linestyle='--',label='Frozen FP32 baseline')
axes[0].set(xlabel='Completed epoch',ylabel='Top-1 (%) — zoomed scale',title='Isolated-sign validation: 5 held-out signer IDs')
axes[0].legend(fontsize=8,frameon=False,loc='lower right')
rows=list(csv.DictReader((run/'experiment_B/training_log.csv').open()))
axes[1].plot([int(r['epoch']) for r in rows],[100*float(r['val_gloss_wer']) for r in rows],color='#b24c36',marker='o',markersize=3,label='Native-cadence candidate')
axes[1].axhline(88,color='#777',linestyle='--',label='Prior 10 Hz baseline')
axes[1].set(xlabel='Completed epoch',ylabel='Gloss WER (%) — lower is better',title='Continuous diagnostic: 7 clips, signer IDs unknown')
axes[1].legend(fontsize=8,frameon=False)
for ax in axes:ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Separate tasks and metrics; validation evidence only — no production integration',fontsize=11)
fig.savefig(run/'validation_results.png',dpi=150);fig.savefig(run/'validation_results.svg');plt.close(fig)
