"""Export known local Keras weights and same-input parity fixtures, never retrain it."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from common import run_root,sha,write
p=argparse.ArgumentParser();p.add_argument('--run-root',required=True,type=Path);args=p.parse_args();run=run_root(args.run_root)
os.environ['KERAS_HOME']=str(run/'cache/keras');os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(1)
source=Path('D:/derived/derived/apphear3/runs/tcn_streaming_conversational_fps_v30/best.keras')
model=tf.keras.models.load_model(source,compile=False,safe_mode=True)
weights={};configuration={}
for i in range(4):
    for part in ('conv1','conv2','ln1','ln2'):
        layer=model.get_layer(f'tcn{i}_{part}');configuration[f'{i}_{part}']=layer.get_config()
        for j,array in enumerate(layer.get_weights()):weights[f'{i}_{part}_{j}']=array
    if i==0:
        for j,array in enumerate(model.get_layer('tcn0_proj').get_weights()):weights[f'0_proj_{j}']=array
for j,array in enumerate(model.get_layer('logits_perframe').get_weights()):weights[f'head_{j}']=array
x=np.load(run/'experiment_A/data/val/X.npy',mmap_mode='r')[:8].copy();expected=model(x,training=False).numpy()
out=run/'experiment_A/warmstart';out.mkdir();np.savez_compressed(out/'source_weights.npz',**weights)
np.savez_compressed(out/'transfer_fixture.npz',X=x,expected_probabilities=expected)
write(out/'source_contract.json',{'source':str(source),'sha256':sha(source),'configuration':configuration,'classes':236,'schema':'3.0',
    'vocabulary_sha256':sha(run/'experiment_A/vocabulary.json'),'no_source_modification':True})
print('KERAS_WEIGHTS_AND_FIXTURE_SAVED')
