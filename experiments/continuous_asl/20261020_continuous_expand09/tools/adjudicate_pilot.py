"""Measured continuation decision, using TRAIN signals; DEV selection unchanged."""
import csv,json
from core import RUN,read,write,sha

def main():
 if (RUN/'pilot_outcome.json').exists():print('PILOT_ALREADY_ADJUDICATED');return
 base=RUN/'models/reference48';status=read(base/'training_status.json');initial=read(base/'initial_train_monitor.json')
 final=read(base/'last_epoch_train.json');selected=read(base/'selected_checkpoint_train.json');logs=list(csv.DictReader((base/'training_log.csv').open(newline='')))
 final_monitor=float(logs[-1]['train_monitor_ctc_loss']);ratio=final_monitor/initial['loss']
 signal=final['gloss_wer']<=.8 and ratio<=.5 and status['checkpoint_reload_predictions_match']
 pointer=read(base/'checkpoints/best.json')
 write(RUN/'model_selection_freeze.json',{'manifest_sha256':status['manifest_sha256'],'trainer_sha256':status['script_sha256'],
       'models':{'reference48':{'pointer':pointer,'status_sha256':sha(base/'training_status.json'),'config_sha256':sha(base/'config.json')}},
       'selection':'Exact original9DEV WER then loss; no test reads'})
 result={'status':'PASS_LEARNING_SIGNAL' if signal else 'BLOCKED_POOR_TRAIN_LEARNING_INVESTIGATE_BEFORE_MAIN',
         'train':status['unique_train_utterances_consumed'],'dev':9,'epochs':status['completed_epochs'],
         'selected_epoch':pointer['epoch'],'final_epoch_train_wer':final['gloss_wer'],'selected_checkpoint_train_wer':selected['gloss_wer'],
         'initial_monitor_wer':initial['gloss_wer'],'final_monitor_wer':float(logs[-1]['train_monitor_wer']),
         'monitor_ctc_loss_ratio':ratio,'dev_best':read(base/'dev_best.json'),
         'fit_reload_seconds':status['seconds'],'peak_allocated_vram_bytes':status['peak_allocated_vram_bytes'],
         'checkpoint_path':str(base/pointer['path']),'checkpoint_sha256':pointer['sha256'],'manifest_sha256':status['manifest_sha256'],
         'test_accessed':False,'production_promoted':False,
         'meaning':'Minimal TRAIN optimization signal only; DEV WER remains the unchanged selection metric. Not generalization or a guaranteed fix.'}
 write(RUN/'pilot_outcome.json',result);print(json.dumps({k:v for k,v in result.items() if k!='dev_best'}),flush=True)

if __name__=='__main__':main()
