"""Save source decisions and retained failures; no downloads or fitting."""
from core import RUN, read, write, sha

def main():
    if (RUN/'source_status.json').exists():return
    write(RUN/'source_status.json',{'acquisition_limits':read(RUN/'acquisition_limits.v2.json'),
      'sources':[
        {'source':'Official DePaul NCSLGR ELAN conversion','url':'https://asl.cs.depaul.edu/corpus/index.html',
         'archive':'https://asl.cs.depaul.edu/corpus/elanBUcorpus.zip','status':'DOWNLOADED_CRC_PASS_AUTHENTIC_MAIN_GLOSS',
         'excluded':['ali sync mismatch','roadtrip2 timing/missing-half annotations','DSP separate rights','ambiguous multipart URLs','overlapping utterance spans'],
         'official_video_archive':'https://asl.cs.depaul.edu/corpus/video.zip','archive_head_bytes':4108493785,
         'archive_status':'NOT_DOWNLOADED_TO_AVOID_DUPLICATE_VIEWS_AND_MEDIA; exact BU index-linked frontal files acquired separately'},
        {'source':'ASLLRP2025 SignStream3 continuous utterances','url':'https://www.bu.edu/asllrp/ASL-SignBank-and-other-Resources.html',
         'terms':'https://www.bu.edu/asllrp/dai-terms.html','status':'ANNOTATIONS_VERIFIED_MEDIA_DOWNLOAD_IN_PROGRESS',
         'annotation_manifest':'asllrp2025_catalog/plan.json','permission_scope':'Research/education only; no redistribution or commercial deployment.',
         'target_scope':'Literal dominant-realized occurrence events, including hidden events. Solely nondominant events absent from publisher CSV. No invented two-hand supervision.',
         'label_version_caveat':'Publisher documents convention differences between SignStream2 and3. Preserve each source literal event string; no canonicalization to improve DEV.'},
        {'source':'How2Sign','url':'https://how2sign.github.io/#download',
         'status':'EXISTING_VALIDATION_ARCHIVE_REUSED; AUTHENTIC_GLOSS_DOWNLOAD_NOT_PUBLISHED_IN_INSPECTED_TABLE',
         'official_validation_frontal_rgb_id':'1DhLH8tIBn9HsTzUJUfsEOGcP4l9EvOiO',
         'official_training_frontal_rgb_id':'1VX7n0jjW0pW3GEdgOks3z8nqE6iI6EnW',
         'labels':'English translations original and realigned, not ASL gloss. No31GB unlabeled training download.',
         'preserved_prior_failed_url':'https://drive.google.com/uc?export=download&id=1oVZyTWhHShyqshC2kUrfWnBF8apIR7Z1',
         'preserved_prior_error':'HTTP_404','prior_failure_utc':'2026-10-19T16:53:14.5183865Z',
         'archive_reverification':'Not repeated; previously verified1778074855 bytes. Authoritative prior receipt and hash retained untouched.'},
        {'source':'Cokely EKU','url':'https://encompass.eku.edu/cokely_videos/','status':'ANNOTATIONS_HTTP403_NO_BYPASS',
         'evidence':'expand04/logs/cokely_Annotations_20261020T104719_617.jsonl','next':'Publisher restores public EAF access or separately provides authorized copies.'},
        {'source':'Gallaudet MoLo','url':'https://sites.google.com/gallaudet.edu/card/data/molo/molo-primary-data',
         'annotation_url':'https://osf.io/9uevh','error':'HTTP403','status':'NO_GLOSS_PAYLOAD_AVAILABLE; NO_UNLABELED_VIDEO_DOWNLOAD'},
        {'source':'Gallaudet O5S5','url':'https://sites.google.com/gallaudet.edu/o5s5asl/home',
         'annotation_url':'https://osf.io/37w6t','error':'HTTP403','status':'NO_ACCESS_BYPASS_OR_UNLABELED_DOWNLOAD'},
        {'source':'Purdue RVL-SLLL','url':'https://engineering.purdue.edu/RVL/Database/ASL/asl-database-front.htm',
         'status':'SIGNED_LICENSE_AND_CREDENTIALS_REQUIRED_NO_SIGNATURE_OR_ACCOUNT_ACTION'},
        {'source':'CLERC Epee','url':'https://huggingface.co/datasets/CLERC-DATA/epee',
         'status':'PUBLIC128x3_KEYPOINTS_ARE_INCOMPATIBLE_WITH_CANDIDATE01_RAW_BODY_HAND_VALIDITY_CONTRACT; RGB_LICENSE_REQUIRED',
         'next':'Separate explicit pose-only model/feature contract if authorized; never pad into378 schema.'},
        {'source':'RIT ASL-Homework-RGBD','url':'https://latlab.ist.rit.edu/lrec2022/',
         'status':'Full corpus Databrary authorized access not established; public sample not counted as bulk corpus.'}],
      'failures_retained':[
        {'command':'& experiments/continuous_asl/20261020_continuous_expand06/Acquire-Media.ps1',
         'error':'Add-Content: logs/depaul_media_20261020T112257_422.jsonl parent directory missing',
         'recovery':'Create experiment-local logs directory; resumed same media receipt/hash, no duplicate download.'},
        {'command':'python -B experiments/continuous_asl/20261020_continuous_expand06/tools/prepare_timed.py --watch',
         'error':"NameError: name 'Path' is not defined",'log':'logs/timed_preparation_01.log',
         'recovery':'Added pathlib.Path import before any media processing; retained failed log; second invocation continues.'}],
      'diagnosis_sha256':sha(RUN/'wer_diagnosis.json'),'production_changed':False})

if __name__=='__main__':main()
