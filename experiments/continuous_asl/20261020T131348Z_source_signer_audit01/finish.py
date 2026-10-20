"""Finalize the no-fit experiment design and preserve all prior evidence."""
from audit import *
import re

def md(name,text):
 with (RUN/name).open('x',encoding='utf-8') as f:f.write(text.rstrip()+'\n')
def main():
 guarded=read(RUN/'guarded_decision.json');d=guarded['candidate'];inc=read(RUN/'identifiability.json');issues=read(RUN/'identity_issues.json');details=read(RUN/'guarded_fold_details.json')
 allrows=read(PARENT/'data/processed/train_dev/manifest.json');tr=[r for r in allrows if r['split']=='train'];lookup={r['sample_id']:r for r in tr}
 def family(r):
  match=re.match(r'^(\d{4}-\d{2}-\d{2})_',r['source_group']);return r['signer_id']+'::'+(match[1] if match else 'UNRESOLVED_ALL_UNDATED_MASTERS')
 fold=read(RUN/'folds'/f"{d['fold']}.json");fit=[lookup[i] for i in fold['train_sample_ids']];ev=[lookup[i] for i in fold['internal_eval_sample_ids']]
 # Also quantify why a direct small split of the latest497TRAIN task is weak.
 latest=[r for r in read(PREV/'frozen_manifest.json') if r['split']=='train'];held=set(fold['eval_capture_families']);sf=[r for r in latest if family(r) not in held];se=[r for r in latest if family(r) in held];ss=support(sf)
 sc=[r for r in se if set(r['glosses'])<=set(ss)];sc5=[r for r in se if all(ss[t]>=5 for t in r['glosses'])]
 save('small_candidate497_check.json',{'TRAIN':len(sf),'internal_eval':len(se),'compatible_eval':len(sc),'support5_eval':len(sc5),'compatible_by_signer':dict(Counter(r['signer_id'] for r in sc)),
  'status':'PARTIAL_TOO_SMALL_FOR_MULTI_SIGNER_TRANSFER_SCREEN','note':'Only24compatible records,23fromone signer; prefer the fullTRAIN guarded design for an exploratory pilot.'})
 # Label overlap is literal only; never treat this as semantic pool equivalence.
 catalog=read(PARENT/'eligible_corpus/manifest.json');nc=[r for r in catalog if r['training_pool'].startswith('NCSLGR')]
 pools={'ASLLRP3':set(support(tr)),'NCSLGR2':set(support(nc))};shared=pools['ASLLRP3']&pools['NCSLGR2']
 signerlabels=defaultdict(set);signercounts=Counter();signersources=defaultdict(set)
 for r in tr+nc:
  if r.get('signer_id'):signerlabels[r['signer_id']].update(r['glosses']);signercounts[r['signer_id']]+=1;signersources[r['signer_id']].add(r['source_group'])
 sl=[];keys=sorted(signerlabels)
 for i,a in enumerate(keys):
  for b in keys[i+1:]:sl.append({'signer_A':a,'signer_B':b,'shared_literal_labels':len(signerlabels[a]&signerlabels[b]),'A_unique_labels':len(signerlabels[a]),'B_unique_labels':len(signerlabels[b]),
   'contract':'SAME_ASLLRP3_LITERAL_CONTRACT' if a.startswith('asllrp') and b.startswith('asllrp') else 'NO_CROSS_POOL_TARGET_EQUIVALENCE_OR_PERSON_BRIDGE_PROVEN'})
 table('SIGNER_LABEL_OVERLAP.csv',sl)
 save('pool_overlap.json',{'ASLLRP3_TRAIN_labels':len(pools['ASLLRP3']),'NCSLGR2_TRAIN_labels':len(pools['NCSLGR2']),'shared_literal_spellings':len(shared),
  'compatible_targets':'UNPROVEN; shared spelling does not establish annotation-equivalent targets','explicit_signer_counts':dict(signercounts),'explicit_source_keys_per_signer':{k:len(v) for k,v in signersources.items()}})
 cf=list(csv.DictReader((RUN/'CANDIDATE_FOLDS.csv').open(encoding='utf-8-sig')))
 foldtable='\n'.join(f"|{r['fold']}|{r['TRAIN_utterances'] or '—'}|{r['internal_eval_utterances'] or '—'}|{r['compatible_eval_utterances'] or '—'}|{r['support5_eval_utterances'] or '—'}|{r['status']}|" for r in cf)
 part=list(csv.DictReader((RUN/'SAVED_MODEL_PARTITION_SUMMARY.csv').open(encoding='utf-8-sig')))
 parttable='\n'.join(f"|{r['model']}|{r['split']}|{r['utterances']}|{float(r['WER'])*100:.3f}%|{r['empty_outputs']}|{r['predicted_token_diversity']}|{float(r['blank_argmax_fraction']):.4f}|" for r in part)
 errors=list(csv.DictReader((RUN/'SAVED_MODEL_GROUP_ERRORS.csv').open(encoding='utf-8-sig')))
 signertable='\n'.join(f"|{r['model']}|{r['split']}|{r['group_id']}|{r['utterances']}|{float(r['WER'])*100:.3f}%|{r['reference_tokens_unseen_in_actual_fit']}|{r['top_prediction']} ({float(r['top_prediction_fraction'] or 0)*100:.1f}%)|" for r in errors if r['group_axis']=='signer_id' and r['model']=='prior497')
 md('SOURCE_SIGNER_AUDIT.md',f'''# TRAIN-only source/signer audit

**PASS audit execution and frozen preservation. PARTIAL recording-transfer design. BLOCKED independent dataset/signer attribution and crossed experiments. No training was run or authorized.**

## Scope and membership

The current3,121eligible catalog is preserved. This inventory has **2,708applicable TRAIN utterances**:1,706ASLLRP3from the frozen current split, plus1,002NCSLGR2. The latter have explicit TRAIN membership in the historical906TRAIN manifest or the96new timed-TRAIN manifest; catalog tags such as TRAIN_POOL_NOT_FIT were not treated as sufficient proof by themselves. The413current ASLLRP DEV records are excluded from identity/fold construction, even though they were historicalTRAIN in older experiments. Reserved17TEST was not used.

SOURCE_SIGNER_COVERAGE.csv contains every audited stable utterance ID, corpus-scoped source/signer ID, literal ordered gloss sequence, token/observation counts, feature path/hash/status, applicable frozen TRAIN membership and annotation/signer provenance. Feature/finite/timestamp/CTC evidence was reused from the prior3,121-file audit; existence and metadata-based CTC feasibility were checked here. No video decoding, appearance-based identity inference, label editing, acquisition or new predictions occurred.

There are no missing source_group strings, but that does not prove complete underlying recording identity. ASLLRP has1,706records with publisher parent-master names and participant joins; NCSLGR has547EAF/full-source joins and455index-media IDs with uncertain underlying recording granularity. NCSLGR has988missing signer records. Its14known records have two corpus-scoped signer IDs (13and1utterances); the numerous remaining unknowns cannot be treated as one person.

## Identity and source structure

|ASLLRP TRAIN signer ID|Utterances|Published recording IDs|
|---|---|---|
|asllrp_dai_participant_2|754|10|
|asllrp_dai_participant_3|617|19|
|asllrp_dai_participant_5|335|10|

Every one of the39ASLLRP recording IDs contains exactly one known signer; all three signers appear in multiple recording IDs. No source contains multiple known signers. Cramér's V(recording,signer)=**1.0**. The recording-indicator design has rank39; adding signer indicators leaves rank39: **zero additional rank**. This proves the fixed recording and signer effects are perfectly aliased in this incidence structure. It does not prove either factor caused the model failure.

Dataset and recording are different levels. The ASLLRP TRAIN data have one source dataset and3known signers; NCSLGR has2explicit IDs and988unknowns. No verified same-person cross-dataset bridge is present. Dataset/label-pool and corpus-scoped identity are confounded; unknown IDs cannot establish a bridge. NCSLGR main-gloss and ASLLRP dominant-realized-occurrence targets remain separate. They share{len(shared)}literal spellings in currentTRAIN, which is insufficient to establish semantic annotation compatibility. Cross-dataset source transfer is **BLOCKED**.

The required source-by-signer table is SOURCE_SIGNER_CROSS_TAB.csv, with corpus and label-pool scope. Additional DATASET_SIGNER_CROSS_TAB.csv, SIGNER_BY_LABEL_POOL.csv and SOURCE_BY_LABEL_POOL.csv distinguish levels. SIGNER_LABEL_OVERLAP.csv describes literal target overlap without claiming semantic or person equivalence.

No multiple-source exact raw-hash aliases or duplicate decoded-pixel utterance hashes were found in this TRAIN inventory. Reuse of a full source hash by distinct utterances within one recording is expected. No identifier collisions across pool namespaces, multi-URL source keys or multi-collection source keys appeared. These checks cannot detect all re-encodings or unknown alternate views. Only current manifest view artifacts are audited; additional published camera views do not become training records automatically. Unresolved NCSLGR parent/view links block its crossed folds.

## Conservative recording-family guard

Distinct master filenames are not proof of independent recording ancestry. The Ben files form a numbered introduction-to-conclusion series; independence from a shared original is not established. Keep all335undated Ben records together. For date-coded masters, keep all masters with the same known signer/date-shaped prefix together. These are **conservative grouping proxies**, not inferred identities, authenticated capture dates or claims that every grouped file is physically one recording.

The guard gives8families:2for participant2,5for participant3,1unresolved Ben family. Preliminary filename-based A folds split these families and are explicitly marked PARTIAL_FILENAME_ONLY_GUARD_FAILED; their assignments are retained as completed audit evidence, not recommended for fitting. Entire masters, collections and conservative families remain together in the guarded folds. No adjacent utterance or known repeated view/content group crosses an assignment. Unknown cross-date re-encodings remain a limitation.

## Candidate folds

|Candidate|Internal TRAIN|Internal eval|All-label-compatible eval|Support≥5eval|Status|
|---|---|---|---|---|---|
{foldtable}

Every actual fold has a separate folds/*.json membership manifest with a parent-manifest hash. All members come from currentASLLRP TRAIN. Full internal evaluation is retained; compatible/support5views include entire reference sequences only. EVAL_VIEW_EXCLUSIONS.csv records every auxiliary-view exclusion and its literal labels; no reference tokens were removed from full evaluation. CANDIDATE_FOLDS.csv reports signers/sources, vocabulary, token coverage, CTC slack, exact-content leakage checks and manifest hashes. LABEL_COVERAGE_BY_FOLD.csv reports distinct TRAIN support bins and token frequencies for both sides.

The guarded pooled fold has1,500TRAIN from34published masters/6families and206internal-eval utterances from5masters/2families. All evaluation signers occur inTRAIN; TRAIN includes three signers, eval two. TRAIN has2,287labels; eval607, with377shared. Full eval has1,794reference tokens; token coverage81.05%. Its80compatible utterances cover only27.03% of full-eval tokens and are imbalanced:74participant3,6participant2. The stronger-support view has27utterances. These restricted views control unknown/low-support labels partially; they do not match topic, sequence difficulty or frequency distributions. There are only two eval family clusters, so broad confidence claims are not supported.

A direct guarded split of the latest497TRAIN task would leave446TRAIN/51eval, but only24compatible records (23participant3,1participant2) and2support5records. That is too small for a multi-signer transfer screen. The larger currentTRAIN design is the better exploratory candidate; it still needs a TRAIN-learning gate before transfer WER is interpreted.

Leave-one-signer-out B folds also hold out all that person's recordings; they test **joint signer+recording transfer**, not an independent signer effect. A crossed C fold is impossible with these observed cells and was not fabricated. Ben has no safe within-person family holdout under the conservative guard. NCSLGR missing identities/recording lineage/target compatibility prevent an alternative crossed design.

## Existing model evidence, without new evaluation

Partition statistics reused from saved predictions and logs:

|Model|Partition|Utterances|WER|Empty outputs|Predicted labels|Blank argmax fraction|
|---|---|---|---|---|---|---|
{parttable}

The latest497model's observed per-signer summaries:

|Model|Partition|Signer ID|Utterances|WER|Unseen reference tokens in actual fit|Dominant emission share|
|---|---|---|---|---|---|---|
{signertable}

SAVED_MODEL_GROUP_ERRORS.csv includes per-recording and per-signer errors, output lengths/empties, label diversity, dominant-emission share and effective-fit support. It reports previously computed TRAIN resubstitution and frozenDEV separately. Per-group blank fractions are UNKNOWN because per-utterance frame logits were not saved; empty-output rates are not substituted for blank probability. Recoverable global1706TRAIN blank statistics were reused from its completed replay.

TRAIN scores show underfitting/memorization behavior, not source transfer. FrozenDEV uses one held-out signer and different recordings, so its high WER does not distinguish the factors. The compatible92DEV failed with zero unsupported labels in the last task, which excludes missing vocabulary as a sufficient explanation. Sparse support, source/recording/topic effects, signer variation and visual feature/model limitations remain competing explanations. Metadata correlations and current WER do not establish causality.

All old recognition models trained on some or all of the proposed internal holdout records. **Do not replay or warm-start those models and call it unseen-source performance.** A later source pilot needs a fresh recognition model/optimizer or separately proven nonleaking initialization, TRAIN-only normalization/vocabulary and the new frozen membership.

## Decision

Recommend one **source-recording-family transfer pilot**, exploratory and conditional on later explicit training authorization; see NEXT_EXPERIMENT_DECISION.md. It can ask whether failure persists with known signer identities and unseen families. Independent source-dataset/signer causal effects and a crossed comparison remain BLOCKED. No fit, checkpoint selection or promotion occurred here.
''')
 gd=details[d['fold']];held=guarded['held_family_by_signer'];heldtext='\n'.join(f'- `{s}`: `{key}`.' for s,key in sorted(held.items()))
 md('NEXT_EXPERIMENT_DECISION.md',f'''# Next experiment decision — no training authorized

**Recommendation: SOURCE-TRANSFER PILOT, restricted to conservative recording families within ASLLRP3.** Audit execution PASS; experimental identifiability PARTIAL; independent dataset-vs-signer attribution and crossed design BLOCKED. This document is a design for later approval, not an instruction to launch a fit now.

Use `{RUN/'folds'/f"{d['fold']}.json"}`

Manifest SHA256 `{d['manifest_sha256']}`.

Internal TRAIN1,500, eval206. TRAIN has3known signers and34published masters across6conservative families. Eval has2known signers already inTRAIN and5masters across2families. Keep every member of each master/collection/family on one side. No frozen externalDEV or TEST participation in assignment, fitting or selection.

Held-out family proxies, selected by metadata SHA256 ordering before group WER inspection:

{heldtext}

All undated Ben files remain together inTRAIN. Their independent parent-recording ancestry is unresolved. Date-shaped prefixes are conservative provenance groups, not verified capture timestamps. No facial identity or master-name person inference was used.

**Question:** can an otherwise identical recognizer learn complete sequences from other families of known signers and transfer to these unseen families? Failure here, after a meaningful TRAIN-learning gate, would show that an unseen signer is not necessary for the failure. Success would motivate a separate controlled joint signer/source study; it would not prove the old DEV gap was caused by signer variation.

**Label controls:** preserve literal ASLLRP3 targets; fit vocabulary and normalization only on1,500internalTRAIN. Report full206eval WER with its81.05%token coverage, all80whole-sequence compatible-eval WER (zero reference OOV), and27support≥5whole-sequence WER. Report reference-error support strata and insertions separately. All IDs/views are frozen in the manifest. Compatible eval is74participant3/6participant2, so participant2 results remain descriptive; no group removal to improve a score. Do not compare these restricted views directly to the old full-task WER or85.96%nine-NCSLGRbaseline.

**Smallest controlled run if later authorized:** one fresh48channel causalCTC model with unchanged378features/decoder/loss/optimizer family; bounded smoke-first execution, not a seed/LR sweep. No older ASL recognition weights may be loaded because they saw these internal-eval records. Save seed/config/TRAIN-only vocab+normalization/optimizer/RNG/checkpoints and hashes. Use the internal compatible view for local diagnostic selection only; leave full-taskDEV/TEST frozen. Limit the first run to a declared short epoch/wall/VRAM budget before execution.

**Predeclared screen for a future run:** require TRAIN WER≤50%, at least10distinct predicted labels and multi-token output on at least25% of a fixedTRAIN monitor before interpreting transfer. Then require compatible-eval WER≤90%, at least10distinct nonblank predicted labels, and multi-token hypotheses on at least25% of80compatible examples. Also report full206WER, support5view and both family-level errors. Stop on nonfinite/resource failure or persistent collapse. These are engineering screening criteria, not statistical acceptance or production accuracy targets; two family clusters cannot support a population confidence claim. A failed TRAIN gate makes the source-transfer inference inconclusive rather than proving a source effect.

**Remaining confounders:** source family includes date/session/background/topic/utterance/annotation and label-frequency shifts; source and signer fixed effects are rank-aliased; unknown parent lineage and re-encodings cannot be fully excluded by hashes. Source-dataset transfer needs verified same-person bridges and compatible target contracts across datasets, neither present. An independently crossed factorial experiment would require shared, verified recording-condition/session definitions across multiple signers, with matched supervision and permissions; never manufacture those relationships.

The smaller497-task family split has only24compatible examples and two with≥5TRAIN support, so it is not the recommended multi-signer screen. Within-one-person guarded alternatives are included but have one held-out family each and limited shared-label coverage. No new data acquisition is required for this exploratory design; resolving lineage or planning a genuine crossed capture is the smallest next **metadata/data-design** step if independent causal attribution is required.

**No model training was executed in this task. Wait for a later explicit training authorization.** The immediate runnable command is the read-only verifier listed in commands_and_results.txt.
''')
 py=ROOT/'experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe'
 md('commands_and_results.txt',f'''Initial existing read-only verification (run first):
& '{py}' -B '{PREV/'verify.py'}'
PASS:65label-support artifacts;46prior diagnostic artifacts;1061continuation artifacts;20745protectedmetadata;3121eligible;noTESTscoring.

TRAIN identity/fold audit:
& '{py}' -B '{RUN/'audit.py'}'
PASS1.311seconds for bounded metadata/prediction analysis;2708applicableTRAIN IDs. No model execution, fit, media decoding or downloads.

Conservative parent-family guard:
& '{py}' -B '{RUN/'family_guard.py'}'
PASS. Preliminary filename-only A folds retained but not fit-ready. New guarded folds keep related/undated masters together. No signer IDs inferred.

Finalize reports:
& '{py}' -B '{RUN/'finish.py'}'
Requiredoutputs:SOURCE_SIGNER_AUDIT.md;SOURCE_SIGNER_COVERAGE.csv;SOURCE_SIGNER_CROSS_TAB.csv;CANDIDATE_FOLDS.csv;LABEL_COVERAGE_BY_FOLD.csv;NEXT_EXPERIMENT_DECISION.md;commands_and_results.txt;artifact_hashes.sha256.

Exact next command — READ ONLY:
& '{py}' -B '{RUN/'verify.py'}'
Verifies new hashes, guarded TRAIN-only folds and frozen prior chain; no model inference, fitting, acquisition or TEST scoring.

Future training: NOT AUTHORIZED. No training command has been scheduled or run. No APK, Android integration, production selection, thresholds, avatar/GLB, frozen manifest/checkpoint/result or reservedTEST change.
''')
 md('SESSION_HANDOFF.md','No-fit TRAIN-only source/signer audit complete. Read SOURCE_SIGNER_AUDIT.md and NEXT_EXPERIMENT_DECISION.md. Guarded1500TRAIN/206internalEval family design is exploratory; independent dataset/signer causal attribution remains blocked. Original filename-only A folds are not fit-ready after lineage guard. No queued training/download. Wait for later explicit fit authorization. Existing root continuity/progress files and all frozen assets preserved. Review: scope PASS, protection PASS, transfer design PARTIAL, crossed experiment BLOCKED.')
 save('finish_input_hashes.json',INPUTS)
 paths=sorted(p for p in RUN.rglob('*') if p.is_file() and p.name!='artifact_hashes.sha256' and '__pycache__' not in p.parts)
 md('artifact_hashes.sha256','\n'.join(f'{sha(p)}  {p.relative_to(RUN).as_posix()}' for p in paths))
 print(json.dumps({'status':'REPORTS_COMPLETE_NO_TRAINING','new_artifacts':len(paths),'TRAIN_inventory':2708,'recommendation':guarded['recommendation'],'identifiability':'PARTIAL; independent crossed effects BLOCKED'}))

if __name__=='__main__':main()
