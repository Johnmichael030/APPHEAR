"""Create the diagnosis from measured outputs; exclusive new-directory writes."""
from datetime import datetime, timezone, timedelta
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def read(name):return json.loads((HERE/name).read_text(encoding='utf-8-sig'))
def save(name,text):
    with (HERE/name).open('x',encoding='utf-8') as stream:stream.write(text)
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for c in iter(lambda:stream.read(1048576),b''):h.update(c)
    return h.hexdigest()

def main():
    a=read('audit_results.json');checks=read('checks.json');s=read('supplement_results.json');pool=read('pool_details.json');last=read('latest_epoch_dev_replay.json');origin=read('source_contract_checks.json')
    with (HERE/'prediction_audit.csv').open(encoding='utf-8-sig',newline='') as stream:pred=list(csv.DictReader(stream))
    mainrows=[r for r in pred if r['run']=='main2119']
    before=read('preservation_before.json')['files'];changed=[]
    for path,meta in before.items():
        stat=Path(path).stat()
        if [stat.st_size,stat.st_mtime_ns]!=meta:changed.append(path)
    if changed:raise ValueError('FROZEN_METADATA_CHANGED:'+str(changed))
    if not all(c['passed'] for c in checks) or s['status']!='PASS' or origin['status']!='PASS':raise ValueError('DIAGNOSTIC_CHECKS_REQUIRE_REVIEW')
    date=datetime.now(timezone(timedelta(hours=8))).isoformat()
    pilot=ROOT/'experiments/continuous_asl/20261020T102748Z_continuation_pilot01'
    pilot_status=json.loads((pilot/'training_status.json').read_text(encoding='utf-8'))
    results=[
      ('PASS','Completed package preserved','High',f'Initial Validate-Completed.ps1 exited0/PASS_READ_ONLY_VERIFICATION. Final metadata checks cover {len(before)} protected/indexed paths; no changes. Canonical3025eligible, original manifests and six overlap plus one frame-count exclusions remain unchanged.'),
      ('PASS','Selected model predicts blank at every DEV timestep','High','epoch2:430/430argmax blank0,9/9hypotheses empty,0S/57D/0I; exact saved-prediction replay. This directly explains100%WER.'),
      ('PASS','Epoch2 selection matches declared rule','High','All17epochs have DEV WER1.0; tie-break loss minimum occurs at epoch2. Patience15 then stops at17. No checkpoint-selection implementation bug found.'),
      ('PASS','71.95%TRAIN WER belongs to epoch17','High','Full2119TRAIN replay:3529S+8967D+62I over17453tokens =71.95324586%. Epoch2 TRAIN:1S+17451D+0I =99.99427033%. Both match saved full predictions exactly.'),
      ('PASS','CTC/token/decoder/metric implementation checks','High','blank0/UNK1, TRAIN-sorted unique vocab, checkpoint/config/manifest identity, all3025CTC length checks, no neural downsampling/truncation, independent greedy collapse/edit distance, synthetic exhaustive prefix sums and numeric CTCLoss checks pass.'),
      ('PASS','Normalization and feature contract','High','Each model TRAIN mean/std recomputes bit-exactly; same378-feature schema, finite values and increasing timestamps. Three saved raw fixtures reproduce features and causal prefixes exactly. Padding/prefix forward tests pass; no video reprocessing.'),
      ('PARTIAL','Cross-run comparison equivalence','High','Same9DEV IDs, literal refs, feature hashes, greedy decoder, metric and48-channel architecture; different TRAIN size, corpus, vocab, normalization, budgets and fitted weights. WER contrast is valid on this fixed diagnostic set, not a controlled architecture/data-scaling or signer-independent benchmark.'),
      ('PARTIAL','Known train/DEV leakage checks','High for checked keys','No sample ID, collection, source_group, feature SHA or exact raw SHA intersection for each run. Shared prompt text is audited separately. Unknown signer IDs and no exhaustive perceptual duplicate detector prevent a complete independence claim.'),
      ('BLOCKED','Signer-independent generalization','High','All9DEV signer IDs unknown, one collection ncslgr10p.xml, one URL session/date2001_8_6,9indexed source groups. EAF tier attributes do not establish verified global signer identities. An annotator/author name is not treated as a signer.'),
      ('PARTIAL','Internally consistent publisher label pools','High for joins','ASLLRP17453literal CSV events/2119utterances and all906NCSLGR TRAIN plus9DEV exact index strings pass.451newTRAIN+9DEV EAF event/boundary joins pass.403labels overlap, but the label/hand/hidden/HOLD conventions are not established as equivalent.'),
      ('BLOCKED','Merging the two target pools','High','Do not merge now. Feature representation matches; target scope and cross-version equivalence require a versioned expert-reviewed mapping or explicitly source-conditioned task, new vocab IDs, and an independent evaluation protocol. No labels/frames changed.'),
      ('UNKNOWN','Independent physical/linguistic boundary correctness','High confidence in limitation','Publisher frame/ms consistency is checked, not independent visual alignment. Some older NCSLGR records have only ordered utterance labels. Source timing does not itself certify visual boundary accuracy.'),
      ('PASS','Smallest decoder-only corrective test executed','High','Fixed prefix beam10/top16, no LM/penalty/sweep, identical logits/vocab/ref. Main remains100%WER; baseline85.96%, pilot96.49%. This tested decoder alternative does not fix the failure; pruned beam is not proof about every possible decoder.'),
      ('PARTIAL','Most likely underlying explanation: weak fit plus domain/task mismatch','Moderate','The selected checkpoint is still blank-collapsed even on TRAIN; the later checkpoint learns TRAIN but fails the old DEV. New pool has2739labels/17453tokens with1331singleton labels versus baseline115labels/494tokens. The source/annotation/camera change and tiny old DEV prevent useful selection. Contributions are confounded; no causal ablation or new fit was done.'),
      ('UNKNOWN','OOV or missing-hand observations as the sole root cause','Low support for sole-cause claim','Main has6OOV DEV tokens, baseline9yetbetterWER. All57deletions cannot be explained by6OOVs. DEV hand-observed rates25.8%left/39.8%right vs mainTRAIN47.8%/48.9%; real data-quality/domain differences exist but their causal effect is not isolated.'),
      ('BLOCKED','Production promotion and generalization claims','High','Technical prediction/data checks passed; signer-independent evaluation and cross-pool equivalence remain incomplete. The100%DEV checkpoint must not be promoted. The subsequent user instruction authorizes a separate bounded continuation on the unchanged research split; it does not establish product readiness.'),
    ]
    with (HERE/'findings.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['status','finding','confidence','evidence']);writer.writerows(results)
    lines=['# Diagnosis of the completed AppHear continuous-ASL run','',f'Created {date} (Asia/Shanghai).','',
      '**PASS diagnostic execution and preservation; PARTIAL causal attribution; BLOCKED production promotion and signer-independent claims.** A subsequent explicit user instruction authorizes a separate bounded research continuation once critical engineering checks pass. Those checks passed; see the authorization addendum below.','',
      'The direct cause of100%DEV WER is verified: the frozen epoch-2 model chooses CTC blank at all430DEV timesteps, so all nine hypotheses are empty and all57reference tokens are deleted. The decoder faithfully reflects the logits. No token-ID, blank-ID, length/truncation, vocabulary-loading, normalization or WER implementation bug was found in the bounded checks.','',
      'The most likely underlying explanation is an early blank solution selected by a poorly informative nine-utterance cross-corpus DEV criterion, combined with a much sparser large-vocabulary fitting problem and a source/annotation shift. Confidence is moderate in that attribution, high in the blank-output/selection mechanism. Those factors were not independently ablated, so this report does not claim that longer training, label normalization, a new learning rate or a decoder threshold would fix the problem.','',
      '## Reproduced results','',
      '| Frozen run | TRAIN | Vocab incl. blank/UNK | Selected epoch | DEV WER | S/D/I | Blank argmax frames | Empty hypotheses | OOV DEV tokens |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,title in [('baseline','expand03 baseline'),('pilot906','expand09 pilot'),('main2119','expand08 main')]:
        r=a['runs'][name];v=r['dev']
        lines.append(f'| {title} | {r["train"]} | {r["vocab_size"]} | {r["selected_epoch"]} | {100*v["gloss_wer"]:.2f}% | {v["substitutions"]}/{v["deletions"]}/{v["insertions"]} | {100*r["blank_argmax_fraction"]:.2f}% | {r["empty_hypotheses"]}/9 | {v["out_of_vocabulary_tokens"]}/57 |')
    lines += ['',
      'All three selected-checkpoint CPU replays match their saved literal hypotheses and edit counts exactly. Small CPU/CUDA loss differences are recorded, not interpreted as changed predictions. Baseline checkpoint SHA256:489ec5883ca440364ab10690ee5cfb9d0b0b0e6942df8f477509ea4ba6faf99c. Main selected checkpoint SHA256:f7a2b8bff5638671747352e2808f44fc2c60487a78d31a74cc49e978a46e7364. Main frozen manifest SHA256:adebeb9e2f20179eb59b7f79805ea9226ad6c9c0594e6cba193c771947cdaf40.','',
      'The two TRAIN numbers belong to different checkpoints: epoch17 has71.95324586%WER (3529S/8967D/62I;17453reference tokens), while selected epoch2 has99.99427033% (1S/17451D/0I). Both full2119TRAIN prediction lists were independently reproduced without fitting. Replaying the already logged epoch17 on DEV gives100%WER with10S/47D/0I and4/9empty hypotheses. Thus later fitting reduces the early global blank collapse on TRAIN, but does not recover correct tokens on this DEV. No later checkpoint was selected or promoted.','',
      'All17logged DEV WER values are1.0. The declared score is the tuple(WER,loss); epoch2 has the smallest DEV loss6.3637338744 and is correctly selected. That checkpoint emits blank on all430DEV frames. Its per-utterance mean blank probability ranges from86.39%to92.34%; even the best nonblank logit never beats blank. The model is not producing a hidden repeated-token string that is accidentally removed by postprocessing.','',
      '## Example-level references and hypotheses','',
      'Tokens below are literal source labels. Case, locus suffixes, plus signs and whole fingerspelled words are unchanged. The CSV contains all27selected-model/example rows, raw argmax IDs, blank probabilities, lengths, OOV labels and token IDs.','',
      '| Sample | Literal reference | Baseline hypothesis | Pilot hypothesis | Main hypothesis |','|---|---|---|---|---|']
    ids=[r['sample_id'] for r in mainrows[:3]]
    for sample in ids:
        group={r['run']:r for r in pred if r['sample_id']==sample}
        def seq(key,field):
            values=json.loads(group[key][field]);return ' '.join(values) if values else '<EMPTY>'
        lines.append(f'| `{sample}` | `{seq("main2119","reference")}` | `{seq("baseline","hypothesis")}` | `{seq("pilot906","hypothesis")}` | `{seq("main2119","hypothesis")}` |')
    lines += ['', '## Findings and confidence','', '| Status | Finding | Confidence | Evidence / limitation |','|---|---|---|---|']
    lines += [f'| {status} | {finding} | {confidence} | {evidence} |' for status,finding,confidence,evidence in results]
    lines += ['',
      '## What the bounded checks establish','',
      '- **PASS CTC convention:** blank0, unknown1, unique TRAIN-sorted vocabulary; every TRAIN target round-trips without blank/unknown and every cached sequence is CTC-feasible including adjacent repeats. The causal TCN uses stride1 and preserves T. No additional neural downsampling, fixed-window truncation or dropped feature blocks was found. Three synthetic CTCLoss values match exhaustive3^4path sums to float64 precision; these fixtures are not model-evaluation evidence.',
      '- **PASS decoding/metric:** independent raw-ID collapse agrees with the project decoder; argmax(logits) and argmax(log-softmax) agree; synthetic blank/repeat tests and independent Levenshtein distances pass. WER is total literal-token edits divided by57, not per-utterance averaging or whitespace re-tokenization. No case folding, punctuation stripping or English-to-gloss conversion occurs.',
      '- **PASS normalization:** each checkpoint carries its own TRAIN-only mean/std. Recomputing from its exact frozen TRAIN partition is bit-exact. Clipping affects only0.01292%of main-normalized DEV feature values versus0.02991%of mainTRAIN; gross normalization clipping is not supported as the cause. Checkpoint/config/vocabulary/manifest identity passes for all three models.',
      '- **PASS feature parity within tested scope:** all3025eligible records plus9DEV have the378float32 schema, finite features, increasing genuine timestamps and feasible CTC targets. Three cached raw examples (old NCSLGR, timed NCSLGR, ASLLRP) reproduce all378features exactly, including causal prefixes. No video decode or feature extraction was repeated. Frozen-model batched-padding/single-sequence/prefix logits agree within recorded numerical tolerance.',
      '- **PARTIAL temporal/linguistic validity:** publisher ASLLRP frame bounds and exact media counts, EAF millisecond bounds and literal annotation joins pass. This is not an independent review that the depicted signer action aligns with every event. Old utterance-only supervision cannot supply independent boundary truth.','',
      'The6main OOV tokens are HOLD×2, IX-loc-arc×1, READ++×2 and LATERwg×1. They impose a literal-token error floor of at least6/57=10.53%, not100%. The baseline has9OOVtokens and still scores better; the pilot has1OOVtoken and scores96.49%. Coverage alone is insufficient. CTC loss maps absent reference labels to<UNK> while WER retains their literal identities. This is consistent across runs, but losses across different vocabularies are not directly comparable and the DEV-loss tie-break is only a within-run rule.','',
      '## DEV provenance and comparison limits','',
      'The nine DEV items all come from ncslgr10p.xml, with indexed source groups986,987,988,989,990,992,993,994,996 and the same URL recording session2001_8_6. These are nine indexed media/source groups, not nine independent signers or confirmed independent recording sessions. All nine frozen signer_id fields are null; no trustworthy global IDs were recovered from the inspected EAF metadata. Annotation authors/annotators are not inferred as signers. The prompts repeatedly concern family, reading, buying and possessions, so lexical/topic coverage is narrow.',
      '',
      'Why this DEV set exists: expand03/tools/core.py explicitly assigned eligible collection ncslgr10p.xml to DEV and kept other collections for TRAIN, excluding the earlier exposed validation collections. Independently reconstructing its original filter reproduces exactly these nine IDs: frontal view0, a whole-utterance movie with an unambiguous annotation join,2..12lexical tokens and its original exclusion of quoted/complex/classifier/fingerspelled labels. Later TRAIN label policy broadened while DEV stayed fixed. This was a custom collection split, not an official publisher split, random representative validation sample or verified signer split. Later runs explicitly carried the same IDs/labels/feature hashes forward. See source_contract_checks.json and comparison_equivalence.csv.',
      '',
      'Known leakage checks found no TRAIN/DEV overlap in the checked IDs, collection, source_group, feature hash or raw-media hash for any of the three fits. Full existing package duplicate evidence was reused. Exact literal prompt repetitions are separately recorded and do not by themselves mean duplicate performances. Signer leakage and re-encoded or semantic duplicates remain unverified. The17reserved TEST utterances were not opened for predictions, annotation analysis or tuning. The initial validator hashes their existing protected artifacts as designed; no TEST inference occurred.',
      '',
      '## Label-pool compatibility','',
      'NCSLGR2 has906eligible utterances,920literal labels,5133target tokens and493singleton labels. ASLLRP3 has2119utterances,2739labels,17453tokens and1331singletons (48.59%of labels). They share403literal strings (union3256); identical strings do not establish identical annotation scope. The mean label exposure is6.37tokens/label in the main pool, with a strongly sparse tail. Those observations support a difficult data/learning problem, not proof of a particular optimizer defect.',
      '',
      'The ASLLRP3 source uses literal dominant-realized Occurrence label events, including publisher-hidden events and full compounds/fingerspelled labels. The NCSLGR2 pool uses literal main-gloss conventions, including source HOLD and repetition notation; its new timed subset obtains utterance spans from English-translation-tier coordinates, but never uses English values as target labels.17453ASLLRP CSV event joins,451timedTRAIN+9DEV EAF joins and915NCSLGR index string joins all pass. The shared feature function retains both hands/body and missingness; it does not make these label projections interchangeable.',
      '',
      'Merge decision: **BLOCKED** pending an explicitly versioned expert review of target scope and cross-version conventions, a new vocabulary mapping and a suitable independent evaluation protocol. No automatic stripping of wg,++,:i orHOLD; no relabeling to main-entry gloss; no guessed nondominant labels; no union manifest was fitted. All3025eligible records remain intact in their original separate pools.','',
      '## Smallest safe corrective test and next action','',
      '**Executed PASS test / negative correction result:** change only decoding on the saved logits to a predeclared prefix beam of10 with top16symbols/frame, preserving blank0, weights, normalization, vocabulary and references; no LM, insertion penalty, threshold sweep or selection. Synthetic exhaustive tests validate the implementation. It gives exactly the same WER for all three models, including100%for the main model. This tested decoder alternative does not repair the failure. It is a pruned approximation, so no claim about all beam search variants is made. The outputs are counterfactual diagnostics, not replacement official results.',
      '',
      '**Next data/evaluation step for generalization claims (BLOCKED pending verification):** complete a publisher-provenance signer/source mapping and have a qualified ASL annotator review a small source-matched evaluation packet under one declared literal target convention. The packet must come from authorized recordings genuinely unused for fitting, with a signer/source split frozen before fitting a new model on that protocol; previously trained records may only be called TRAIN diagnostics. Do not repurpose the protected17TEST set or silently rewrite these nine DEV references. Start with explicit review of HOLD, IX-loc-arc, READ++ and LATERwg scope and the dominant/hidden-event projection; proposed aliases are not ground truth. Independently verify representative visual boundaries and hand-observation quality. No acquisition, participant recording or relabeling occurred in this diagnostic.',
      '',
      '**Authorization addendum:** after these checks completed, the user explicitly directed a bounded pilot on the frozen2,119TRAIN/original9DEV split if no critical bug or data-integrity issue was found. The isolated internal label/feature contracts pass, the two pools stay separate and no critical implementation defect was found. Therefore a new bounded continuation was executed at `../20261020T102748Z_continuation_pilot01`, from the frozen epoch17 weights/AdamW/RNG state. Only the continuation budget/stopping protocol changes; model, learning rate, normalization, vocabulary, data and decoder remain fixed. Initial window3epochs, total cap8additional epochs/120seconds; continue beyond3only with a predeclared2/57token DEV gain and TRAIN learning. This is an authorized optimization test under the existing cross-corpus diagnostic, not a claim that the unresolved signer/evaluation gates passed. The pilot has separate actual logs/checkpoints/status and does not promote a model.','',
      f'**Measured budget-only test result: PARTIAL recognition, PASS execution.** The continuation completed{pilot_status["completed_additional_epochs"]}additional CUDA epochs18–20, then stopped with `{pilot_status["status"]}`. Full TRAIN WER improved71.9532%→{100*pilot_status["last_train_wer"]:.4f}%at epoch20, but all three DEV WER values remained100%. New-pilot DEV tie-break selected epoch18 (TRAIN WER{100*pilot_status["best_new_train_wer"]:.4f}%, DEV0correct tokens); this is a local pointer only and does not replace original epoch2. Epoch loop{pilot_status["epoch_loop_seconds"]:.3f}s; fitting/reload/fullTRAINdiagnostics{pilot_status["fit_reload_full_train_diagnostics_seconds"]:.3f}s; peak allocated{pilot_status["peak_allocated_vram_bytes"]/2**30:.4f}GiB. Best-new-checkpoint SHA256:`{pilot_status["best_checkpoint"]["sha256"]}`. This single-budget test did not repair DEV, so no further continuation was launched. The result strengthens evidence for a generalization/evaluation mismatch beyond simple initial blank collapse, without isolating its linguistic or visual cause.','',
      '## Reproduction, artifacts and preservation','',
      f'The CPU audit took{a["elapsed_seconds"]:.3f}s; supplemental label/CTC/final-epoch replay took{s["elapsed_seconds"]:.3f}s, excluding Python startup and initial package verification. No training/backward/optimizer update occurred in these diagnostic scripts; the separately authorized CUDA pilot above is explicitly distinguished. No downloads, production integration, APK, Android app data or avatar/GLB changes occurred. Diagnostic writes are confined to this directory, and new pilot writes to its own directory. The original06/08manifests,3025eligible examples, six overlap exclusions and one frame-count mismatch exclusion are unchanged.',
      '',
      'Required artifacts: DIAGNOSIS.md; prediction_audit.csv; comparison_equivalence.csv; label_pool_compatibility.csv; commands_and_results.txt; artifact_hashes.sha256. Detailed machine-readable evidence: checks.json, audit_results.json, annotation_join_checks.json, supplement_results.json, pool_details.json, ctc_lengths.csv, all saved DEV logits/counterfactuals, latest_epoch_dev_replay.json and preservation snapshots. All findings use PASS/PARTIAL/BLOCKED/UNKNOWN; PASS means only the stated bounded check.',
      '',
      'Read-only command (no fitting or TEST inference):','',
      '```powershell',f"& '{HERE / 'Verify-Diagnostics.ps1'}'",'```','',
      'Historical execution commands and exit results are preserved in commands_and_results.txt. artifact_hashes.sha256 covers the closed new diagnostic files except itself. The three *_input_artifacts.json/input_artifacts.json records contain hashes for frozen inputs actually read. verify_outputs.py validates those hashes and all protected metadata. The current shared CTC/model/decoder/metric source hash matches the hash captured in the main fitting provenance. No prior report/checkpoint was overwritten. The local continuity/review instructions were applied within the explicitly requested diagnostic scope; the existing root memory and progress documents remain unchanged.']
    # Render spacing in prose without changing literal examples, file paths or hashes.
    prose='\n'.join(lines)+'\n'
    for old,new in {
        'cause of100%':'cause of 100%', 'all430':'all 430', 'all57':'all 57', 'all17':'all 17', 'All17':'All 17',
        'all nine':'all nine', 'epoch-2':'epoch-2', 'epoch17':'epoch 17', 'epoch2':'epoch 2',
        '9/9hypotheses':'9/9 hypotheses','430DEV':'430 DEV','57reference':'57 reference',
        '2119TRAIN':'2,119 TRAIN','17453reference':'17,453 reference','logged DEV WER values are1.0':'logged DEV WER values are 1.0',
        'same9DEV':'same 9 DEV', 'All three':'All three','all3025':'all 3,025', 'All3025':'All 3,025',
        'over17453tokens':'over 17,453 tokens', '2739labels/17453tokens':'2,739 labels / 17,453 tokens',
        '1331singleton':'1,331 singleton','115labels/494tokens':'115 labels / 494 tokens',
        '6main OOV':'6 main OOV','HOLD×2':'HOLD × 2','IX-loc-arc×1':'IX-loc-arc × 1','READ++×2':'READ++ × 2','LATERwg×1':'LATERwg × 1',
        '9OOVtokens':'9 OOV tokens','1OOVtoken':'1 OOV token','same URL recording session2001':'same URL recording session 2001',
        'nocheckpoint':'no checkpoint', 'main has6OOV':'main has 6 OOV','baseline9yetbetter':'baseline 9 yet better',
        'to<UNK>':'to <UNK>', 'all27selected':'all 27 selected', 'width378':'width 378',
        '25.8%left/39.8%right':'25.8% left / 39.8% right','47.8%/48.9%':'47.8% / 48.9%',
        'standard15':'standard 15','lowest DEV loss6.363':'lowest DEV loss 6.363',
        'smallest DEV loss6.363':'smallest DEV loss 6.363', 'ranges from86.39%to92.34%':'ranges from 86.39% to 92.34%',
        'overlaps3025':'overlaps 3,025', 'all 3,025eligible':'all 3,025 eligible','All 3,025eligible':'All 3,025 eligible',
        'main2119TRAIN':'main 2,119 TRAIN', 'of mainTRAIN':'of main TRAIN',
        'all378features':'all 378 features','same378':'same 378', 'the378float32':'the 378 float32',
        'and9DEV':'and 9 DEV','plus9DEV':'plus 9 DEV', 'the17reserved':'the 17 reserved',
        'sequence ncslgr10p':'sequence ncslgr10p','groups986':'groups 986','view0':'view 0','join,2..12lexical':'join, 2–12 lexical',
        'with906eligible':'with 906 eligible', 'has906eligible':'has 906 eligible', '920literal':'920 literal',
        '5133target':'5,133 target','493singleton':'493 singleton','has2119utterances':'has 2,119 utterances',
        '2739labels,17453tokens':'2,739 labels, 17,453 tokens','48.59%of':'48.59% of','share403literal':'share 403 literal',
        'union3256':'union 3,256','is6.37tokens/label':'is 6.37 tokens/label',
        'only0.01292%of':'only 0.01292% of','versus0.02991%of':'versus 0.02991% of',
        '451timedTRAIN+9DEV':'451 timed TRAIN + 9 DEV','451newTRAIN+9DEV':'451 new TRAIN + 9 DEV',
        '915NCSLGR':'915 NCSLGR','17453ASLLRP':'17,453 ASLLRP','ASLLRP17453literal':'ASLLRP 17,453 literal',
        'column0':'column 0','and4/9empty':'and 4/9 empty','and all9DEV':'and all 9 DEV',
        'same100%':'same 100%','only6OOVs':'only 6 OOVs','by6OOVs':'by 6 OOVs',
        'of10 with top16symbols/frame':'of 10 with top 16 symbols/frame','preserving blank0':'preserving blank 0',
        'exhaustive3^4path':'exhaustive 3^4 path','blank0/UNK1':'blank 0 / UNK 1',
        'blank0, unknown1':'blank 0, unknown 1','stride1':'stride 1','blank0,':'blank 0,',
        'all 3,025records':'all 3,025 records','the current100%':'the current 100%',
        'Current100%':'Current 100%','The current100%':'The current 100%',
        'kept other collections':'kept other collections','/2119utterances':' / 2,119 utterances',
        'the same WER':'the same WER','including100%for':'including 100% for',
        'a blank solution':'a blank solution','selected epoch 2 has99.99427033%':'selected epoch 2 has 99.99427033%',
        'epoch 17 has71.95324586%WER':'epoch 17 has 71.95324586% WER',
        'gives100%WER':'gives 100% WER','have DEV WER1.0':'have DEV WER 1.0',
        'from ncslgr10p.xml':'from ncslgr10p.xml','No case folding':'No case folding'
    }.items():prose=prose.replace(old,new)
    save('DIAGNOSIS.md',prose)
    py=ROOT/'experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe'
    cmd=['AppHear continuous-ASL diagnostic command ledger',f'Finalized {date}','',
      '1. FIRST ACTION, completed exit0:',f"& '{ROOT / 'experiments/continuous_asl/20261020_continuous_expand08/Validate-Completed.ps1'}'",
      'Result: PASS_READ_ONLY_VERIFICATION;16132indexed new artifact entries,1268expand03content hashes,4prior guards,4606protected metadata checks,27additional frozen hashes/mtimes.17TESTnot evaluated; production false. Initial console evidence is in this chat tool session18162; no duplicate validation was performed.',
      '', '2. CPU inference and bounded contracts, completed exit0:',f"& '{py}' -B '{HERE / 'audit.py'}'",f'Result: {len(checks)}checks passed; {a["elapsed_seconds"]:.6f}s. See audit_console.txt, checks.json and audit_results.json.',
      '', '3. Targeted annotation joins, synthetic numeric CTC and already logged latest-epoch DEV replay, completed exit0:',f"& '{py}' -B '{HERE / 'supplement.py'}'",f'Result: {s["status"]};{s["EAF_joins"]}EAFjoins,{s["index_joins"]}indexjoins,4additional tests;{s["elapsed_seconds"]:.6f}s. See supplement_console.txt.',
      '', '4. Source/decoder contract and exact original DEV filter reconstruction, completed exit0:',f"& '{py}' -B '{HERE / 'verify_dev_origin.py'}'",'Result: PASS_SOURCE_CONTRACT_CHECKS; two checks passed. Shared fitting-source hash matches; original DEV filter reconstructs exactly nine IDs.',
      '', '5. Generate diagnosis and SHA256 inventory:',f"& '{py}' -B '{HERE / 'finalize_diagnosis.py'}'",'This command creates reports only. It does not fit, select models or evaluate TEST.',
      '', '6. Exact safe next/recheck command:',f"& '{HERE / 'Verify-Diagnostics.ps1'}'",'Read-only hashes and metadata; no inference. Verification stdout records actual result.',
      '', 'Failed exploratory reads retained in chat: guessed file tools/catalog_asllrp.py did not exist; actual asllrp_catalog.py was inspected. Original03 tools/initialize.py did not exist; actual tools/core.py was inspected. No data/training command failed in this diagnostic. Large exploratory output was truncated; subsequent reads were bounded. No dependencies were installed.',
      '', 'Only variable changed in the model-output counterfactual: decoder greedy -> fixed prefix beam10/top16. Results85.964912%,96.491228%,100%. No temperature, thresholds, normalization, weights, labels or checkpoint selection changed.',
      'All44main checks,4supplemental test records and2source-contract checks are bounded diagnostics. Synthetic fixtures are explicitly identified and never counted as model evaluation.']
    save('commands_and_results.txt','\n'.join(cmd)+'\n')
    save('readiness.json',json.dumps({'diagnosis_execution':'PASS','preservation':'PASS','root_cause_attribution':'PARTIAL','source_compatible_signer_independent_evaluation':'BLOCKED','bounded_research_pilot_explicitly_authorized_by_subsequent_user':True,'pilot_directory':'../20261020T102748Z_continuation_pilot01','production_promotion':'BLOCKED','train_data_kept':3025,'tests_used':False,'findings':dict(Counter(r[0] for r in results))},indent=2)+'\n')
    files=[p for p in sorted(HERE.iterdir()) if p.is_file() and p.name!='artifact_hashes.sha256']
    save('artifact_hashes.sha256','# Closed diagnostic files, SHA256; excludes this checksum file. Frozen input hashes are recorded in input_artifacts.json and supplement_input_artifacts.json.\n'+''.join(f'{digest(p)}  {p.name}\n' for p in files))
    print(json.dumps({'status':'DIAGNOSIS_SAVED','new_artifacts':len(files),'checks_passed':len(checks)+len(s['tests'])+len(origin['checks']),'preserved_metadata_files':len(before),'training_performed':False,'diagnosis_path':str(HERE/'DIAGNOSIS.md')},indent=2))

if __name__=='__main__':main()
