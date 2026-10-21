import csv,hashlib,json,pathlib
from collections import Counter,defaultdict
root=pathlib.Path.cwd();out=pathlib.Path('evidence/current_continuous_detection_task.txt').read_text().strip();out=pathlib.Path(out)
base=root/'experiments/continuous_asl/20261020T143609Z_learning_diagnostic01'
fold=json.loads((base/'fold_manifest.json').read_text());ids=set(fold['train_sample_ids'])
rows=[r for r in json.loads((base/'resolved_manifest.json').read_text()) if r['sample_id'] in ids]
assert len(rows)==len(ids)==1500 and len({r['sample_id'] for r in rows})==1500
assert len({r['training_pool'] for r in rows})==1
vocab=json.loads((root/'packages/continuous_asl_research_01/android_assets/continuous_asl/vocabulary.json').read_text())
assert set(vocab[2:])=={t for r in rows for t in r['glosses']}
support=Counter(t for r in rows for t in set(r['glosses']));occurrences=Counter(t for r in rows for t in r['glosses'])
signers=defaultdict(set);sources=defaultdict(set)
for r in rows:
 for t in set(r['glosses']): signers[t].add(r['signer_id']);sources[t].add(r['source_group'])
english=json.loads((root/'android/app/src/continuousAsl/assets/continuous_asl_output/english.json').read_text())['word_equivalents']
small=json.loads((root/'experiments/continuous_asl/20261021T112738_nested192_control01/subset_manifest.json').read_text())
small_support=Counter(t for r in small for t in set(r['glosses']))
entries=[{'model_id':i,'literal_label':t,'distinct_TRAIN_utterances':support[t],'token_occurrences':occurrences[t],'TRAIN_signers':len(signers[t]),'TRAIN_source_recordings':len(sources[t]),'nested192_utterances':small_support[t],'english_word_equivalent':english.get(t,'UNMAPPED'),'planning_gap_to5_utterances':max(0,5-support[t])} for i,t in enumerate(vocab) if i>=2]
with (out/'VOCABULARY_SUPPORT.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(entries[0]));w.writeheader();w.writerows(entries)
buckets={name:sum(f(n) for n in support.values()) for name,f in [('1',lambda n:n==1),('2_4',lambda n:2<=n<=4),('5_9',lambda n:5<=n<=9),('10_19',lambda n:10<=n<=19),('20plus',lambda n:n>=20)]}
summary={'status':'PASS TRAIN-only support audit; NO recognition vocabulary expansion','TRAIN_utterances':len(rows),'literal_target_labels':len(support),'output_classes_with_specials':len(vocab),'token_occurrences':sum(occurrences.values()),'support_buckets_distinct_utterances':buckets,'TRAIN_signers':len(set(r['signer_id'] for r in rows)),'TRAIN_source_recordings':len(set(r['source_group'] for r in rows)),'labels_without5support':sum(n<5 for n in support.values()),'English_word_equivalents':len(english),'new_recognition_classes_added':0,'reason':'The current task already contains2,287 literal labels; sparse support and transfer failure dominate. Appending vocabulary strings cannot change the trained classifier. Keep pools separate and add authentic examples before expanding classes.'}
(out/'VOCABULARY_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
report='# Recognition vocabulary decision\n\n**PASS audit; no new recognition classes added.**\n\n'
report+=f'The current model has {len(support):,} literal target labels plus blank0 andUNK1. These are ASLLRP occurrence labels, including variants and spatial/classifier notation; they are not {len(support):,} interchangeable English words.\n\n'
report+=f'In the frozen1,500TRAIN utterances, {buckets["1"]:,} labels occur in only one distinct utterance and {sum(n<5 for n in support.values()):,} occur in fewer than five. TRAIN has{summary["TRAIN_signers"]}publisher signer IDs. Repeated token occurrences within one utterance do not count as additional support.\n\n'
report+='Adding classes now would dilute already sparse supervision. The model output matrix, vocabulary IDs and genuine labeled examples must be trained/evaluated together; adding strings or English aliases cannot teach visual recognition. No target/label normalization, ontology merger, fabricated examples or unsupported semantics were introduced.\n\n'
report+='VOCABULARY_SUPPORT.csv contains the complete current literal label list, model IDs, distinct-utterance support, occurrence frequencies, signer/source coverage and existing English-equivalent mapping. A planning gap to5 is informational, not a validated quality threshold.\n\n'
report+='Next data step: qualified review of the existing12TRAIN event package, then consented natural phrases repeating a small chosen set of existing labels across independent signers/sessions, with predeclared no-sign and unseen-signer evaluation. No qualified reviewer or new participant permissions are currently established. English display coverage remains a separate75-label/40-pattern stage and was not expanded or presented as a recognition improvement.\n'
(out/'VOCABULARY_DECISION.md').write_text(report,encoding='utf-8')
print(json.dumps(summary,indent=2))
