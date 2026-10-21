import json, pathlib
root=pathlib.Path.cwd()
task=pathlib.Path('evidence/current_continuous_automatic_task.txt').read_text().strip()
vocab=json.loads((root/'packages/continuous_asl_research_01/android_assets/continuous_asl/vocabulary.json').read_text())
# Explicit lexical equivalents only. No stripping of handshape, locus, variant, or classifier notation.
plain='HELLO WORLD GOOD MORNING AFTERNOON NIGHT WATER FOOD COFFEE TEA MILK HELP WANT NEED LIKE KNOW UNDERSTAND HAVE NOT PLEASE MORE FINISH YES NO THANKS HOME SCHOOL WORK TODAY TOMORROW YESTERDAY NOW LATER EARLY LATE GO COME STOP WAIT READY HAPPY SAD ANGRY TIRED HUNGRY THIRSTY HOT COLD BIG SMALL FAMILY FRIEND MOTHER FATHER BROTHER SISTER CHILDREN PERSON PEOPLE WOMAN MAN DEAF HEARING SIGN LANGUAGE ENGLISH BOOK PHONE COMPUTER NAME WHAT WHERE WHEN WHY HOW WHO SORRY AGAIN ALL SOME NOTHING SOMETHING EAT DRINK BUY GIVE SEE LOOK READ WRITE LEARN TEACH LOVE'.split()
words={w:w.lower() for w in plain if w in vocab}
words.update({k:v for k,v in {'IX-1p':'I','IX-2p':'you','THANK-YOU':'thank you'}.items() if k in vocab})
phrases=[]
def add(gloss,english):
    labels=gloss.split()
    if all(t in vocab for t in labels): phrases.append({'gloss':labels,'english':english})
for g,e in [('HELLO','Hello.'),('THANK-YOU','Thank you.'),('SORRY','Sorry.'),('GOOD MORNING','Good morning.'),('GOOD AFTERNOON','Good afternoon.'),('GOOD NIGHT','Good night.')]: add(g,e)
# Deliberately finite, literal engineering templates. These do not infer tense, questions,
# nonmanual grammar, spatial referents, or claim a qualified ASL translation review.
for subject,english_subject in [('IX-1p','I'),('IX-2p','You')]:
    for verb in ['WANT','NEED','LIKE','HAVE']:
        for obj in ['WATER','FOOD','COFFEE','TEA','MILK'] + (['HELP'] if verb in ['WANT','NEED'] else []):
            add(f'{subject} {verb} {obj}', f'{english_subject} {verb.lower()} {obj.lower()}.')
            add(f'{subject} NOT {verb} {obj}', f'{english_subject} do not {verb.lower()} {obj.lower()}.')
    for verb in ['UNDERSTAND','KNOW']:
        add(f'{subject} {verb}',f'{english_subject} {verb.lower()}.')
        add(f'{subject} NOT {verb}',f'{english_subject} do not {verb.lower()}.')
asset=root/'android/app/src/continuousAsl/assets/continuous_asl_output'
asset.mkdir(parents=True,exist_ok=True)
policy={'schema_version':'apphear.continuous_asl.automatic_display.v1','automatic_display':True,'automatic_speech':False,'automatic_history_commit':False,'stable_token_ms':200.0,'segment_idle_ms':1200.0,'status':'UNCALIBRATED_DISPLAY_HEURISTICS_NOT_ACCURACY_THRESHOLDS'}
language={'schema_version':'apphear.continuous_asl.limited_english.v1','qualified_linguistic_review':False,'provenance':'Manually authored engineering word equivalents and finite literal sentence templates; no participant data. Not an ASL grammar model. Not linguistically validated.','fallback':'Preserve order; show every unmapped literal label as [untranslated: LABEL]. Never remove annotation morphology or infer pronoun loci.','word_equivalents':words,'phrases':phrases}
for name,data in [('policy',policy),('english',language)]: (asset/f'{name}.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
result={'word_equivalents':len(words),'exact_phrase_templates':len(phrases),'model_classes':len(vocab),'unmapped_literal_targets':len(set(vocab[2:])-set(words))}
(pathlib.Path(task)/'english_coverage.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
