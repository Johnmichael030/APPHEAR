The later urgent directive requires useful training before all2,000 features
are ready. Experiment09 therefore freezes the complete906-example NCSLGR
TRAIN pool and runs a bounded pilot while ASLLRP preparation continues. This
main experiment requires its predeclared TRAIN learning signal to PASS as well
as the clean eight-utterance diagnostic. If the pilot remains deletion-dominant
with final TRAIN WER above80% or fails to halve monitored CTC loss, this main
run stops at the gate for an evidence-based diagnosis. No extra GPU hours are
launched blindly. Original9DEV selection and protected tests remain unchanged.
