param([ValidateSet('Verify','Acquire','ResumeScratch','ResumeWarm','ResumeRepeat','ResumeContinuous')][string]$Phase='Verify')
$ErrorActionPreference='Stop'
$repo='D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite'
$run='D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261020T004837_separate02'
Set-Location -LiteralPath $repo
switch($Phase){
 'Verify' { & "$repo/tools/asl_research_v2/Invoke-Phase.ps1" -RunRoot $run -Phase Verify }
 'Acquire' { & "$repo/tools/asl_research_v2/Invoke-How2Sign.ps1" -RunRoot $run -DryRun; & "$repo/tools/asl_research_v2/Invoke-How2Sign.ps1" -RunRoot $run }
 'ResumeScratch' { & "$repo/tools/asl_research_v2/Invoke-Phase.ps1" -RunRoot $run -Phase TrainA -Resume }
 'ResumeWarm' { & "$repo/tools/asl_research_v2/Invoke-Phase.ps1" -RunRoot $run -Phase WarmStart -Resume }
 'ResumeRepeat' { & "$repo/tools/asl_research_v2/Invoke-Phase.ps1" -RunRoot $run -Phase WarmRepeat -Resume }
 'ResumeContinuous' { & "$repo/tools/asl_research_v2/Invoke-Phase.ps1" -RunRoot $run -Phase TrainB -Resume }
}
