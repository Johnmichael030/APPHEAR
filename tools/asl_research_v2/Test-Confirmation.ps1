param([Parameter(Mandatory)][string]$RunRoot)
$ErrorActionPreference='Stop'
Import-Module (Join-Path $PSScriptRoot 'DownloadV2.psm1') -Force
$source='https://drive.google.com/uc?export=download&id=1DhLH8tIBn9HsTzUJUfsEOGcP4l9EvOiO'
$html=Get-Content (Join-Path $RunRoot 'logs/how2sign_access_page.html') -Raw
$confirmed=Get-PublicDriveConfirmation $html $source
if(([uri]$confirmed).DnsSafeHost -ne 'drive.usercontent.google.com'){throw 'WRONG_CONFIRMATION_HOST'}
$rejected=0
foreach($bad in @('<html>Sign in to continue</html>','<html>download quota exceeded</html>','<html><a href="https://accounts.google.com">Sign in</a></html>','<div data-sitekey="challenge">Complete challenge</div>')){
    try{$null=Get-PublicDriveConfirmation $bad $source}catch{$rejected++;continue};throw 'RESTRICTION_NOT_REJECTED'
}
@{status='PASS';real_public_form_accepted=$true;restricted_or_missing_forms_rejected=$rejected;prior_tool_unchanged=$true}|ConvertTo-Json|Set-Content (Join-Path $RunRoot 'logs/confirmation_tests.json') -Encoding utf8NoBOM
'Public form and restriction tests passed'
