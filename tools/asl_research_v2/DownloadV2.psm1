# Public, bounded, resumable HTTPS acquisition. No authentication or TLS bypass.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'Common.psm1')
function Get-DownloadPlan {
    param([string]$ManifestPath,[string]$RunRoot,[string[]]$Ids,[long]$MaxTotalBytes=67108864)
    $run = Get-ExperimentRoot $RunRoot
    $manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json -AsHashtable
    if ($manifest.schema -ne 'apphear.continuous_asl.download_plan.v1') { throw 'MANIFEST_SCHEMA' }
    $selected = @($manifest.items | Where-Object { !$Ids -or $_.id -in $Ids })
    if ($Ids -and @($selected).Count -ne @($Ids | Select-Object -Unique).Count) { throw 'UNKNOWN_DOWNLOAD_ID' }
    if (@($selected.id | Select-Object -Unique).Count -ne $selected.Count) { throw 'DUPLICATE_DOWNLOAD_ID' }
    $total = 0L
    $outputs = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($entry in $selected) {
        if ($entry.authorization -ne 'RESEARCH_METADATA_OR_INITIAL_SAMPLE') { throw "ACQUISITION_NOT_APPROVED:$($entry.id)" }
        $uri = [Uri]$entry.url
        if (!$uri.IsAbsoluteUri -or $uri.Scheme -ne 'https' -or $uri.UserInfo) { throw 'HTTPS_PUBLIC_URL_REQUIRED' }
        if ($uri.DnsSafeHost -notin $manifest.allowed_hosts) { throw 'DOWNLOAD_HOST_NOT_ALLOWLISTED' }
        if ($entry.split -eq 'test') { throw 'TEST_PAYLOAD_NOT_AUTHORIZED' }
        if ([long]$entry.max_bytes -le 0) { throw 'BYTE_LIMIT_REQUIRED' }
        $target = Assert-IsolatedOutput (Join-Path $run ('data/raw/' + $entry.filename)) (Join-Path $run 'data/raw')
        if (!$outputs.Add($target)) { throw 'DUPLICATE_OUTPUT_PATH' }
        $total += [long]$entry.max_bytes
        [PSCustomObject]@{id=$entry.id;url=$entry.url;kind=$entry.kind;split=$entry.split;
            target=$target;max_bytes=[long]$entry.max_bytes;expected_bytes=$entry.expected_bytes;
            expected_sha256=$entry.sha256;max_expanded_bytes=[long]$entry.max_expanded_bytes;
            annotation_type=$entry.annotation_type;allowed_hosts=$manifest.allowed_hosts}
    }
    if ($total -gt $MaxTotalBytes) { throw "DOWNLOAD_BUDGET_EXCEEDED:$total > $MaxTotalBytes" }
    $drive = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($run))
    if ($drive.AvailableFreeSpace -lt ($total + 1073741824L)) { throw 'INSUFFICIENT_DISK_HEADROOM' }
}
function Assert-PayloadFormat {
    param([string]$Path,[string]$Kind)
    $stream=[IO.File]::OpenRead($Path)
    try { $prefix=[byte[]]::new(2048);$length=$stream.Read($prefix,0,$prefix.Length) } finally { $stream.Dispose() }
    $text=[Text.Encoding]::UTF8.GetString($prefix,0,$length).TrimStart([char[]]@([char]0xFEFF,[char]32,[char]13,[char]10))
    if ($Kind -ne 'html' -and $text -match '(?is)^(<!doctype\s+html|<html|<head|<body)') { throw 'HTML_IS_NOT_A_DATASET_PAYLOAD' }
    switch ($Kind) {
        'zip' { if ($length -lt 4 -or $prefix[0] -ne 80 -or $prefix[1] -ne 75) { throw 'ZIP_MAGIC_MISMATCH' } }
        'gzip' { if ($length -lt 2 -or $prefix[0] -ne 31 -or $prefix[1] -ne 139) { throw 'GZIP_MAGIC_MISMATCH' } }
        'json' { $null=Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json -AsHashtable -ErrorAction Stop }
        'csv' { if ($length -eq 0 -or $text.IndexOf([char]0) -ge 0) { throw 'INVALID_TEXT_PAYLOAD' } }
        'xml' { if ($text -notmatch '^<') { throw 'XML_PREFIX_MISMATCH' } }
        'video' { if ($length -lt 12 -or [Text.Encoding]::ASCII.GetString($prefix,4,4) -notin @('ftyp','moov','mdat','wide','free','skip')) { throw 'UNRECOGNIZED_VIDEO_CONTAINER' } }
        {$_ -in 'html','text'} { if ($length -eq 0) { throw 'EMPTY_PAYLOAD' } }
        default { if ($Kind -notin @('zip','gzip','json','csv','xml','video','html','text')) { throw 'UNSUPPORTED_PAYLOAD_KIND' } }
    }
}
function Get-PublicDriveConfirmation {
    param([string]$Html,[string]$OriginalUrl)
    # A hidden .grecaptcha-badge stylesheet on a public warning is not a challenge.
    # Reject actual challenge widgets/scripts and restriction text, not CSS/link names.
    $visible=[regex]::Replace($Html,'(?is)<(style|script)\b[^>]*>.*?</\1>',' ')
    $visible=[regex]::Replace($visible,'(?s)<[^>]+>',' ')
    if ($Html -match '(?is)<(?:iframe|script)\b[^>]*(?:recaptcha|hcaptcha)|data-sitekey\s*=' -or $visible -match '(?i)quota|too many users|access denied|request access|captcha|you need access|sign in to continue|verify you are human') {
        throw 'DRIVE_QUOTA_AUTH_OR_ACCESS_RESTRICTION: no bypass attempted'
    }
    $form=[regex]::Match($Html,'(?is)<form\b[^>]*id="download-form"[^>]*>.*?</form>')
    if (!$form.Success) { throw 'DRIVE_HTML_OR_EXPIRED_LINK: no public confirmation form' }
    $action=[regex]::Match($form.Value,'action="([^"]+)"').Groups[1].Value
    $endpoint=[Uri][System.Web.HttpUtility]::HtmlDecode($action)
    if ($endpoint.Scheme -ne 'https' -or $endpoint.DnsSafeHost -ne 'drive.usercontent.google.com') { throw 'UNTRUSTED_DRIVE_CONFIRMATION' }
    $fields=@{}
    foreach($inputTag in [regex]::Matches($form.Value,'(?is)<input\b[^>]*>')) {
        $name=[regex]::Match($inputTag.Value,'name="([^"]+)"').Groups[1].Value
        $value=[regex]::Match($inputTag.Value,'value="([^"]*)"').Groups[1].Value
        if ($name -notin @('id','export','confirm','uuid')) { continue }
        $fields[$name]=[System.Web.HttpUtility]::HtmlDecode($value)
    }
    $originalId=[System.Web.HttpUtility]::ParseQueryString(([Uri]$OriginalUrl).Query)['id']
    if (!$originalId -or $fields['id'] -ne $originalId) { throw 'DRIVE_FILE_ID_CHANGED' }
    $query=($fields.GetEnumerator() | ForEach-Object {[Uri]::EscapeDataString($_.Key)+'='+[Uri]::EscapeDataString($_.Value)}) -join '&'
    return $endpoint.GetLeftPart([UriPartial]::Path)+'?'+$query
}
function Assert-ResumeResponse {
    param([long]$Offset,[int]$Status,$Range,[string]$OldValidator,[string]$NewValidator,[long]$MaximumBytes)
    if($Offset -eq 0){if($Status -ne 200){throw 'UNREQUESTED_PARTIAL_RESPONSE'};return}
    if($Status -ne 206 -or !$Range -or $Range.Unit -ne 'bytes' -or $Range.From -ne $Offset){throw 'RANGE_NOT_HONOURED'}
    if(!$OldValidator -or !$NewValidator -or $NewValidator -ne $OldValidator){throw 'RESUME_VALIDATOR_CHANGED_OR_MISSING'}
    if(!$Range.HasLength -or !$Range.HasRange -or $Range.To -ne ($Range.Length-1) -or $Range.Length -gt $MaximumBytes){throw 'INCOMPLETE_OR_OVERSIZE_RESUMED_RANGE'}
}
function Invoke-PublicDownload {
    param($Entry,[string]$RunRoot,[int]$MaximumAttempts=3,[int]$TimeoutSeconds=180)
    $run=Get-ExperimentRoot $RunRoot
    $target=Assert-IsolatedOutput $Entry.target (Join-Path $run 'data/raw')
    $receiptPath=$target+'.receipt.json'
    if (Test-Path -LiteralPath $target) {
        if (!(Test-Path -LiteralPath $receiptPath)) { throw 'EXISTING_FILE_WITHOUT_RECEIPT' }
        $receipt=Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json -AsHashtable
        $hash=Get-FileSha256 $target
        if ($hash -ne $receipt.sha256 -or $receipt.source_url -ne $Entry.url) { throw 'CACHED_PAYLOAD_IDENTITY_MISMATCH' }
        if ($Entry.expected_sha256 -and $hash -ne $Entry.expected_sha256) { throw 'OFFICIAL_HASH_MISMATCH' }
        Assert-PayloadFormat $target $Entry.kind
        return [PSCustomObject]@{id=$Entry.id;status='VERIFIED_REUSE';bytes=(Get-Item $target).Length;sha256=$hash}
    }
    $part=$target+'.part'
    $httpMeta=$part+'.http.json'
    $null=[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($target))
    $lastError=$null
    for($attempt=1;$attempt -le $MaximumAttempts;$attempt++) {
        $handler=[Net.Http.HttpClientHandler]::new()
        $handler.AllowAutoRedirect=$false
        $handler.UseCookies=$true
        $handler.CookieContainer=[Net.CookieContainer]::new()
        $client=[Net.Http.HttpClient]::new($handler)
        $client.Timeout=[Threading.Timeout]::InfiniteTimeSpan
        $cancel=[Threading.CancellationTokenSource]::new([TimeSpan]::FromSeconds($TimeoutSeconds))
        $response=$null
        try {
            $offset=0L;$validator=$null
            if(Test-Path -LiteralPath $part) {
                if(!(Test-Path -LiteralPath $httpMeta)) { throw 'PARTIAL_WITHOUT_VALIDATOR: retained for review' }
                $old=Get-Content -LiteralPath $httpMeta -Raw | ConvertFrom-Json -AsHashtable
                if($old.source_url -ne $Entry.url -or !$old.validator) { throw 'PARTIAL_WITHOUT_VALIDATOR: retained for review' }
                $offset=(Get-Item -LiteralPath $part).Length;$validator=$old.validator
            }
            $url=[string]$Entry.url;$confirmed=$false
            for($redirect=0;$redirect -le 8;$redirect++) {
                $uri=[Uri]$url
                if($uri.Scheme -ne 'https' -or $uri.DnsSafeHost -notin $Entry.allowed_hosts -or $uri.UserInfo) { throw 'UNTRUSTED_REDIRECT' }
                $request=[Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Get,$uri)
                $request.Headers.UserAgent.ParseAdd('AppHear-Research-Acquisition/1.0')
                if($offset -gt 0) {
                    $null=$request.Headers.TryAddWithoutValidation('Range',"bytes=$offset-")
                    $null=$request.Headers.TryAddWithoutValidation('If-Range',$validator)
                }
                try { $response=$client.SendAsync($request,[Net.Http.HttpCompletionOption]::ResponseHeadersRead,$cancel.Token).GetAwaiter().GetResult() }
                finally { $request.Dispose() }
                $status=[int]$response.StatusCode
                if($status -in @(301,302,303,307,308)) {
                    $url=[Uri]::new($uri,$response.Headers.Location).AbsoluteUri
                    $response.Dispose();$response=$null;continue
                }
                if($status -in @(429,500,502,503,504)) { throw "TRANSIENT_HTTP_$status" }
                if($status -ne 200 -and $status -ne 206) { throw "HTTP_$status" }
                $contentType=[string]$response.Content.Headers.ContentType
                if($contentType -match '^text/html' -and $Entry.kind -ne 'html') {
                    if($uri.DnsSafeHost -notin @('drive.google.com','drive.usercontent.google.com','docs.google.com') -or $confirmed -or $offset -gt 0) { throw 'HTML_IS_NOT_A_DATASET_PAYLOAD' }
                    $htmlStream=$response.Content.ReadAsStreamAsync($cancel.Token).GetAwaiter().GetResult()
                    try {
                        $buffer=[byte[]]::new(65537);$count=0
                        while($count -lt $buffer.Length) {$n=$htmlStream.ReadAsync($buffer,$count,$buffer.Length-$count,$cancel.Token).GetAwaiter().GetResult();if($n -eq 0){break};$count+=$n}
                        if($count -gt 65536){throw 'DRIVE_CONFIRMATION_PAGE_TOO_LARGE'}
                        $url=Get-PublicDriveConfirmation ([Text.Encoding]::UTF8.GetString($buffer,0,$count)) $Entry.url
                        $confirmed=$true
                    } finally {$htmlStream.Dispose();$response.Dispose();$response=$null}
                    continue
                }
                break
            }
            if(!$response -or $redirect -gt 8) { throw 'REDIRECT_LIMIT' }
            if($offset -gt 0 -and ($status -ne 206 -or $response.Content.Headers.ContentRange.From -ne $offset)) { throw 'RANGE_NOT_HONOURED: retained partial; no overwrite' }
            if($offset -eq 0 -and $status -eq 206) { throw 'UNREQUESTED_PARTIAL_RESPONSE' }
            $length=$response.Content.Headers.ContentLength
            if($null -ne $length -and $offset+$length -gt $Entry.max_bytes) { throw 'HTTP_LENGTH_EXCEEDS_BUDGET' }
            $newValidator=if($response.Headers.ETag -and !$response.Headers.ETag.IsWeak){$response.Headers.ETag.ToString()}elseif($response.Content.Headers.LastModified){$response.Content.Headers.LastModified.ToString('R')}else{$null}
            Assert-ResumeResponse $offset $status $response.Content.Headers.ContentRange $validator $newValidator $Entry.max_bytes
            Write-ExperimentJson @{source_url=$Entry.url;validator=$newValidator;http_status=$status;content_type=$contentType} $httpMeta $run
            $mode=if($offset -gt 0){[IO.FileMode]::Append}else{[IO.FileMode]::CreateNew}
            $output=[IO.File]::Open($part,$mode,[IO.FileAccess]::Write,[IO.FileShare]::None)
            $input=$response.Content.ReadAsStreamAsync($cancel.Token).GetAwaiter().GetResult()
            $count=0L
            try {
                $buffer=[byte[]]::new(1048576)
                while(($read=$input.ReadAsync($buffer,0,$buffer.Length,$cancel.Token).GetAwaiter().GetResult()) -gt 0) {
                    $count+=$read
                    if($offset+$count -gt $Entry.max_bytes) { throw 'STREAM_BYTE_LIMIT_EXCEEDED' }
                    $output.Write($buffer,0,$read)
                }
                $output.Flush($true)
            } finally {$input.Dispose();$output.Dispose()}
            if($null -ne $length -and $count -ne $length) { throw 'INCOMPLETE_HTTP_BODY' }
            $bytes=(Get-Item -LiteralPath $part).Length
            if($Entry.expected_bytes -and $bytes -ne $Entry.expected_bytes) { throw 'EXPECTED_SIZE_MISMATCH' }
            Assert-PayloadFormat $part $Entry.kind
            $hash=Get-FileSha256 $part
            if($Entry.expected_sha256 -and $hash -ne $Entry.expected_sha256) { throw 'OFFICIAL_HASH_MISMATCH' }
            [IO.File]::Move($part,$target,$false)
            Write-ExperimentJson @{id=$Entry.id;source_url=$Entry.url;bytes=$bytes;sha256=$hash;
                official_hash_available=[bool]$Entry.expected_sha256;kind=$Entry.kind;split=$Entry.split;
                annotation_type=$Entry.annotation_type;completed_utc=[DateTime]::UtcNow.ToString('o')} $receiptPath $run
            return [PSCustomObject]@{id=$Entry.id;status='DOWNLOADED';bytes=$bytes;sha256=$hash}
        } catch {
            $lastError=$_.Exception.Message
            Write-ExperimentJson @{id=$Entry.id;attempt=$attempt;error=$lastError;source_url=$Entry.url;
                partial_retained=Test-Path -LiteralPath $part;time_utc=[DateTime]::UtcNow.ToString('o')} (Join-Path $run ("logs/download_"+$Entry.id+"_attempt"+$attempt+".json")) $run
            if($lastError -notmatch 'TRANSIENT_HTTP_|timed out|canceled|transport|connection|incomplete|name resolution' -or $attempt -eq $MaximumAttempts) { throw }
            Start-Sleep -Seconds ([Math]::Min(8,[Math]::Pow(2,$attempt)))
        } finally {
            if($response){$response.Dispose()}
            $cancel.Dispose();$client.Dispose()
        }
    }
    throw $lastError
}
Export-ModuleMember -Function Get-DownloadPlan,Assert-PayloadFormat,Get-PublicDriveConfirmation,Assert-ResumeResponse,Invoke-PublicDownload


