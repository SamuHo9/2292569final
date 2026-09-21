param(
    [Parameter(Mandatory = $true)]
    [string]$LeftMaskDir,
    [Parameter(Mandatory = $true)]
    [string]$RightMaskDir,
    [string]$SplitManifest,
    [string]$LeftStatus,
    [string]$RightStatus,
    [string]$OutputRoot,
    [string]$PythonExe = 'python',
    [string]$SlicerExe = 'C:\Program Files\SlicerSALT 6.0.0\SlicerSALT.exe'
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$prepareScript = Join-Path $PSScriptRoot 'prepare_mask_split.py'
$icpScript = Join-Path $PSScriptRoot 'ICP.py'
if (-not $SplitManifest) { $SplitManifest = Join-Path $projectRoot 'SPHARM\split_data\current_split_manifest.json' }
if (-not $LeftStatus) { $LeftStatus = Join-Path $projectRoot 'SPHARM\split_data\ALL_Left_file_status.csv' }
if (-not $RightStatus) { $RightStatus = Join-Path $projectRoot 'SPHARM\split_data\ALL_Right_file_status.csv' }
if (-not $OutputRoot) { $OutputRoot = Join-Path $PSScriptRoot 'fixed_reference_rerun' }

foreach ($requiredPath in @($prepareScript, $icpScript, $SplitManifest, $LeftStatus, $RightStatus, $SlicerExe)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required file not found: $requiredPath"
    }
}
foreach ($sourceDir in @($LeftMaskDir, $RightMaskDir)) {
    if (-not (Test-Path -LiteralPath $sourceDir -PathType Container)) {
        throw "Mask directory not found: $sourceDir"
    }
}

$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
if ((Test-Path -LiteralPath $OutputRoot) -and (Get-ChildItem -Force -LiteralPath $OutputRoot | Select-Object -First 1)) {
    throw "OutputRoot is not empty. Choose a fresh folder so old ICP files cannot mix with this run: $OutputRoot"
}

Write-Host '[1/4] Preparing copies of the audited patient-level train/test masks...'
& $PythonExe $prepareScript `
    --left-source $LeftMaskDir `
    --right-source $RightMaskDir `
    --split-manifest $SplitManifest `
    --left-status $LeftStatus `
    --right-status $RightStatus `
    --output-root $OutputRoot
if ($LASTEXITCODE -ne 0) { throw "Mask split preparation failed (exit $LASTEXITCODE)" }

function Invoke-IcpJob {
    param(
        [Parameter(Mandatory = $true)][string]$InputDir,
        [Parameter(Mandatory = $true)][string]$JobOutput,
        [Parameter(Mandatory = $true)][ValidateSet('fit_reference', 'fixed_reference')][string]$Mode,
        [string]$ReferenceTemplate
    )
    $arguments = @(
        '--no-main-window', '--no-splash', '--python-script', $icpScript,
        '--input_dir', $InputDir, '--output_dir', $JobOutput,
        '--output_voxels', '128', '--max_iterations', '20', '--tolerance', '0.00005',
        '--pairwise_iterations', '100', '--pairwise_tolerance', '0.0001',
        '--pairwise_landmarks', '200', '--interpolation', 'nn'
    )
    if ($Mode -eq 'fit_reference') {
        $arguments += '--fit_reference'
    } else {
        if (-not $ReferenceTemplate) { throw 'fixed_reference requires a reference template' }
        $arguments += @('--reference_template', $ReferenceTemplate)
    }

    Write-Host "  ICP $Mode`: $InputDir"
    $argumentLine = ($arguments | ForEach-Object {
        '"' + ([string]$_).Replace('"', '\"') + '"'
    }) -join ' '
    try {
        $slicerProcess = Start-Process -FilePath $SlicerExe -ArgumentList $argumentLine `
            -PassThru -Wait -WindowStyle Hidden
        $processExit = $slicerProcess.ExitCode
    } catch {
        throw "Could not launch or wait for SlicerSALT: $($_.Exception.Message)"
    }

    $statusPath = Join-Path $JobOutput 'icp_status.json'
    if (-not (Test-Path -LiteralPath $statusPath -PathType Leaf)) {
        throw "ICP did not write status file: $statusPath (process exit $processExit)"
    }
    $status = Get-Content -Raw -Encoding utf8 -LiteralPath $statusPath | ConvertFrom-Json
    if ($status.success -ne $true -or $status.mode -ne $Mode) {
        throw "ICP status is not successful for ${JobOutput}: $($status | ConvertTo-Json -Compress)"
    }
    $alignedDir = Join-Path $JobOutput 'aligned_nifti'
    $alignedCount = @(Get-ChildItem -File -LiteralPath $alignedDir -Filter '*.nii.gz').Count
    if ($alignedCount -ne [int]$status.subjects -or $alignedCount -eq 0) {
        throw "Aligned NIfTI count mismatch at ${alignedDir}: status=$($status.subjects), files=$alignedCount"
    }
    if ($processExit -ne 0) {
        Write-Warning "SlicerSALT returned exit $processExit, but its success status and aligned-file count passed: $JobOutput"
    }
    return $status
}

$jobs = @()
Write-Host '[2/4] Fitting one groupwise ICP reference per hemisphere from TRAIN masks only...'
foreach ($side in @('left', 'right')) {
    $fitInput = Join-Path $OutputRoot "masks\$side\train"
    $fitOutput = Join-Path $OutputRoot "icp\$side\reference_fit"
    $fitStatus = Invoke-IcpJob -InputDir $fitInput -JobOutput $fitOutput -Mode 'fit_reference'
    $template = Join-Path $fitOutput 'mean_shape.ply'
    $templateContract = "${template}.json"
    if (-not (Test-Path -LiteralPath $template -PathType Leaf) -or
        -not (Test-Path -LiteralPath $templateContract -PathType Leaf)) {
        throw "Training reference or its metadata is missing for ${side}: $template"
    }
    $templateHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $template).Hash.ToLowerInvariant()
    $contract = Get-Content -Raw -Encoding utf8 -LiteralPath $templateContract | ConvertFrom-Json
    if ($contract.template_sha256 -ne $templateHash -or $fitStatus.reference_sha256 -ne $templateHash) {
        throw "Reference hash mismatch for $side; do not continue to test alignment."
    }
    $jobs += [pscustomobject]@{
        Side = $side; Split = 'train_reference_fit'; Mode = $fitStatus.mode
        Subjects = [int]$fitStatus.subjects; Reference = $template; ReferenceSHA256 = $templateHash
        Output = $fitOutput
    }

    Write-Host "[3/4] Re-aligning both $side train and test masks to the frozen training reference..."
    foreach ($split in @('train', 'test')) {
        $inputDir = Join-Path $OutputRoot "masks\$side\$split"
        $jobOutput = Join-Path $OutputRoot "icp\$side\${split}_fixed"
        $status = Invoke-IcpJob -InputDir $inputDir -JobOutput $jobOutput `
            -Mode 'fixed_reference' -ReferenceTemplate $template
        if ($status.reference_sha256 -ne $templateHash) {
            throw "The $side $split job did not use the frozen training reference."
        }
        $jobs += [pscustomobject]@{
            Side = $side; Split = $split; Mode = $status.mode; Subjects = [int]$status.subjects
            Reference = $template; ReferenceSHA256 = $status.reference_sha256; Output = $jobOutput
        }
    }
}

$result = [pscustomobject]@{
    Schema = 'fixed_reference_icp_rerun_v1'
    GeneratedAt = (Get-Date).ToUniversalTime().ToString('o')
    SplitManifest = (Resolve-Path -LiteralPath $SplitManifest).Path
    StagedManifest = (Join-Path $OutputRoot 'pre_icp_split_manifest.json')
    Jobs = $jobs
    NextStep = 'Run SPHARM separately for each side/split using the fixed production template; keep each ICP output directory as the SPHARM output root so its processing contract reads icp_status.json.'
}
$reportPath = Join-Path $OutputRoot 'fixed_reference_icp_rerun_manifest.json'
$result | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 -LiteralPath $reportPath
Write-Host '[4/4] Fixed-reference ICP rerun finished and verified.'
Write-Host "Run report: $reportPath"
