param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('left', 'right')]
    [string]$Side,
    [string]$InputDir,
    [string]$InputFile,
    [Parameter(Mandatory = $true)]
    [string]$OutputDir,
    [string]$ReferenceRoot,
    [string]$SlicerExe = 'C:\Program Files\SlicerSALT 6.0.0\SlicerSALT.exe'
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$icpScript = Join-Path $PSScriptRoot 'ICP.py'
if (-not $ReferenceRoot) {
    $ReferenceRoot = Join-Path $PSScriptRoot 'references\legacy_groupwise_all_381_v1'
}

if ([bool]$InputDir -eq [bool]$InputFile) {
    throw 'Specify exactly one input mode: -InputDir for batch or -InputFile for a single mask.'
}
if ($InputDir -and -not (Test-Path -LiteralPath $InputDir -PathType Container)) {
    throw "Input directory not found: $InputDir"
}
if ($InputFile -and -not (Test-Path -LiteralPath $InputFile -PathType Leaf)) {
    throw "Input file not found: $InputFile"
}
foreach ($requiredFile in @($icpScript, $SlicerExe)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required file not found: $requiredFile"
    }
}

$referenceDir = Join-Path $ReferenceRoot $Side
$referenceTemplate = Join-Path $referenceDir 'mean_shape.ply'
$referenceMetadata = "$referenceTemplate.json"
foreach ($requiredFile in @($referenceTemplate, $referenceMetadata)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Reference bundle is incomplete: $requiredFile"
    }
}
$contract = Get-Content -Raw -Encoding UTF8 -LiteralPath $referenceMetadata | ConvertFrom-Json
if ($contract.version -ne 'fixed-legacy-groupwise-reference-v1' -or
    $contract.independent_test_reference -ne $false) {
    throw 'This runner expects the audited legacy reference bundle and its all-subject provenance flag.'
}
$actualReferenceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $referenceTemplate).Hash.ToLowerInvariant()
if ($contract.template_sha256 -ne $actualReferenceHash) {
    throw "Reference hash mismatch for $referenceTemplate"
}

if ($InputFile) {
    $expectedInputs = 1
    $inputNames = @((Get-Item -LiteralPath $InputFile).Name)
} else {
    $inputFiles = @(Get-ChildItem -LiteralPath $InputDir -File -Recurse | Where-Object {
        $_.Name -match '(?i)\.(nii\.gz|nii|hdr|nrrd)$'
    })
    if (@($inputFiles | Where-Object { $_.Name -match '(?i)label' }).Count -gt 0) {
        $inputFiles = @($inputFiles | Where-Object { $_.Name -match '(?i)label' })
    }
    $expectedInputs = $inputFiles.Count
    $inputNames = @($inputFiles | ForEach-Object { $_.Name })
}
if ($expectedInputs -lt 1) {
    throw "No supported input volumes found in $InputDir"
}
$oppositeSide = if ($Side -eq 'left') { 'right|rh' } else { 'left|lh' }
$wrongSideNames = @($inputNames | Where-Object {
    $_ -match "(?i)(?:^|[_-])(?:$oppositeSide)(?:[_-]|\.)"
})
if ($wrongSideNames.Count -gt 0) {
    throw "Input side does not match -Side $Side. Example opposite-side file: $($wrongSideNames[0])"
}

$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
if ((Test-Path -LiteralPath $OutputDir) -and
    (Get-ChildItem -Force -LiteralPath $OutputDir | Select-Object -First 1)) {
    throw "Output directory is not empty. Use a fresh directory: $OutputDir"
}
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

$invariantCulture = [System.Globalization.CultureInfo]::InvariantCulture
$spacing = ([double]$contract.output_spacing).ToString($invariantCulture)
$arguments = @(
    '--no-main-window', '--no-splash', '--python-script', $icpScript,
    '--output_dir', $OutputDir,
    '--reference_template', $referenceTemplate,
    '--output_voxels', [string]$contract.output_voxels,
    '--output_spacing', $spacing,
    '--max_iterations', '20', '--tolerance', '0.00005',
    '--pairwise_iterations', '100', '--pairwise_tolerance', '0.0001',
    '--pairwise_landmarks', '200', '--interpolation', 'nn', '--invert_transform', 'auto'
)
if ($InputFile) {
    $arguments += @('--input_file', (Resolve-Path -LiteralPath $InputFile).Path)
    $inputMode = 'single'
} else {
    $arguments += @('--input_dir', (Resolve-Path -LiteralPath $InputDir).Path)
    $inputMode = 'batch'
}

Write-Host "ICP mode: fixed legacy reference ($Side; $inputMode; $expectedInputs input(s))"
Write-Host "Reference: $referenceTemplate"
Write-Host "Reference SHA-256: $actualReferenceHash"
Write-Host "Output: $OutputDir"
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

$statusPath = Join-Path $OutputDir 'icp_status.json'
if (-not (Test-Path -LiteralPath $statusPath -PathType Leaf)) {
    throw "ICP did not write status. Slicer exit code: $processExit. Check $OutputDir\icp_debug_log.txt"
}
$status = Get-Content -Raw -Encoding UTF8 -LiteralPath $statusPath | ConvertFrom-Json
if ($status.success -ne $true -or $status.mode -ne 'fixed_reference') {
    throw "ICP status is not successful fixed-reference mode: $($status | ConvertTo-Json -Compress)"
}
if ($status.reference_sha256 -ne $actualReferenceHash -or
    $status.geometry_version -ne 'fixed-legacy-groupwise-reference-v1' -or
    $status.independent_test_reference -ne $false) {
    throw 'ICP output provenance does not match the selected reference bundle.'
}
$alignedDir = Join-Path $OutputDir 'aligned_nifti'
$alignedCount = @(Get-ChildItem -LiteralPath $alignedDir -File | Where-Object {
    $_.Name -match '(?i)\.nii(\.gz)?$'
}).Count
if ([int]$status.subjects -ne $expectedInputs -or $alignedCount -ne $expectedInputs) {
    throw "Output count mismatch: input=$expectedInputs, status=$($status.subjects), aligned=$alignedCount"
}
if ($processExit -ne 0) {
    throw "Slicer returned exit code $processExit even though the output status was written. Review the ICP log."
}

$summary = [pscustomobject]@{
    Schema = 'fixed_reference_icp_run_v1'
    CreatedAtUtc = (Get-Date).ToUniversalTime().ToString('o')
    Side = $Side
    InputMode = $inputMode
    InputPath = if ($InputFile) { (Resolve-Path -LiteralPath $InputFile).Path } else { (Resolve-Path -LiteralPath $InputDir).Path }
    Subjects = $alignedCount
    Reference = (Resolve-Path -LiteralPath $referenceTemplate).Path
    ReferenceSHA256 = $actualReferenceHash
    ReferenceVersion = $contract.version
    IndependentTestReference = $false
    Output = $OutputDir
    IcpParameters = $status.parameters
}
$summary | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $OutputDir 'run_summary.json')
Write-Host "Verified $alignedCount aligned result(s). Run summary: $(Join-Path $OutputDir 'run_summary.json')"
