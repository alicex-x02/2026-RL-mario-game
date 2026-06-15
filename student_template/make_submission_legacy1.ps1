param(
    [Parameter(Mandatory=$true)]
    [string]$TeamId
)

$ErrorActionPreference = "Stop"
$here = if ($PSScriptRoot) {
    $PSScriptRoot
} else {
    Split-Path -Parent $MyInvocation.MyCommand.Path
}

$agentPath = Join-Path $here "agent.py"
$modelPath = Join-Path $here "model.pt"
$trainPath = Join-Path $here "train.py"
$metaPath = Join-Path $here "submission_meta.json"

if (-not (Test-Path -LiteralPath $agentPath)) {
    throw "agent.py not found"
}
if (-not (Test-Path -LiteralPath $modelPath)) {
    throw "model.pt not found. Train a model before packaging."
}
if (-not (Test-Path -LiteralPath $trainPath)) {
    throw "train.py not found. Include the training script used to create model.pt."
}

$zip = Join-Path $here "${TeamId}_submission.zip"
if (Test-Path -LiteralPath $zip) {
    Remove-Item -LiteralPath $zip -Force
}

$files = New-Object System.Collections.Generic.List[string]
$files.Add($modelPath)

Get-ChildItem -LiteralPath $here -File -Filter "*.py" | ForEach-Object {
    if ($_.FullName) {
        $files.Add($_.FullName)
    }
}
if (Test-Path -LiteralPath $metaPath) {
    $files.Add($metaPath)
}

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::Open(
    $zip,
    [System.IO.Compression.ZipArchiveMode]::Create
)
try {
    foreach ($file in $files) {
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $archive,
            $file,
            [System.IO.Path]::GetFileName($file),
            [System.IO.Compression.CompressionLevel]::Optimal
        ) | Out-Null
    }
} finally {
    $archive.Dispose()
}
Write-Host "Created $zip"
