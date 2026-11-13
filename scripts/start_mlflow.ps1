param([switch]$Restart, [int]$StartupTimeoutSeconds = 120)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$python = [IO.Path]::GetFullPath((Join-Path $projectRoot '.venv/Scripts/python.exe'))
$logDir = Join-Path $projectRoot 'outputs/mlflow'
$uri = 'http://127.0.0.1:5000'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$listener = Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($listener) {
    $chain = @()
    $ancestorId = $listener.OwningProcess
    for ($i = 0; $i -lt 6 -and $ancestorId; $i++) {
        $ancestor = Get-CimInstance Win32_Process -Filter "ProcessId=$ancestorId"
        if (!$ancestor) { break }
        $chain += $ancestor
        $ancestorId = $ancestor.ParentProcessId
    }
    $owned = $chain | Where-Object {
        $_.CommandLine -match [regex]::Escape($python) -and $_.CommandLine -match '(?:-m\s+mlflow\s+server|mlflow\.server)'
    }
    if (!$owned) { throw 'Port 5000 is occupied by another service. It will not be stopped or reused.' }
    $null = Invoke-RestMethod "$uri/health" -TimeoutSec 5
    if (!$Restart) {
        Write-Host "MLflow already running: $uri"
        return
    }
    # Terminate the project CLI tree, so its HTTP worker is stopped too.
    $cli = $chain | Where-Object {
        $_.CommandLine -match [regex]::Escape($python) -and $_.CommandLine -match '-m\s+mlflow\s+server'
    } | Select-Object -Last 1
    if (!$cli) { throw 'Cannot identify the project MLflow CLI for restart.' }
    & taskkill /PID $cli.ProcessId /T /F | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Could not stop the project MLflow server.' }
}
if (!(Test-Path -LiteralPath $python)) { throw 'Create .venv and install requirements-tracking.txt first.' }
$database = [IO.Path]::GetFullPath((Join-Path $projectRoot 'mlflow.db')).Replace('\', '/')
$artifacts = [IO.Path]::GetFullPath((Join-Path $projectRoot 'mlartifacts'))
New-Item -ItemType Directory -Force -Path $artifacts | Out-Null
$artifactUri = ([Uri]$artifacts).AbsoluteUri
$arguments = @('-m', 'mlflow', 'server', '--backend-store-uri', "`"sqlite:///$database`"",
    '--serve-artifacts', '--artifacts-destination', "`"$artifactUri`"", '--default-artifact-root', 'mlflow-artifacts:/',
    '--host', '127.0.0.1', '--port', '5000', '--workers', '1')
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logDir 'stdout.log') -RedirectStandardError (Join-Path $logDir 'stderr.log')
$deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
do {
    $process.Refresh()
    if ($process.HasExited) { throw "MLflow exited. Read $logDir/stderr.log" }
    try {
        $null = Invoke-RestMethod "$uri/health" -TimeoutSec 3
        @{pid = $process.Id; tracking_uri = $uri; database = $database; artifacts = $artifacts} |
            ConvertTo-Json | Set-Content (Join-Path $logDir 'process.json')
        Write-Host "MLflow ready: $uri"
        return
    } catch { }
    Start-Sleep -Seconds 2
} until ((Get-Date) -ge $deadline)
throw "MLflow startup timed out. Read $logDir/stderr.log"
