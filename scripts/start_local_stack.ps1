param(
    [ValidateSet('balanced', 'high')]
    [string]$FurnitureDetail = 'high',
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$workerUrl = 'http://127.0.0.1:8001'
$workerHealth = $null
try {
    $workerHealth = Invoke-RestMethod "$workerUrl/health" -TimeoutSec 3
} catch {
    # An absent worker is started below.
}
if ($workerHealth -and $workerHealth.engine -ne 'TripoSR Quality') {
    throw 'Port 8001 is occupied by another service.'
}
if (!$workerHealth) {
    if (Get-NetTCPConnection -LocalPort 8001 -State Listen -ErrorAction SilentlyContinue) {
        throw 'Port 8001 is occupied. Inspect that process before starting the furniture worker.'
    }
    $python = Join-Path $projectRoot '.venv-furniture-quality/Scripts/python.exe'
    if (!(Test-Path -LiteralPath $python)) {
        throw 'Set up .venv-furniture-quality as described in docs/furniture-quality-experiment.md.'
    }
    $logDir = Join-Path $projectRoot 'outputs/furniture_worker'
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $previousEnvironment = @{
        ARCHIGEN_FURNITURE_DETAIL = $env:ARCHIGEN_FURNITURE_DETAIL
        OMP_NUM_THREADS = $env:OMP_NUM_THREADS
        MKL_NUM_THREADS = $env:MKL_NUM_THREADS
        HF_HOME = $env:HF_HOME
        U2NET_HOME = $env:U2NET_HOME
    }
    try {
        $env:ARCHIGEN_FURNITURE_DETAIL = $FurnitureDetail
        $env:OMP_NUM_THREADS = '2'
        $env:MKL_NUM_THREADS = '2'
        $env:HF_HOME = Join-Path $projectRoot 'outputs/runtime/huggingface'
        $env:U2NET_HOME = Join-Path $projectRoot 'outputs/runtime/rembg'
        $processOptions = @{
            FilePath = $python
            WorkingDirectory = $projectRoot
            WindowStyle = 'Hidden'
            PassThru = $true
            ArgumentList = @('-m', 'uvicorn', 'backend.app.furniture_worker:app', '--host', '127.0.0.1', '--port', '8001')
            RedirectStandardOutput = Join-Path $logDir 'stdout.log'
            RedirectStandardError = Join-Path $logDir 'stderr.log'
        }
        $worker = Start-Process @processOptions
    } finally {
        foreach ($name in $previousEnvironment.Keys) {
            [Environment]::SetEnvironmentVariable($name, $previousEnvironment[$name], 'Process')
        }
    }
    $deadline = (Get-Date).AddMinutes(3)
    do {
        if ($worker.HasExited) {
            throw "GPU worker failed. Read $logDir/stderr.log."
        }
        Start-Sleep -Seconds 2
        try {
            $workerHealth = Invoke-RestMethod "$workerUrl/health" -TimeoutSec 3
        } catch {
            $workerHealth = $null
        }
    } until ($workerHealth -or (Get-Date) -ge $deadline)
    if (!$workerHealth) {
        throw "GPU worker startup timed out. Read $logDir/stderr.log."
    }
}

Push-Location $projectRoot
try {
    $arguments = @('compose', 'up', '-d', '--wait', '--wait-timeout', '180')
    if (!$SkipBuild) { $arguments += '--build' }
    & docker @arguments
    if ($LASTEXITCODE -ne 0) { throw 'Docker Compose startup failed.' }
    Write-Host "ArchiGen is ready at http://127.0.0.1:3000 with $($workerHealth.device)."
} finally {
    Pop-Location
}
