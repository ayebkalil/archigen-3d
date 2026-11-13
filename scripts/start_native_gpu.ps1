param(
    [ValidateSet('balanced', 'high')][string]$FurnitureDetail = 'high',
    [int]$StartupTimeoutSeconds = 300
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$logDir = Join-Path $projectRoot 'outputs/native_gpu'
$gpuPython = [IO.Path]::GetFullPath((Join-Path $projectRoot '.venv-furniture-quality/Scripts/python.exe'))
$apiPython = [IO.Path]::GetFullPath((Join-Path $projectRoot '.venv/Scripts/python.exe'))
foreach ($python in @($gpuPython, $apiPython)) {
    if (!(Test-Path -LiteralPath $python)) { throw "Missing $python. See docs/furniture-quality-experiment.md." }
}
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Get-Listener([int]$Port) {
    Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
}
function Test-ProjectPython($Process, [string]$Python, [string]$ModulePattern) {
    if ($Process.CommandLine -notmatch $ModulePattern) { return $false }
    if ($Process.CommandLine -match [regex]::Escape($Python)) { return $true }
    # Windows venv launchers may spawn the base interpreter with a rewritten argv.
    $parent = Get-CimInstance Win32_Process -Filter "ProcessId=$($Process.ParentProcessId)"
    return ($parent.CommandLine -match [regex]::Escape($Python) -and $parent.CommandLine -match $ModulePattern)
}
function Wait-Ready([string]$Url, $Process, [string]$LogName) {
    $deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
    do {
        $Process.Refresh()
        if ($Process.HasExited) { throw "$LogName exited. Read $logDir/$LogName.stderr.log" }
        try { return Invoke-RestMethod $Url -TimeoutSec 3 } catch { }
        Start-Sleep -Seconds 2
    } until ((Get-Date) -ge $deadline)
    throw "Startup timed out: $Url. Read $logDir/$LogName.stderr.log"
}
function Start-ServiceProcess([string]$Executable, [string[]]$Arguments, [string]$Directory, [string]$Name) {
    Start-Process -FilePath $Executable -ArgumentList $Arguments -WorkingDirectory $Directory -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $logDir "$Name.stdout.log") `
        -RedirectStandardError (Join-Path $logDir "$Name.stderr.log")
}

# Only restart a backend whose executable and command identify this checkout.
$apiListener = Get-Listener 8000
if ($apiListener) {
    $oldApi = Get-CimInstance Win32_Process -Filter "ProcessId=$($apiListener.OwningProcess)"
    if (!(Test-ProjectPython $oldApi $apiPython 'uvicorn\s+(?:backend\.)?app\.main:app')) {
        throw 'Port 8000 belongs to another service. Leave it running and free the project port manually.'
    }
}
$workerListener = Get-Listener 8001
if ($workerListener) {
    $workerHealth = Invoke-RestMethod 'http://127.0.0.1:8001/health' -TimeoutSec 3
    if ($workerHealth.engine -ne 'TripoSR Quality' -or $workerHealth.device -notmatch 'cuda') {
        throw 'Port 8001 does not belong to a healthy CUDA furniture worker.'
    }
    $oldWorker = Get-CimInstance Win32_Process -Filter "ProcessId=$($workerListener.OwningProcess)"
    if (!(Test-ProjectPython $oldWorker $gpuPython 'uvicorn\s+backend\.app\.furniture_worker:app')) {
        throw 'Port 8001 does not belong to this checkout.'
    }
}
$webListener = Get-Listener 3000
if ($webListener) {
    $web = Invoke-WebRequest 'http://127.0.0.1:3000' -UseBasicParsing -TimeoutSec 5
    if ($web.Content -notmatch '(?i)ArchiGen' -or $web.Content -notmatch '/@vite/client') {
        throw 'Port 3000 does not belong to the ArchiGen Vite frontend.'
    }
} elseif (!(Test-Path (Join-Path $projectRoot 'frontend/node_modules/vite/bin/vite.js'))) {
    throw 'Install the frontend first: cd frontend; npm ci'
}

$names = @('ARCHIGEN_FURNITURE_DETAIL', 'ARCHIGEN_FURNITURE_SERVICE_URL', 'ARCHIGEN_FURNITURE_QUALITY',
    'HF_HOME', 'U2NET_HOME', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTHONUNBUFFERED',
    'MLFLOW_TRACKING_URI', 'ARCHIGEN_MLFLOW_ENABLED', 'MLFLOW_HTTP_REQUEST_TIMEOUT', 'MLFLOW_HTTP_REQUEST_MAX_RETRIES')
$previous = @{}
foreach ($name in $names) { $previous[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
try {
    if ($env:ARCHIGEN_MLFLOW_ENABLED -ne '0') {
        & (Join-Path $PSScriptRoot 'start_mlflow.ps1')
        $env:ARCHIGEN_MLFLOW_ENABLED = '1'
        $env:MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'
        $env:MLFLOW_HTTP_REQUEST_TIMEOUT = '3'
        $env:MLFLOW_HTTP_REQUEST_MAX_RETRIES = '0'
    }
    $env:ARCHIGEN_FURNITURE_DETAIL = $FurnitureDetail
    $env:HF_HOME = Join-Path $projectRoot 'outputs/runtime/huggingface'
    $env:U2NET_HOME = Join-Path $projectRoot 'outputs/runtime/rembg'
    $env:OMP_NUM_THREADS = '2'
    $env:MKL_NUM_THREADS = '2'
    $env:PYTHONUNBUFFERED = '1'
    $env:ARCHIGEN_FURNITURE_SERVICE_URL = $null
    $env:ARCHIGEN_FURNITURE_QUALITY = $null
    if ($workerListener) {
        & taskkill /PID $workerListener.OwningProcess /F | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'Could not stop the previous project worker.' }
    }
    $worker = Start-ServiceProcess $gpuPython @('-m', 'uvicorn', 'backend.app.furniture_worker:app', '--host', '127.0.0.1', '--port', '8001') $projectRoot 'worker'
    $health = Wait-Ready 'http://127.0.0.1:8001/health' $worker 'worker'
    if ($health.device -notmatch 'cuda') { throw 'The worker did not initialize on CUDA.' }
    if ($apiListener) {
        & taskkill /PID $apiListener.OwningProcess /F | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'Could not stop the previous project backend.' }
    }
    $env:ARCHIGEN_FURNITURE_SERVICE_URL = 'http://127.0.0.1:8001'
    $api = Start-ServiceProcess $apiPython @('-m', 'uvicorn', 'app.main:app', '--app-dir', 'backend', '--host', '127.0.0.1', '--port', '8000') $projectRoot 'backend'
    $null = Wait-Ready 'http://127.0.0.1:8000/openapi.json' $api 'backend'
    if (!$webListener) {
        $node = (Get-Command node -ErrorAction Stop).Source
        $frontend = Start-ServiceProcess $node @('node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', '3000', '--strictPort') (Join-Path $projectRoot 'frontend') 'frontend'
        $null = Wait-Ready 'http://127.0.0.1:3000' $frontend 'frontend'
    }
    @{worker = $worker.Id; backend = $api.Id; frontend = $(if ($webListener) {$webListener.OwningProcess} else {$frontend.Id}); detail = $FurnitureDetail} |
        ConvertTo-Json | Set-Content (Join-Path $logDir 'processes.json')
    Write-Host "ArchiGen ready: http://127.0.0.1:3000/#studio | $($health.device) | $FurnitureDetail"
    Write-Host "Logs: $logDir"
} finally {
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $previous[$name], 'Process') }
}
