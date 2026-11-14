param(
    [switch]$Gpu = $true,
    [switch]$SkipBuild,
    [switch]$ReplaceNative,
    [int]$WaitTimeoutSeconds = 900
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$files = @('-f', 'docker-compose.yml')
if ($Gpu) { $files += @('-f', 'docker-compose.gpu.yml') }
$nativePython = Join-Path $projectRoot '.venv/Scripts/python.exe'
$transition = Join-Path $PSScriptRoot 'docker_native_transition.py'
$nativeStopped = $false
$started = $false
$previousRevision = $env:ARCHIGEN_REVISION
$logDir = Join-Path $projectRoot 'outputs/docker'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Invoke-Compose([string[]]$Arguments) {
    & docker compose @files @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Docker Compose failed: $Arguments" }
}
function Invoke-Native([string]$Command) {
    & $nativePython $transition $Command
    if ($LASTEXITCODE -ne 0) { throw "Native service transition failed: $Command" }
}

Push-Location $projectRoot
try {
    & docker info --format '{{.ServerVersion}}' *> $null
    if ($LASTEXITCODE -ne 0) { throw 'Start Docker Desktop with the WSL2 backend first.' }
    if (Get-Command git -ErrorAction SilentlyContinue) {
        $revision = & git rev-parse --short HEAD 2>$null
        if ($revision) { $env:ARCHIGEN_REVISION = $revision }
    }
    Invoke-Compose -Arguments @('config', '--quiet')
    # Build and probe CUDA before interrupting native services.
    if (!$SkipBuild) { Invoke-Compose -Arguments @('build') }
    if ($Gpu) {
        Invoke-Compose -Arguments @('run', '--rm', '--no-deps', 'furniture', 'python', '-c',
            'import torch; assert torch.cuda.is_available(); x=torch.ones(32,32).cuda(); print(torch.cuda.get_device_name(0), (x@x).sum().item())')
    }
    # Compose resolves both process variables and the optional .env file.
    $configuration = (& docker compose @files config --format json) | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve Compose configuration.' }
    $ports = @{}
    foreach ($role in @('frontend', 'backend', 'mlflow')) {
        $port = [int]$configuration.services.$role.ports[0].published
        if ($ports.ContainsKey($port)) { throw "Duplicate host port: $port" }
        $ports[$port] = "archigen_$role"
    }
    $published = & docker ps --format '{{.Names}} {{.Ports}}'
    $occupied = @()
    foreach ($port in $ports.Keys) {
        if (!(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)) { continue }
        if (!($published | Where-Object { $_ -match "^$($ports[$port]) " -and $_ -match ":$port->" })) { $occupied += $port }
    }
    if ($occupied.Count -or $ReplaceNative) {
        if (!$ReplaceNative) { throw "Ports $occupied are occupied. Use -ReplaceNative only for this project's native stack." }
        if (!(Test-Path -LiteralPath $nativePython)) { throw 'Native migration requires the existing .venv; a fresh Docker install does not.' }
        if ($ports[3000] -ne 'archigen_frontend' -or $ports[8000] -ne 'archigen_backend' -or $ports[5000] -ne 'archigen_mlflow') {
            throw '-ReplaceNative requires the original ports 3000, 8000 and 5000.'
        }
        Invoke-Native 'inspect'
        Invoke-Native 'suspend'
        $nativeStopped = $true
    }
    $database = ($configuration.services.mlflow.volumes | Where-Object target -eq '/mlflow/mlflow.db').source
    $artifacts = ($configuration.services.mlflow.volumes | Where-Object target -eq '/mlflow/artifacts').source
    if (!(Test-Path -LiteralPath $database)) { New-Item -ItemType File -Path $database | Out-Null }
    New-Item -ItemType Directory -Force -Path $artifacts | Out-Null
    Invoke-Compose -Arguments @('run', '--rm', '--no-deps', '--entrypoint', 'python', 'mlflow', '/app/scripts/docker_data.py', 'backup')
    Invoke-Compose -Arguments @('run', '--rm', '--no-deps', 'model-init')
    $started = $true
    Invoke-Compose -Arguments @('up', '-d', '--wait', '--wait-timeout', "$WaitTimeoutSeconds")
    $web = ($ports.Keys | Where-Object { $ports[$_] -eq 'archigen_frontend' })
    $mlflow = ($ports.Keys | Where-Object { $ports[$_] -eq 'archigen_mlflow' })
    Write-Host "ArchiGen: http://localhost:$web/#studio"
    Write-Host "MLflow: http://localhost:$mlflow"
    Write-Host 'Logs: docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f'
    if (!$Gpu) { Write-Warning 'Without the GPU worker, furniture reconstruction is unavailable.' }
} catch {
    $failure = $_
    & docker compose @files logs --no-color --tail 100 *> (Join-Path $logDir 'startup-failure.log')
    if ($nativeStopped) {
        if ($started) { & docker compose @files down }
        Invoke-Native 'restore'
    }
    throw $failure
} finally {
    $env:ARCHIGEN_REVISION = $previousRevision
    Pop-Location
}
