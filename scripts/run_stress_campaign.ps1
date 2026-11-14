param([ValidateRange(1,480)][int]$BudgetMinutes = 30,
      [string]$Manifest = 'configs/stress_campaign.json', [switch]$ComponentsOnly)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$python = Join-Path $projectRoot '.venv-furniture-quality/Scripts/python.exe'
$previousOwnerPid = $env:ARCHIGEN_STRESS_OWNER_PID
$env:ARCHIGEN_STRESS_OWNER_PID = "$PID"
Push-Location $projectRoot
try {
    & (Join-Path $PSScriptRoot 'start_mlflow.ps1')
    try {
        & $python -m src.stress.manage_stack suspend
        if ($LASTEXITCODE -ne 0) { throw 'Could not safely suspend the project services; no campaign started.' }
        $extraArgs = @()
        if ($ComponentsOnly) { $extraArgs += '--components-only' }
        & $python -m src.stress.campaign --budget-minutes $BudgetMinutes --manifest $Manifest @extraArgs
        $campaignExit = $LASTEXITCODE
    } finally {
        # Restore only the services recorded by the ownership-checked suspension.
        $statePath = Join-Path $projectRoot 'outputs/stress/stack-state.json'
        if (Test-Path -LiteralPath $statePath) {
            $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
            if ($state.owner_pid -eq $PID -and $state.suspended) {
                & $python -m src.stress.manage_stack restore
                if ($LASTEXITCODE -ne 0) { throw 'Restoration failed; see outputs/stress/stack-state.json. Run scripts/start_native_gpu.ps1.' }
            }
        }
    }
    if ($campaignExit -ne 0) { throw "Campaign ended with exit code $campaignExit; partial results retained." }
} finally { $env:ARCHIGEN_STRESS_OWNER_PID = $previousOwnerPid; Pop-Location }
