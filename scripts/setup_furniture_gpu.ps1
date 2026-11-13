param([string]$UvPath = '')

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Push-Location $projectRoot
try {
    if (!$UvPath) {
        $UvPath = Join-Path $projectRoot '.venv/Scripts/uv.exe'
        if (!(Test-Path -LiteralPath $UvPath)) {
            $command = Get-Command uv -ErrorAction SilentlyContinue
            if (!$command) { throw 'Install uv first: python -m pip install uv' }
            $UvPath = $command.Source
        }
    }
    function Invoke-Checked([string[]]$Arguments) {
        & $UvPath @Arguments
        if ($LASTEXITCODE -ne 0) { throw "uv failed: $Arguments" }
    }
    $cache = 'outputs/runtime/cache'
    Invoke-Checked @('python', 'install', '3.11', '--install-dir', 'outputs/runtime/python', '--no-bin', '--cache-dir', $cache)
    $runtime = Get-ChildItem outputs/runtime/python -Directory -Filter 'cpython-3.11*-windows-x86_64-none' |
        Sort-Object Name -Descending | Select-Object -First 1
    $python = Join-Path $projectRoot '.venv-furniture-quality/Scripts/python.exe'
    if (!(Test-Path -LiteralPath $python)) {
        Invoke-Checked @('venv', '.venv-furniture-quality', '--python', (Join-Path $runtime.FullName 'python.exe'), '--cache-dir', $cache)
    }
    & $python -c "import sys; assert sys.version_info[:2] == (3, 11), 'Use Python 3.11 for this environment'"
    if ($LASTEXITCODE -ne 0) { throw 'Incompatible furniture environment.' }
    Invoke-Checked @('pip', 'install', '--python', $python, 'torch==2.6.0', 'torchvision==0.21.0', '--index-url', 'https://download.pytorch.org/whl/cu124', '--cache-dir', $cache)
    Invoke-Checked @('pip', 'install', '--python', $python, '-r', 'requirements-furniture-quality.txt', '--cache-dir', $cache)
    $revision = '107cefdc244c39106fa830359024f6a2f1c78871'
    if (!(Test-Path TripoSR)) {
        & git init TripoSR
        if ($LASTEXITCODE -ne 0) { throw 'TripoSR initialization failed.' }
        & git -C TripoSR remote add origin https://github.com/VAST-AI-Research/TripoSR.git
        if ($LASTEXITCODE -ne 0) { throw 'TripoSR remote setup failed.' }
    }
    $head = & git -C TripoSR rev-parse --verify --quiet HEAD
    if ($head -ne $revision) {
        if (& git -C TripoSR status --porcelain) { throw 'TripoSR has local changes; preserve them before changing its revision.' }
        & git -C TripoSR config remote.origin.promisor true
        & git -C TripoSR config remote.origin.partialclonefilter blob:none
        & git -C TripoSR fetch --filter=blob:none --depth 1 origin $revision
        if ($LASTEXITCODE -ne 0) { throw 'TripoSR fetch failed.' }
        & git -C TripoSR sparse-checkout init --cone
        if ($LASTEXITCODE -ne 0) { throw 'TripoSR sparse checkout initialization failed.' }
        & git -C TripoSR sparse-checkout set tsr
        if ($LASTEXITCODE -ne 0) { throw 'TripoSR source selection failed.' }
        & git -C TripoSR checkout --detach $revision
        if ($LASTEXITCODE -ne 0) { throw 'TripoSR checkout failed.' }
    }
    $patched = Select-String -LiteralPath 'TripoSR/tsr/models/isosurface.py' -SimpleMatch 'torchmcubes is unavailable; use ArchiGen quality extraction'
    if (!$patched) {
        & git -C TripoSR apply --check --unidiff-zero --ignore-space-change ../patches/triposr-optional-marching-cubes.patch
        if ($LASTEXITCODE -ne 0) { throw 'The TripoSR import patch cannot be applied.' }
        & git -C TripoSR apply --unidiff-zero --ignore-space-change ../patches/triposr-optional-marching-cubes.patch
        if ($LASTEXITCODE -ne 0) { throw 'TripoSR patch failed.' }
    } else {
        & git -C TripoSR apply --reverse --check --unidiff-zero --ignore-space-change ../patches/triposr-optional-marching-cubes.patch
        if ($LASTEXITCODE -ne 0) { throw 'The TripoSR import patch cannot be verified.' }
    }
    & $python scripts/download_furniture_models.py
    if ($LASTEXITCODE -ne 0) { throw 'Model download or verification failed.' }
    & $python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; x=torch.ones(16, device='cuda'); print(torch.__version__, torch.cuda.get_device_name(0), (x @ x).item())"
    if ($LASTEXITCODE -ne 0) { throw 'CUDA verification failed.' }
    Write-Host 'GPU furniture environment ready. Run .\scripts\start_native_gpu.ps1'
} finally { Pop-Location }
