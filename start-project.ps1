param(
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $repoRoot '.venv\Scripts\python.exe'
$backendDir = Join-Path $repoRoot 'backend'
$frontendDir = Join-Path $repoRoot 'frontend'

if (-not (Test-Path $pythonExe)) {
    throw "Virtual environment not found at $pythonExe"
}

function Test-TcpPort {
    param(
        [string]$HostName,
        [int]$Port
    )

    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $async = $client.BeginConnect($HostName, $Port, $null, $null)
        if (-not $async.AsyncWaitHandle.WaitOne(1000, $false)) {
            return $false
        }
        $client.EndConnect($async)
        return $true
    }
    catch {
        return $false
    }
    finally {
        $client.Close()
    }
}

Write-Host 'Starting SecureML Ops...' -ForegroundColor Cyan

if (Test-TcpPort -HostName '127.0.0.1' -Port 8000) {
    Write-Host 'Backend port 8000 is already in use. Skipping backend launch.' -ForegroundColor Yellow
}
else {
    Start-Process -FilePath 'powershell.exe' -WorkingDirectory $repoRoot -ArgumentList @(
        '-NoExit',
        '-Command',
        "Set-Location '$repoRoot'; & '$pythonExe' -m uvicorn backend.main:app --reload --port 8000"
    )
}

if (Test-TcpPort -HostName '127.0.0.1' -Port 6379) {
    Start-Process -FilePath 'powershell.exe' -WorkingDirectory $repoRoot -ArgumentList @(
        '-NoExit',
        '-Command',
        "Set-Location '$repoRoot'; & '$pythonExe' -m celery -A backend.worker:celery_app worker --loglevel=info --pool=solo"
    )
}
else {
    Write-Host 'Redis is not listening on 127.0.0.1:6379. Celery worker was not started.' -ForegroundColor Yellow
    Write-Host 'Install Memurai or Redis locally, then rerun this script to enable attack jobs.' -ForegroundColor Yellow
}

if (Test-TcpPort -HostName '127.0.0.1' -Port 5173) {
    Write-Host 'Frontend port 5173 is already in use. Skipping frontend launch.' -ForegroundColor Yellow
}
else {
    Start-Process -FilePath 'powershell.exe' -WorkingDirectory $frontendDir -ArgumentList @(
        '-NoExit',
        '-Command',
        "Set-Location '$frontendDir'; npm run dev"
    )
}

if (-not $NoBrowser) {
    Start-Sleep -Seconds 2
    Start-Process 'http://127.0.0.1:5173'
}

Write-Host 'Launcher started backend and frontend. Check the new terminals for logs.' -ForegroundColor Green
