$ErrorActionPreference = 'SilentlyContinue'

$patterns = @(
    'uvicorn backend.main:app',
    'celery -A backend.worker:celery_app',
    'vite'
)

Get-CimInstance Win32_Process | Where-Object {
    $cmd = $_.CommandLine
    if ([string]::IsNullOrWhiteSpace($cmd)) {
        return $false
    }

    foreach ($pattern in $patterns) {
        if ($cmd -like "*$pattern*") {
            return $true
        }
    }

    return $false
} | ForEach-Object {
    Stop-Process -Id $_.ProcessId -Force
}

Write-Host 'Stopped SecureML Ops processes if they were running.' -ForegroundColor Green
