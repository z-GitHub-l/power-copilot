$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
$logsDir = Join-Path $projectRoot ".demo-logs"

$backendHost = "127.0.0.1"
$backendPort = 8000
$frontendPort = 8501
$backendUrl = "http://$backendHost`:$backendPort"
$frontendUrl = "http://$backendHost`:$frontendPort"

if (-not (Test-Path $pythonExe)) {
    throw "Virtual environment not found. Run: python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
}

if (-not (Test-Path (Join-Path $projectRoot ".env"))) {
    Copy-Item (Join-Path $projectRoot ".env.example") (Join-Path $projectRoot ".env")
}

New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

# Some shells expose duplicate Path/PATH entries. Windows PowerShell 5.1 can
# fail when Start-Process enumerates that environment, so keep one canonical key.
$pathValue = [Environment]::GetEnvironmentVariable("Path", "Process")
if ($pathValue) {
    [Environment]::SetEnvironmentVariable("PATH", $null, "Process")
    [Environment]::SetEnvironmentVariable("Path", $pathValue, "Process")
}

$backendOutLog = Join-Path $logsDir "backend.out.log"
$backendErrLog = Join-Path $logsDir "backend.err.log"
$frontendOutLog = Join-Path $logsDir "frontend.out.log"
$frontendErrLog = Join-Path $logsDir "frontend.err.log"

function Test-TcpPort {
    param(
        [string]$HostName,
        [int]$Port,
        [int]$TimeoutMs = 800
    )

    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $asyncResult = $client.BeginConnect($HostName, $Port, $null, $null)
        if (-not $asyncResult.AsyncWaitHandle.WaitOne($TimeoutMs, $false)) {
            return $false
        }

        $client.EndConnect($asyncResult)
        return $true
    }
    catch {
        return $false
    }
    finally {
        $client.Close()
    }
}

function Test-HttpOk {
    param([string]$Url)

    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
        return ([int]$response.StatusCode -ge 200 -and [int]$response.StatusCode -lt 400)
    }
    catch {
        return $false
    }
}

function Wait-HttpReady {
    param(
        [string]$Name,
        [string]$Url,
        [string]$ErrorLog
    )

    for ($attempt = 1; $attempt -le 30; $attempt++) {
        if (Test-HttpOk $Url) {
            return
        }
        Start-Sleep -Milliseconds 500
    }

    throw "$Name did not become ready at $Url. Check log: $ErrorLog"
}

function Start-DemoProcess {
    param(
        [string]$Name,
        [array]$Arguments,
        [string]$ReadyUrl,
        [int]$Port,
        [string]$OutLog,
        [string]$ErrLog
    )

    if (Test-HttpOk $ReadyUrl) {
        Write-Host "$Name already running: $ReadyUrl"
        return $null
    }

    if (Test-TcpPort -HostName $backendHost -Port $Port) {
        throw "$Name port $Port is already in use, but $ReadyUrl is not responding. Stop the conflicting process or change the port."
    }

    $process = Start-Process `
        -FilePath $pythonExe `
        -ArgumentList $Arguments `
        -WorkingDirectory $projectRoot `
        -RedirectStandardOutput $OutLog `
        -RedirectStandardError $ErrLog `
        -WindowStyle Hidden `
        -PassThru

    Wait-HttpReady -Name $Name -Url $ReadyUrl -ErrorLog $ErrLog
    Write-Host "$Name started: $ReadyUrl"
    Write-Host "$Name PID: $($process.Id)"
    return $process
}

$backendArgs = @(
    "-m",
    "uvicorn",
    "app.main:app",
    "--host",
    $backendHost,
    "--port",
    "$backendPort"
)

$frontendArgs = @(
    "-m",
    "streamlit",
    "run",
    "frontend/streamlit_app.py",
    "--server.headless",
    "true",
    "--server.port",
    "$frontendPort"
)

Start-DemoProcess `
    -Name "Backend" `
    -Arguments $backendArgs `
    -ReadyUrl "$backendUrl/health" `
    -Port $backendPort `
    -OutLog $backendOutLog `
    -ErrLog $backendErrLog | Out-Null

Start-DemoProcess `
    -Name "Frontend" `
    -Arguments $frontendArgs `
    -ReadyUrl $frontendUrl `
    -Port $frontendPort `
    -OutLog $frontendOutLog `
    -ErrLog $frontendErrLog | Out-Null

Write-Host ""
Write-Host "Demo is ready:"
Write-Host "  Backend:  $backendUrl"
Write-Host "  API docs: $backendUrl/docs"
Write-Host "  Frontend: $frontendUrl"
Write-Host "Logs:"
Write-Host "  $backendOutLog"
Write-Host "  $backendErrLog"
Write-Host "  $frontendOutLog"
Write-Host "  $frontendErrLog"
