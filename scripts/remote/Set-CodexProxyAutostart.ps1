param(
    [switch]$Remove,
    [ValidatePattern('^[a-zA-Z0-9_.@-]+$')][string]$Server = 'dr@10.137.11.194',
    [ValidateRange(1,65535)][int]$LocalProxyPort = 7890,
    [ValidateRange(1024,65535)][int]$RemoteProxyPort = 17890
)
$ErrorActionPreference = 'Stop'
$taskName = 'Codex-Laptop-Proxy'
function Stop-ProxyTask {
    $existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if (-not $existing) { return }
    $owned = @()
    $payload = [regex]::Match([string]$existing.Actions[0].Arguments, '[A-Za-z0-9+/]{120,}={0,2}').Value
    if ($payload) {
        $processes = @(Get-CimInstance Win32_Process)
        $matched = @($processes | Where-Object {
            $_.Name -in @('powershell.exe', 'pythonw.exe') -and $_.CommandLine -and $_.CommandLine.Contains($payload)
        })
        $frontier = @($matched | Where-Object { $_.ParentProcessId -notin @($matched.ProcessId) })
        while ($frontier.Count) {
            $owned += $frontier
            $parents = @($frontier.ProcessId)
            $frontier = @($processes | Where-Object { $_.ParentProcessId -in $parents -and $_.ProcessId -notin @($owned.ProcessId) })
        }
    }
    Stop-ScheduledTask -TaskName $taskName
    # Stopping a scheduled action can leave native children alive. Clean only this
    # action's captured descendants, checking creation time to reject reused PIDs.
    [array]::Reverse($owned)
    foreach ($captured in $owned) {
        $current = Get-CimInstance Win32_Process -Filter "ProcessId=$($captured.ProcessId)"
        if ($current -and $current.CreationDate -eq $captured.CreationDate) {
            Stop-Process -Id $current.ProcessId -Force -ErrorAction SilentlyContinue
        }
    }
    $deadline = (Get-Date).AddSeconds(20)
    while ((Get-ScheduledTask -TaskName $taskName).State -eq 'Running') {
        if ((Get-Date) -ge $deadline) { throw 'Previous tunnel task did not stop.' }
        Start-Sleep -Milliseconds 250
    }
}
if ($Remove) {
    Stop-ProxyTask
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host 'Autostart and its tunnel processes removed. Independent manual tunnels must be stopped separately.'
    exit
}
$source = Join-Path $PSScriptRoot 'Start-CodexProxy.ps1'
$pythonw = (Get-Command pythonw.exe -ErrorAction Stop).Source
Stop-ProxyTask
# Store the script in the task itself. Task Scheduler may see a different filesystem
# view from the installing app, and therefore cannot always read its copied files.
$scriptBody = Get-Content -Raw -LiteralPath $source
$command = "& {`n$scriptBody`n} -Server $Server -LocalProxyPort $LocalProxyPort -RemoteProxyPort $RemoteProxyPort"
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
$powershellPath = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$pathEncoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($powershellPath))
# pythonw is a GUI executable: it has no console for Windows Terminal to display.
# Explicitly suppress console creation for its PowerShell child as well.
$pythonCode = "import base64,subprocess,sys;sys.exit(subprocess.call([base64.b64decode('$pathEncoded').decode('utf-8'),'-NoProfile','-NonInteractive','-WindowStyle','Hidden','-ExecutionPolicy','Bypass','-EncodedCommand','$encoded'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW))"
$arguments = "-c `"$pythonCode`""
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction -Execute $pythonw -Argument $arguments
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $identity
$principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Write-Host "Installed and started $taskName for $identity."
