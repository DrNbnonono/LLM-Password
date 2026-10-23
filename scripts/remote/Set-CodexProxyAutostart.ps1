param(
    [switch]$Remove,
    [ValidatePattern('^[a-zA-Z0-9_.@-]+$')][string]$Server = 'dr@10.137.11.194',
    [ValidateRange(1,65535)][int]$LocalProxyPort = 7890,
    [ValidateRange(1024,65535)][int]$RemoteProxyPort = 17890
)
$ErrorActionPreference = 'Stop'
$taskName = 'Codex-Laptop-Proxy'
if ($Remove) {
    Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host 'Autostart removed. Existing foreground tunnels must be stopped separately.'
    exit
}
$source = Join-Path $PSScriptRoot 'Start-CodexProxy.ps1'
$existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existingTask) {
    Stop-ScheduledTask -TaskName $taskName
    $deadline = (Get-Date).AddSeconds(20)
    while ((Get-ScheduledTask -TaskName $taskName).State -eq 'Running') {
        if ((Get-Date) -ge $deadline) { throw 'Previous tunnel task did not stop.' }
        Start-Sleep -Milliseconds 250
    }
}
# Store the script in the task itself. Task Scheduler may see a different filesystem
# view from the installing app, and therefore cannot always read its copied files.
$scriptBody = Get-Content -Raw -LiteralPath $source
$command = "& {`n$scriptBody`n} -Server $Server -LocalProxyPort $LocalProxyPort -RemoteProxyPort $RemoteProxyPort"
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
$arguments = "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -EncodedCommand $encoded"
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -Argument $arguments
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $identity
$principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Write-Host "Installed and started $taskName for $identity."
