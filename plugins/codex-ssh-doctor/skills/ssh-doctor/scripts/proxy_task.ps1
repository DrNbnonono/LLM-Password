param(
    [ValidateSet('status','restart')][string]$Action = 'status',
    [ValidatePattern('^[a-zA-Z0-9_-]{1,100}$')][string]$TaskName = 'Codex-Laptop-Proxy',
    [switch]$Confirmed
)
$ErrorActionPreference = 'Stop'
$lookupErrors = @()
$task = Get-ScheduledTask -TaskName $TaskName -TaskPath '\' -ErrorAction SilentlyContinue -ErrorVariable lookupErrors
if (-not $task -and @($lookupErrors | Where-Object { $_.FullyQualifiedErrorId -notlike '*CmdletizationQuery_NotFound*' }).Count) {
    throw 'Scheduled task lookup failed; permissions or Task Scheduler service may be unavailable.'
}
if ($Action -eq 'status') {
    if ($task) { @{ exists=$true; state=[string]$task.State } | ConvertTo-Json -Compress }
    else { @{ exists=$false; state='missing' } | ConvertTo-Json -Compress }
    exit 0
}
if (-not $Confirmed) { throw 'Restart requires -Confirmed and an explicit user request.' }
if (-not $task) { throw 'Task does not exist. Restart never installs or rewrites it.' }
$payload = [regex]::Match([string]$task.Actions[0].Arguments, '[A-Za-z0-9+/]{120,}={0,2}').Value
if (-not $payload) { throw 'Task launcher is not recognized; refusing process cleanup.' }
$launcherName = [IO.Path]::GetFileName([string]$task.Actions[0].Execute)
if (@($task.Actions).Count -ne 1 -or $launcherName -notin @('pythonw.exe','powershell.exe')) {
    throw 'Only the known proxy launcher can be restarted.'
}
$body = [Text.Encoding]::Unicode.GetString([Convert]::FromBase64String($payload))
if (-not $body.Contains('CodexLaptopProxy-') -or -not $body.Contains('ExitOnForwardFailure=yes')) {
    throw 'Task payload is not the proxy supervisor.'
}
$service = New-Object -ComObject Schedule.Service
$service.Connect()
$instances = @($service.GetFolder('\').GetTask($TaskName).GetInstances(0))
if ($instances.Count -gt 1) { throw 'Multiple running instances; refusing ambiguous cleanup.' }
$processes = @(Get-CimInstance Win32_Process)
$owned = @()
$frontier = @()
if ($instances.Count) {
    $frontier = @($processes | Where-Object {
        $_.ProcessId -eq $instances[0].EnginePID -and $_.Name -eq $launcherName -and $_.CommandLine -and $_.CommandLine.Contains($payload)
    })
    if ($frontier.Count -ne 1) { throw 'Task instance identity could not be verified.' }
} elseif ($task.State -eq 'Running') {
    throw 'Running task instance is unavailable; refusing cleanup.'
}
while ($frontier.Count) {
    $owned += $frontier
    $parents = @($frontier.ProcessId)
    $frontier = @($processes | Where-Object { $_.ParentProcessId -in $parents -and $_.ProcessId -notin @($owned.ProcessId) })
}
Stop-ScheduledTask -TaskName $TaskName -TaskPath '\'
[array]::Reverse($owned)
foreach ($captured in $owned) {
    $current = Get-CimInstance Win32_Process -Filter "ProcessId=$($captured.ProcessId)"
    if ($current -and $current.CreationDate -eq $captured.CreationDate) {
        Stop-Process -Id $current.ProcessId -Force -ErrorAction SilentlyContinue
    }
}
$deadline = (Get-Date).AddSeconds(20)
while ((Get-ScheduledTask -TaskName $TaskName -TaskPath '\').State -eq 'Running') {
    if ((Get-Date) -ge $deadline) { throw 'Old task did not stop.' }
    Start-Sleep -Milliseconds 250
}
Start-ScheduledTask -TaskName $TaskName -TaskPath '\'
@{ restarted=$true; task=$TaskName } | ConvertTo-Json -Compress
