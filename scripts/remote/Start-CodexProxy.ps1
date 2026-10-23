param(
    [ValidatePattern('^[a-zA-Z0-9_.@-]+$')][string]$Server = 'dr@10.137.11.194',
    [ValidateRange(1,65535)][int]$LocalProxyPort = 7890,
    [ValidateRange(1024,65535)][int]$RemoteProxyPort = 17890,
    [switch]$Once,
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$sshArgs = @('-N', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
    '-o', 'ExitOnForwardFailure=yes', '-o', 'ConnectTimeout=10',
    '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
    '-R', "127.0.0.1:${RemoteProxyPort}:127.0.0.1:${LocalProxyPort}", $Server)
if ($CheckOnly) { ConvertTo-Json -InputObject $sshArgs -Compress; exit 0 }
$mutex = New-Object System.Threading.Mutex($false, "Local\CodexLaptopProxy-$RemoteProxyPort")
if (-not $mutex.WaitOne(0)) { throw 'A tunnel supervisor already owns this port.' }
try {
    do {
        $client = New-Object System.Net.Sockets.TcpClient
        try {
            $pending = $client.BeginConnect('127.0.0.1', $LocalProxyPort, $null, $null)
            if (-not $pending.AsyncWaitHandle.WaitOne(2000)) { throw 'Proxy connect timeout' }
            $client.EndConnect($pending)
            $ready = $true
        } catch {
            $ready = $false
            Write-Host "$(Get-Date -Format s) Waiting for local proxy on port $LocalProxyPort."
        } finally { $client.Dispose() }
        if ($ready) {
            Write-Host "$(Get-Date -Format s) Connecting proxy tunnel to $Server."
            $startInfo = New-Object System.Diagnostics.ProcessStartInfo
            $startInfo.FileName = (Get-Command ssh.exe -ErrorAction Stop).Source
            # All arguments are fixed options, validated numeric ports or a validated host.
            $startInfo.Arguments = $sshArgs -join ' '
            $startInfo.UseShellExecute = $false
            $startInfo.CreateNoWindow = $true
            $sshProcess = New-Object System.Diagnostics.Process
            $sshProcess.StartInfo = $startInfo
            $sshStarted = $false
            try {
                $sshStarted = $sshProcess.Start()
                $sshProcess.WaitForExit()
                $tunnelExit = $sshProcess.ExitCode
            } finally {
                if ($sshStarted -and -not $sshProcess.HasExited) { $sshProcess.Kill() }
                $sshProcess.Dispose()
            }
            Write-Host "$(Get-Date -Format s) SSH exited: $tunnelExit"
            if ($Once) { exit $tunnelExit }
        } elseif ($Once) { exit 2 }
        if (-not $Once) { Start-Sleep -Seconds 15 }
    } while (-not $Once)
} finally { $mutex.ReleaseMutex(); $mutex.Dispose() }
