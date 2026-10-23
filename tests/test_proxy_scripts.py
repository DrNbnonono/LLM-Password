import json
import pathlib
import subprocess
import unittest
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/remote/Start-CodexProxy.ps1"

class TunnelTests(unittest.TestCase):
    def call_script(self, *args):
        return subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT), *args], capture_output=True, text=True, errors="replace")

    def test_preview_binds_only_loopback_and_has_keepalive(self):
        result = self.call_script("-CheckOnly")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(result.stdout)
        self.assertIn("127.0.0.1:17890:127.0.0.1:7890", args)
        self.assertIn("ExitOnForwardFailure=yes", args)
        self.assertIn("ServerAliveInterval=15", args)
        self.assertIn("BatchMode=yes", args)
        self.assertIn("StrictHostKeyChecking=yes", args)
        self.assertEqual(args[-1], "dr@10.137.11.194")

    def test_invalid_port_is_rejected(self):
        result = self.call_script("-CheckOnly", "-LocalProxyPort", "0")
        self.assertNotEqual(result.returncode, 0)

    def test_reinstall_stops_old_task_before_registering(self):
        with tempfile.TemporaryDirectory() as tmp:
            installer = str(ROOT / 'scripts/remote/Set-CodexProxyAutostart.ps1').replace("'", "''")
            cmd = r"""
$ErrorActionPreference='Stop'
$global:events=@()
$token='A'*140
$created=Get-Date '2026-10-23'
$global:fakeProcesses=@(
 [pscustomobject]@{ProcessId=1001;ParentProcessId=10;Name='pythonw.exe';CommandLine=$token;CreationDate=$created},
 [pscustomobject]@{ProcessId=1002;ParentProcessId=1001;Name='powershell.exe';CommandLine=$token;CreationDate=$created},
 [pscustomobject]@{ProcessId=1003;ParentProcessId=1002;Name='ssh.exe';CommandLine='ssh';CreationDate=$created}
)
function Get-ScheduledTask { [pscustomobject]@{State='Ready';Actions=@([pscustomobject]@{Arguments=$token})} }
function Get-CimInstance { param($ClassName,$Filter); if($Filter -match 'ProcessId=(\d+)') { $global:fakeProcesses | Where-Object ProcessId -eq ([int]$Matches[1]) } else { $global:fakeProcesses } }
function Stop-Process { param($Id,[switch]$Force); $global:events+="kill:$Id" }
function Stop-ScheduledTask { $global:events+='stop' }
function New-ScheduledTaskAction { param($Execute,$Argument); $global:taskArgs=$Argument; $global:taskExe=$Execute; 'action' }
function New-ScheduledTaskTrigger { 'trigger' }
function New-ScheduledTaskPrincipal { 'principal' }
function New-ScheduledTaskSettingsSet { 'settings' }
function Register-ScheduledTask { $global:events+='register' }
function Start-ScheduledTask { $global:events+='start' }
$env:LOCALAPPDATA='TEMP_DIR'
& 'INSTALLER'
$global:events -join ','
$global:taskArgs
$global:taskExe
""".replace('TEMP_DIR', tmp.replace("'", "''")).replace('INSTALLER', installer)
            result = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', cmd], capture_output=True, text=True, errors='replace')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('stop,kill:1003,kill:1002,kill:1001,register,start', result.stdout)
            self.assertIn('-EncodedCommand', result.stdout)
            self.assertIn('pythonw.exe', result.stdout)
            self.assertIn('CREATE_NO_WINDOW', result.stdout)

if __name__ == "__main__":
    unittest.main()

