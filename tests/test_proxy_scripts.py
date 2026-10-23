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
            cmd = """
$ErrorActionPreference='Stop'
$global:events=@()
function Get-ScheduledTask { [pscustomobject]@{State='Ready'} }
function Stop-ScheduledTask { $global:events+='stop' }
function New-ScheduledTaskAction { param($Execute,$Argument); $global:taskArgs=$Argument; 'action' }
function New-ScheduledTaskTrigger { 'trigger' }
function New-ScheduledTaskPrincipal { 'principal' }
function New-ScheduledTaskSettingsSet { 'settings' }
function Register-ScheduledTask { $global:events+='register' }
function Start-ScheduledTask { $global:events+='start' }
$env:LOCALAPPDATA='TEMP_DIR'
& 'INSTALLER'
$global:events -join ','
$global:taskArgs
""".replace('TEMP_DIR', tmp.replace("'", "''")).replace('INSTALLER', installer)
            result = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', cmd], capture_output=True, text=True, errors='replace')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('stop,register,start', result.stdout)
            self.assertIn('-EncodedCommand', result.stdout)

if __name__ == "__main__":
    unittest.main()

