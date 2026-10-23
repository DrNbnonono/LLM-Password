import pathlib
import subprocess
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / 'plugins/codex-ssh-doctor/skills/ssh-doctor/scripts/proxy_task.ps1'


class TaskHelperTests(unittest.TestCase):
    def run_helper(self, reuse_pid=False):
        command = r'''
$ErrorActionPreference='Stop'
$global:events=@()
$token=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes(('CodexLaptopProxy- ExitOnForwardFailure=yes '+('A'*100))))
$created=Get-Date '2026-10-23'
$global:processes=@(
 [pscustomobject]@{ProcessId=1001;ParentProcessId=10;Name='pythonw.exe';CommandLine=$token;CreationDate=$created},
 [pscustomobject]@{ProcessId=1002;ParentProcessId=1001;Name='ssh.exe';CommandLine='ssh';CreationDate=$created},
 [pscustomobject]@{ProcessId=2001;ParentProcessId=20;Name='pythonw.exe';CommandLine=$token;CreationDate=$created},
 [pscustomobject]@{ProcessId=2002;ParentProcessId=2001;Name='ssh.exe';CommandLine='ssh';CreationDate=$created}
)
function Get-ScheduledTask { [pscustomobject]@{State='Ready';Actions=@([pscustomobject]@{Arguments=$token;Execute='E:/Python/pythonw.exe'})} }
function New-Object {
 param($ComObject)
 $running=[pscustomobject]@{}
 $running | Add-Member ScriptMethod GetInstances { param($flags); @([pscustomobject]@{EnginePID=1001}) }
 $folder=[pscustomobject]@{Running=$running}
 $folder | Add-Member ScriptMethod GetTask { param($name); $this.Running }
 $service=[pscustomobject]@{Folder=$folder}
 $service | Add-Member ScriptMethod Connect {}
 $service | Add-Member ScriptMethod GetFolder { param($path); $this.Folder }
 $service
}
function Get-CimInstance {
 param($ClassName,$Filter)
 if($Filter -match 'ProcessId=([0-9]+)') {
   $found=$global:processes | Where-Object ProcessId -eq ([int]$Matches[1])
   if(REUSE_PID -and $found.ProcessId -eq 1002) { $found=$found.PSObject.Copy(); $found.CreationDate=$created.AddDays(1) }
   $found
 } else { $global:processes }
}
function Stop-ScheduledTask { $global:events+='stop' }
function Start-ScheduledTask { $global:events+='start' }
function Stop-Process { param($Id,[switch]$Force); $global:events+="kill:$Id" }
function Register-ScheduledTask { throw 'Restart must not rewrite task configuration' }
& 'SCRIPT_PATH' -Action restart -Confirmed
$global:events -join ','
'''.replace('SCRIPT_PATH', str(SCRIPT).replace("'", "''")).replace('REUSE_PID', '$true' if reuse_pid else '$false')
        return subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', command], capture_output=True, text=True, errors='replace')

    def test_restart_only_cleans_owned_tree_and_preserves_task(self):
        result = self.run_helper()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('stop,kill:1002,kill:1001,start', result.stdout)
        self.assertNotIn('kill:200', result.stdout)

    def test_restart_does_not_kill_a_reused_pid(self):
        result = self.run_helper(reuse_pid=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('stop,kill:1001,start', result.stdout)
        self.assertNotIn('kill:1002', result.stdout)


if __name__ == '__main__':
    unittest.main()
