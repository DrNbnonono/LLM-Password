"""Read-only SSH/proxy diagnostics; explicit restart of an existing Windows task."""
import argparse
import json
import platform
import re
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REMOTE_PROBE = r'''
import json,os,pathlib,shutil,socket,subprocess
port=REMOTE_PORT
env=os.environ.copy()
proxy='http://127.0.0.1:'+str(port)
for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']: env[k]=proxy
env['NO_PROXY']=env['no_proxy']='localhost,127.0.0.1,::1'
result={'proxy_tcp':False,'proxy_https':False,'http_status':None,'codex_version':None,'logged_in':None,'desktop_proxy':None}
try:
    with socket.create_connection(('127.0.0.1',port),timeout=3): result['proxy_tcp']=True
except OSError: pass
if result['proxy_tcp'] and shutil.which('curl'):
    try:
        r=subprocess.run(['curl','-q','--proto','=https','--silent','--output','/dev/null','--write-out','%{http_code}','--connect-timeout','5','--max-time','10','--proxy',proxy,'https://auth.openai.com/'],capture_output=True,text=True,timeout=12,env=env)
        result['proxy_https']=r.returncode==0 and r.stdout.strip().isdigit() and int(r.stdout.strip())>0
        result['http_status']=int(r.stdout.strip()) if r.stdout.strip().isdigit() else None
    except (OSError,subprocess.TimeoutExpired): pass
candidates=[]
if shutil.which('codex'): candidates.append(pathlib.Path(shutil.which('codex')))
candidates.append(pathlib.Path.home()/'.local/bin/codex')
def version_key(p): return tuple(int(x) for x in p.parent.parent.name.lstrip('v').split('.') if x.isdigit())
candidates+=sorted((pathlib.Path.home()/'.nvm/versions/node').glob('*/bin/codex'),key=version_key,reverse=True)
binary=next((p for p in candidates if p.is_file() and os.access(p,os.X_OK)),None)
if binary:
    env['PATH']=str(binary.parent)+os.pathsep+env.get('PATH','')
    try:
        r=subprocess.run([str(binary),'--version'],capture_output=True,text=True,timeout=10,env=env)
        if r.returncode==0: result['codex_version']=r.stdout.strip()[:80]
        r=subprocess.run([str(binary),'login','status'],capture_output=True,text=True,timeout=15,env=env)
        result['logged_in']=r.returncode==0
    except (OSError,subprocess.TimeoutExpired): pass
print(json.dumps(result))
'''


def validate_host(host):
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]{0,252}', host):
        raise ValueError('Use an SSH alias or user@host without spaces or shell syntax.')
    return host


def validate_task(name):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', name):
        raise ValueError('Invalid scheduled task name.')
    return name


def run_command(args, timeout=15, input_text=None):
    try:
        result = subprocess.run(args, input=input_text, capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 124, '', 'timeout'
    except OSError:
        return 127, '', 'command unavailable'


def tcp_open(port):
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=2):
            return True
    except OSError:
        return False


def task_command(action, task):
    return ['powershell.exe', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
            '-File', str(Path(__file__).with_name('proxy_task.ps1')), '-Action', action,
            '-TaskName', validate_task(task)]


def diagnose(host, local_port, remote_port, task, runner=run_command, tcp=tcp_open, system=None):
    validate_host(host)
    validate_task(task)
    if not 1 <= local_port <= 65535 or not 1024 <= remote_port <= 65535:
        raise ValueError('Invalid proxy port.')
    report = {'time_utc': datetime.now(timezone.utc).isoformat(), 'host': host,
              'checks': [], 'commands': []}
    def add(name, status, detail):
        report['checks'].append({'name': name, 'status': status, 'detail': detail})
    add('local_proxy', 'pass' if tcp(local_port) else 'fail', f'Local TCP port {local_port}; this is not a VPN exit test.')
    if (system or platform.system()) == 'Windows':
        code, out, _ = runner(task_command('status', task), timeout=15)
        try:
            state = json.loads(out) if code == 0 else {}
        except ValueError:
            state = {}
        if not state:
            add('windows_task', 'warn', 'Unable to query scheduled task; check permissions.')
        else:
            running = state.get('exists') and state.get('state') == 'Running'
            add('windows_task', 'pass' if running else 'fail', f'{task}: {state.get("state", "missing")}')
    else:
        add('windows_task', 'skipped', 'Windows scheduled task inspection requires Windows.')
    callback_busy = tcp(1455)
    add('callback_1455', 'warn' if callback_busy else 'pass', 'Callback port 1455 is occupied.' if callback_busy else 'Callback port 1455 is free; device-code login does not require it.')
    args = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'ClearAllForwardings=yes', '-o', 'UpdateHostKeys=no',
            '-o', 'ConnectTimeout=8', '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2',
            host, 'python3 -']
    report['commands'].append(args)
    code, out, err = runner(args, timeout=55, input_text=REMOTE_PROBE.replace('REMOTE_PORT', str(remote_port)))
    remote = None
    if code != 0:
        reason = 'SSH connection or remote Python probe failed.'
        if 'Host key verification failed' in err:
            reason = 'SSH host key is unknown or changed. Verify the fingerprint manually.'
        elif 'Permission denied' in err:
            reason = 'SSH key authentication was rejected.'
        elif code == 124 or 'timed out' in err:
            reason = 'SSH or remote probe timed out.'
        add('ssh', 'fail', reason)
    else:
        add('ssh', 'pass', 'SSH batch authentication and remote command execution succeeded.')
        try:
            remote = json.loads(out)
            if not isinstance(remote, dict): remote = None
        except ValueError:
            pass
        if remote is None:
            add('remote_probe', 'fail', 'Remote probe did not return valid JSON; no raw output is displayed.')
    if remote is None:
        for name in ['remote_proxy', 'proxy_https', 'codex_version', 'codex_login', 'desktop_proxy']:
            add(name, 'skipped', 'Remote probe unavailable.')
    else:
        add('remote_proxy', 'pass' if remote.get('proxy_tcp') else 'fail', f'Remote TCP port {remote_port}.')
        status = remote.get('http_status')
        add('proxy_https', ('warn' if status and status >= 400 else 'pass') if remote.get('proxy_https') else 'fail',
            f'HTTPS response status: {status}. A response does not prove account or model access.')
        add('codex_version', 'pass' if remote.get('codex_version') else 'fail', 'Codex executable found.' if remote.get('codex_version') else 'Codex not found or version check failed.')
        add('codex_login', 'pass' if remote.get('logged_in') else 'fail', 'CLI reports authenticated.' if remote.get('logged_in') else 'CLI authentication unavailable; complete login separately.')
        desktop = remote.get('desktop_proxy')
        add('desktop_proxy', 'skipped' if desktop is None else ('pass' if desktop else 'warn'),
            'Desktop process environment is not inspected to avoid reading secrets.' if desktop is None else ('Desktop daemon has the expected proxy.' if desktop else 'Desktop daemon proxy differs; reconnect/restart may be needed.'))
    add('model_request', 'skipped', 'No model request was made. This report is not an end-to-end model availability guarantee.')
    report['ok'] = not any(x['status'] == 'fail' for x in report['checks'])
    return report


def restart_proxy(task, confirmed=False):
    if not confirmed:
        raise ValueError('Restart requires an explicit user request and --confirm.')
    if platform.system() != 'Windows':
        raise ValueError('Restart must run on the Windows laptop.')
    args = task_command('restart', task) + ['-Confirmed']
    code, _, _ = run_command(args, timeout=35)
    return {'action': 'restart-proxy', 'ok': code == 0,
            'detail': 'Existing task restarted. Run check to verify the tunnel.' if code == 0 else 'Task restart failed; check that it exists and your account has permission.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'restart-proxy'])
    parser.add_argument('--host', default='dr@10.137.11.194')
    parser.add_argument('--local-port', type=int, default=7890)
    parser.add_argument('--remote-port', type=int, default=17890)
    parser.add_argument('--task-name', default='Codex-Laptop-Proxy')
    parser.add_argument('--confirm', action='store_true')
    args = parser.parse_args()
    try:
        if args.action == 'restart-proxy':
            report = restart_proxy(args.task_name, args.confirm)
        else:
            report = diagnose(args.host, args.local_port, args.remote_port, args.task_name)
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return 0 if report['ok'] else 1
    except ValueError as exc:
        print(json.dumps({'ok': False, 'error': str(exc)}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
