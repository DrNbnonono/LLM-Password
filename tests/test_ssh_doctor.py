import importlib.util
import pathlib
import unittest
from unittest.mock import patch

PATH = pathlib.Path(__file__).resolve().parents[1] / 'plugins/codex-ssh-doctor/skills/ssh-doctor/scripts/doctor.py'


class DoctorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PATH.exists():
            raise AssertionError('SSH doctor implementation is missing')
        spec = importlib.util.spec_from_file_location('ssh_doctor', PATH)
        cls.doctor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.doctor)

    def test_host_cannot_inject_shell_or_ssh_options(self):
        for host in ['-Fconfig', 'dr@host;whoami', 'host$(id)', 'host name', '']:
            with self.subTest(host=host), self.assertRaises(ValueError):
                self.doctor.validate_host(host)
        self.assertEqual(self.doctor.validate_host('dr@10.137.11.194'), 'dr@10.137.11.194')

    def test_ssh_failure_skips_remote_checks_and_hides_raw_output(self):
        def run(args, **kwargs):
            if args[0] == 'ssh':
                return 255, '', 'PRIVATE_TOKEN must not appear'
            return 0, '{"exists":true,"state":"Running"}', ''
        report = self.doctor.diagnose('dr@host', 7890, 17890, 'Codex-Laptop-Proxy', runner=run, tcp=lambda p: True, system='Windows')
        checks = {x['name']: x for x in report['checks']}
        self.assertEqual(checks['ssh']['status'], 'fail')
        self.assertEqual(checks['remote_proxy']['status'], 'skipped')
        self.assertNotIn('PRIVATE_TOKEN', str(report))

    def test_remote_login_failure_is_reported_even_when_proxy_port_is_open(self):
        remote = '{"proxy_tcp":true,"proxy_https":true,"codex_version":"0.164.0","logged_in":false,"desktop_proxy":null}'
        def run(args, **kwargs):
            return (0, remote, '') if args[0] == 'ssh' else (0, '{"exists":true,"state":"Running"}', '')
        report = self.doctor.diagnose('dr@host', 7890, 17890, 'Codex-Laptop-Proxy', runner=run, tcp=lambda p: True, system='Windows')
        checks = {x['name']: x for x in report['checks']}
        self.assertEqual(checks['codex_login']['status'], 'fail')
        self.assertFalse(report['ok'])
        ssh_args = next(x for x in report['commands'] if x[0] == 'ssh')
        self.assertIn('StrictHostKeyChecking=yes', ssh_args)
        self.assertIn('ClearAllForwardings=yes', ssh_args)

    def test_https_probe_does_not_load_curl_config_or_read_process_environment(self):
        self.assertIn("['curl','-q',", self.doctor.REMOTE_PROBE)
        self.assertNotIn("joinpath('environ')", self.doctor.REMOTE_PROBE)

    def test_restart_requires_confirmation_and_never_runs_without_it(self):
        with patch.object(self.doctor, 'run_command') as run:
            with self.assertRaises(ValueError):
                self.doctor.restart_proxy('Codex-Laptop-Proxy', confirmed=False)
            run.assert_not_called()

    def test_check_mode_never_invokes_restart(self):
        calls = []
        def run(args, **kwargs):
            calls.append(args)
            return 255, '', ''
        self.doctor.diagnose('dr@host', 7890, 17890, 'Codex-Laptop-Proxy', runner=run, tcp=lambda p: False, system='Windows')
        self.assertFalse(any('restart' in x for args in calls for x in args))


if __name__ == '__main__':
    unittest.main()
