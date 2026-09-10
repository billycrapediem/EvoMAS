"""Offline regression tests: python -m unittest discover -s tests -v."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from src.agents.spec import AgentSpec, get_backend_for_agent_type
from src.agents.runners.minisweagent import MinisweagentRunner, load_config, redact, register_endpoint_pricing
from src.mas.runtime import _get_runner_class
from minisweagent.models.test_models import DeterministicModel, make_output


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.git('init', '-q')
        self.git('config', 'user.name', 'Test')
        self.git('config', 'user.email', 'test@example.com')
        (self.repo / 'source.py').write_text('value = 1\n')
        self.git('add', '.')
        self.git('commit', '-qm', 'base')
        self.runner = MinisweagentRunner()
        self.runner.full_config['output_dir'] = str(self.root / 'artifacts')
        self.spec = AgentSpec(id='worker', role='worker', model_id='openai:test', temperature=0.2, max_tokens=77)
        self.task = f'Repository: {self.repo}\nInstance ID: example\n\nProblem Statement:\nFix the value.'

    def git(self, *args):
        return subprocess.check_output(['git', *args], cwd=self.repo, text=True)

    def test_dispatch_and_removed_names(self):
        self.assertIs(_get_runner_class('minisweagent'), MinisweagentRunner)
        self.assertEqual(get_backend_for_agent_type('DefaultAgent'), 'minisweagent')
        with self.assertRaisesRegex(ValueError, 'removed'):
            _get_runner_class('sweagent')
        with self.assertRaisesRegex(ValueError, 'removed'):
            get_backend_for_agent_type('SWEAgent')

    def test_custom_endpoint_pricing_alias(self):
        import litellm
        rates = {'input_cost_per_token': 0.000001, 'output_cost_per_token': 0.000002,
                 'litellm_provider': 'example'}
        with patch.dict(litellm.model_cost, {'example-model': rates}, clear=True), patch.object(litellm, 'register_model') as register:
            self.assertEqual(register_endpoint_pricing('openai/example-model'), 'example-model')
            self.assertEqual(register.call_args.args[0]['openai/example-model']['input_cost_per_token'], 0.000001)
            self.assertEqual(register.call_args.args[0]['openai/example-model']['litellm_provider'], 'openai')

    def test_missing_config_fails_and_alias_is_cwd_independent(self):
        with self.assertRaises(FileNotFoundError):
            load_config(str(self.root / 'missing.yaml'))
        self.assertIn('system_template', load_config()['agent'])
        self.assertEqual(load_config('simple')['agent']['step_limit'], 20)

    def test_v2_real_agent_submission_and_context(self):
        model = DeterministicModel(outputs=[make_output('Fixing', [{'command': "printf 'value = 2\\n' > source.py"}], 0),
                                            make_output('Done', [{'command': 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff HEAD'}], 0)])
        with patch('minisweagent.models.get_model', return_value=model):
            result = self.runner.run(self.spec, self.task, {'reports': {'reviewer': 'Check edge cases'}, 'task': self.task})
        self.assertTrue(result.success, result.error)
        self.assertIn('+value = 2', result.content)
        self.assertEqual(result.metadata['exit_status'], 'Submitted')
        self.assertEqual(result.metadata['model_stats']['api_calls'], 2)
        trajectory = json.loads(Path(result.metadata['trajectory_path']).read_text())
        self.assertIn('Check edge cases', trajectory['messages'][1]['content'])
        self.assertFalse(list(self.repo.glob('*.json')))

    def test_single_task_runtime_uses_current_repo_directory(self):
        import yaml
        from types import SimpleNamespace
        from src.utils.mas_runner import MasRunner
        from src.dataset.load_dataset import TaskData
        config = self.root / 'mas.yaml'
        agent_config = self.root / 'agent.yaml'
        agent_config.write_text(yaml.safe_dump(self.runner.full_config))
        config.write_text(yaml.safe_dump({
            'name': 'single', 'backend': 'minisweagent',
            'agents': {'worker': self.spec.model_dump() | {'agent_type': 'DefaultAgent'}},
            'topology': {'reports_to': {}},
            'execution': {'parallel_workers': False, 'agent_config': str(agent_config)},
        }))
        task = TaskData(id='example__repo-1', query='Fix the value', gt='REFERENCE_MUST_NOT_REACH_MODEL',
                        tag=[], source='SWE-BENCH',
                        metadata={'repo': 'example/repo', 'base_commit': self.git('rev-parse', 'HEAD').strip()})
        model = DeterministicModel(outputs=[make_output('Fix', [{'command': "printf 'value = 2\\n' > source.py"}], 0),
                                            make_output('Done', [{'command': 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff HEAD'}], 0)])
        runner = MasRunner(config, 'swe_bench_verified', output_dir=self.root / 'results',
                           verbose=False, use_cache=False, task_timeout=30)
        runner._dataset = SimpleNamespace(get_by_id=lambda task_id: task)
        (self.repo / '.git/info/exclude').write_text('installed.so\n')
        (self.repo / 'installed.so').write_text('compiled artifact')
        with patch.dict(os.environ, {'EVOMAS_REPOS_DIR': str(self.root)}), patch('minisweagent.models.get_model', return_value=model):
            result = runner.run_single_task(task.id)
        self.assertIsNone(result.error)
        self.assertIn('+value = 2', result.result)
        self.assertEqual(self.git('status', '--porcelain'), '')
        self.assertTrue((self.repo / 'installed.so').exists())
        trajectory = next((self.root / 'artifacts').rglob('trajectory.json')).read_text()
        self.assertNotIn('REFERENCE_MUST_NOT_REACH_MODEL', trajectory)

    def test_endpoint_config_and_redaction(self):
        with patch.dict(os.environ, {'API_KEY': 'test-private-credential', 'BASE_URL': 'https://example.test/v1'}):
            with patch('minisweagent.models.get_model') as get_model:
                self.runner.create_agent(self.spec, self.repo)
            name, config = get_model.call_args.args
            self.assertEqual(name, 'openai/test')
            self.assertEqual(config['model_kwargs']['api_base'], 'https://example.test/v1')
            self.assertEqual(config['model_kwargs']['api_key'], 'test-private-credential')
            self.assertEqual(config['model_kwargs']['max_tokens'], 77)
            self.assertEqual(config['model_kwargs']['temperature'], 0.2)
            self.assertNotIn('test-private-credential', json.dumps(redact({'api_key': 'test-private-credential', 'error': 'failed test-private-credential'})))

    def test_diff_includes_staged_unstaged_and_new_files(self):
        (self.repo / 'source.py').write_text('value = 2\n')
        self.git('add', 'source.py')
        (self.repo / 'source.py').write_text('value = 3\n')
        (self.repo / 'new.py').write_text('new = True\n')
        (self.repo / 'patch.txt').write_text('artifact')
        diff = self.runner._get_git_diff(self.repo)
        self.assertIn('+value = 3', diff)
        self.assertIn('new.py', diff)
        self.assertNotIn('patch.txt', diff)
        self.assertTrue(self.runner.validate_patch(self.repo, diff))
        self.assertFalse(self.runner.validate_patch(self.repo, 'Submitted'))

    def test_limit_and_model_errors_are_not_success(self):
        self.runner.full_config['agent']['step_limit'] = 1
        model = DeterministicModel(outputs=[make_output('Still working', [{'command': 'true'}], 0)])
        with patch('minisweagent.models.get_model', return_value=model):
            result = self.runner.run(self.spec, self.task)
        self.assertFalse(result.success)
        self.assertIn('LimitsExceeded', result.error)
        with patch.dict(os.environ, {'API_KEY': 'private-key'}), patch.object(self.runner, 'create_agent', side_effect=RuntimeError('failure private-key')):
            result = self.runner.run(self.spec, self.task)
        self.assertFalse(result.success)
        self.assertNotIn('private-key', result.error)


if __name__ == '__main__':
    unittest.main()
