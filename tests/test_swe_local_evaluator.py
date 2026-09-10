"""Reject evaluator false positives using tiny local repositories and mocked tests."""
import contextlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.dataset import local_swe_evaluator as evaluator
from src.utils import lazy_env_manager


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.setup = {'repo_path': self.tmp.name, 'test_cmd': 'pytest -rA'}
        self.task = {'base_commit': 'abc', 'repo': 'example/repo', 'FAIL_TO_PASS': ['test_fix'],
                     'PASS_TO_PASS': ['test_old'], 'test_patch': 'test diff'}
        patches = [patch.object(evaluator, 'load_instance_data', return_value={'id': 'example__repo-1'}),
                   patch.object(evaluator, 'load_configs', return_value=({'example__repo-1': self.setup}, {'example__repo-1': self.task})),
                   patch.object(evaluator, 'normalize_instance_id', return_value=('example__repo-1', 'example__repo-1')),
                   patch.object(evaluator, 'ensure_env_ready', return_value='test-env'),
                   patch.object(evaluator, 'repo_reset_and_clean_checkout'),
                   patch.object(evaluator, 'get_parser_for_repo', return_value=evaluator.parse_log_pytest)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def evaluate(self, log='PASSED test_fix\nPASSED test_old', applied=True, returncode=0):
        @contextlib.contextmanager
        def apply(*args):
            yield applied
        with patch.object(evaluator, 'apply_patch', apply), patch.object(evaluator, 'run_tests', return_value=(log, returncode)):
            return evaluator.evaluate_patch('example__repo-1', 'diff', use_cache=True)

    def test_complete_results_pass(self):
        self.assertEqual(self.evaluate()['resolved'], 'FULL')

    def test_missing_regression_test_fails(self):
        self.assertEqual(self.evaluate('PASSED test_fix')['error'], 'missing_test_results')

    def test_patch_failure_and_empty_test_list_fail(self):
        self.assertEqual(self.evaluate(applied=False)['error'], 'test_patch_apply_failed')
        self.task['FAIL_TO_PASS'] = []
        self.assertEqual(self.evaluate()['error'], 'missing_required_tests_or_patch')

    def test_test_process_failure_is_not_full_resolution(self):
        self.assertEqual(self.evaluate(returncode=2)['error'], 'test_execution_failed')
        self.assertNotEqual(self.evaluate(returncode=1).get('resolved'), 'FULL')

    def test_pytest_both_output_formats(self):
        self.assertEqual(evaluator.parse_log_pytest('test_x PASSED [100%]\nPASSED test_y'),
                         {'test_x': 'PASSED', 'test_y': 'PASSED'})

    def test_failed_install_not_marked_ready(self):
        with patch.object(lazy_env_manager, 'load_configs', return_value=({'test': self.setup | {'env_name': 'bad-env'}}, {'test': self.task})), \
             patch.object(lazy_env_manager, 'env_exists', return_value=False), \
             patch.object(lazy_env_manager, 'create_conda_env', return_value=True), \
             patch.object(lazy_env_manager, 'install_base_packages', return_value=True), \
             patch.object(lazy_env_manager, 'setup_repository', return_value=False), \
             patch.object(lazy_env_manager, 'cleanup_env') as cleanup:
            self.assertIsNone(lazy_env_manager.ensure_env_ready('test'))
            cleanup.assert_called_once_with('bad-env')


class SetupTests(unittest.TestCase):
    def test_reusing_workspace_republishes_only_selected_instance(self):
        from src.dataset import swe_smoke_setup
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            root = Path(directory)
            dataset = root / 'dataset/swe_bench_verified'
            dataset.mkdir(parents=True)
            workspace = root / 'workspace'
            repo = workspace / 'repos/astropy'
            repo.mkdir(parents=True)
            python = workspace / 'envs/test-env/bin/python'
            python.parent.mkdir(parents=True)
            python.touch()
            instance_id = 'astropy__astropy-12907'
            item = {'id': instance_id, 'tag': ['version:4.3'],
                    'metadata': {'repo': 'astropy/astropy', 'instance_id': instance_id,
                                 'FAIL_TO_PASS': '["test_new"]', 'PASS_TO_PASS': '["test_old"]'}}
            (dataset / 'test.json').write_text(json.dumps([item]))
            (dataset / 'setup_map.json').write_text(json.dumps({'other': {'untouched': True}}))
            setup = {'env_name': 'test-env', 'repo_path': str(repo)}
            (workspace / 'ready.json').write_text(json.dumps({'instance_id': instance_id, 'setup': setup}))
            with patch.object(swe_smoke_setup, 'ROOT', root), patch.object(swe_smoke_setup.subprocess, 'run') as run:
                result = swe_smoke_setup.prepare_instance(instance_id, workspace)
            self.assertEqual(result, setup)
            run.assert_not_called()
            entries = json.loads((dataset / 'setup_map.json').read_text())
            self.assertEqual(entries['other'], {'untouched': True})
            tasks = json.loads((dataset / 'tasks_map.json').read_text())
            key = evaluator.normalize_instance_id(instance_id)[1]
            self.assertEqual(tasks[key]['FAIL_TO_PASS'], ['test_new'])


if __name__ == '__main__':
    unittest.main()
