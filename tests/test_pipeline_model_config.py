"""Shared configuration and client routing checks without model API calls."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.models import config
from src.models.model import OpenAIModel


class PipelineConfigTests(unittest.TestCase):
    def test_shared_model_and_role_overrides(self):
        with patch.object(config, 'load_model_environment'), patch.dict(os.environ, {'MODEL_ID': 'openai:shared'}, clear=True):
            self.assertEqual(config.pipeline_models(), ('openai:shared', 'openai:shared', ['openai:shared']))
            os.environ.update(META_MODEL='openai:meta', JUDGE_MODEL='openai:judge', AGENT_MODELS='openai:a openai:b')
            self.assertEqual(config.pipeline_models(), ('openai:meta', 'openai:judge', ['openai:a', 'openai:b']))

    def test_dotenv_is_root_anchored_and_exports_win(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(config, 'ROOT', Path(directory)), patch.dict(os.environ, {'MODEL_ID': 'openai:export'}, clear=True):
            (Path(directory) / '.env').write_text('MODEL_ID=openai:file\nMETA_MODEL=openai:meta\nAPI_KEY=private-value\n')
            self.assertEqual(config.pipeline_models(), ('openai:meta', 'openai:meta', ['openai:export']))
            self.assertEqual(os.environ['API_KEY'], 'private-value')

    def test_openai_client_uses_shared_endpoint_and_explicit_overrides(self):
        with patch('src.models.model.load_model_environment'), patch.dict(os.environ, {'API_KEY': 'shared-key', 'BASE_URL': 'https://example.test/v1'}, clear=True), patch('openai.OpenAI') as client:
            OpenAIModel('shared')
            self.assertEqual(client.call_args.kwargs['api_key'], 'shared-key')
            self.assertEqual(client.call_args.kwargs['base_url'], 'https://example.test/v1')
            OpenAIModel('explicit', api_key='explicit-key', base_url='https://override.test/v1')
            self.assertEqual(client.call_args.kwargs['api_key'], 'explicit-key')
            self.assertEqual(client.call_args.kwargs['base_url'], 'https://override.test/v1')

    def test_meta_and_judge_default_to_shared_client(self):
        from src.meta_model.metamodel import MetaModel
        from src.dataset.llm_as_judge import LLMAsJudgeEvaluator
        with patch('src.models.model.load_model_environment'), patch.object(config, 'load_model_environment'), patch.dict(os.environ, {'MODEL_ID': 'openai:shared', 'BASE_URL': 'https://example.test/v1', 'API_KEY': 'test-key'}, clear=True), patch('openai.OpenAI') as client:
            meta = MetaModel(memory_evolution=False, verbose=False)
            judge = LLMAsJudgeEvaluator()
            self.assertEqual(meta.model_id, 'openai:shared')
            self.assertEqual(judge.model_id, 'openai:shared')
            self.assertEqual(client.call_count, 2)
            for call in client.call_args_list:
                self.assertEqual(call.kwargs['base_url'], 'https://example.test/v1')

    def test_explicit_cli_model_flags_override_role_defaults(self):
        import contextlib
        import io
        import main
        summary = dict(initial_accuracy=0, final_accuracy=0, initial_reward=0, final_reward=0,
                       improved=False, added_to_pool=False, num_operations=0, output_location='unused')
        arguments = ['main.py', '--dataset', 'swe_bench_verified', '--meta-model-id', 'openai:cli-meta',
                     '--model-list', 'openai:cli-agent', '--llm-as-judge', 'none']
        with patch('sys.argv', arguments), patch.object(main, 'run_evolution_pipeline', return_value=summary) as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main.main(), 0)
        self.assertEqual(run.call_args.kwargs['meta_model_id'], 'openai:cli-meta')
        self.assertEqual(run.call_args.kwargs['model_list'], ['openai:cli-agent'])
        self.assertIsNone(run.call_args.kwargs['llm_as_judge'])

    def test_pool_models_are_constrained_without_changing_allowed_models(self):
        import yaml
        original = 'agents:\n  old:\n    model_id: bedrock:old\n  allowed:\n    model_id: openai:b\n'
        adapted = yaml.safe_load(config.constrain_agent_models(original, ['openai:a', 'openai:b']))
        self.assertEqual(adapted['agents']['old']['model_id'], 'openai:a')
        self.assertEqual(adapted['agents']['allowed']['model_id'], 'openai:b')
        with self.assertRaises(ValueError):
            config.constrain_agent_models(original, [])


if __name__ == '__main__':
    unittest.main()
