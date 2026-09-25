import http.client
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from video_moment_retrieval.cache import ArtifactCache
from video_moment_retrieval.pipeline import IndexConfig
from video_moment_retrieval.providers import OpenRouterBackend, OpenRouterClient
from video_moment_retrieval.search import SearchConfig, SearchEngine
from video_moment_retrieval.store import Store


class ReadinessTests(unittest.TestCase):
    def test_index_config_rejects_invalid_bounds_before_work(self):
        for kwargs in ({'max_seconds': float('nan')}, {'max_seconds': float('inf')},
                       {'max_seconds': 0}, {'max_seconds': -1}, {'window_seconds': 0},
                       {'overlap_seconds': 20}, {'overlap_seconds': -1},
                       {'ocr_every_seconds': float('nan')}, {'height': 0}, {'fps': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                IndexConfig(**kwargs)

    def test_empty_index_fails_before_provider_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp)/'index.sqlite')
            self.addCleanup(store.close)
            provider = Mock()
            engine = SearchEngine(store, ArtifactCache(tmp), Path(tmp), provider, provider, provider)
            with self.assertRaisesRegex(ValueError, 'no searchable records'):
                engine.search('red shirt', SearchConfig())
            self.assertEqual(provider.mock_calls, [])

    def test_missing_provider_choices_or_messages_raise_handled_errors(self):
        client = OpenRouterClient('fixture-key')
        for response in ({}, {'choices': []}, {'choices': [None]},
                         {'choices': [{'message': None}]}, {'choices': [{'message': {}}]}):
            with self.subTest(response=response), patch.object(client, 'request', return_value=response), \
                    self.assertRaises(ValueError):
                client.chat('fixture/model', 'test')

    def test_provider_usage_may_be_missing_or_null(self):
        client = OpenRouterClient('fixture-key')
        response = {'choices': [{'finish_reason': 'stop', 'message': {'content': '{"ok":true}'}}], 'usage': None}
        with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())):
            self.assertEqual(client.chat('fixture/model', 'test'), {'ok': True})
        stats = client.stats()
        self.assertEqual(stats['http_attempts'], 1)
        self.assertFalse(stats['cost_complete'])
        self.assertEqual(stats['reported_cost_usd'], 0)

    def test_null_and_nonfinite_usage_fields_do_not_break_json_reporting(self):
        client = OpenRouterClient('fixture-key')
        client.attempts = 2
        client.usage = [{'usage': None}, {'usage': {'cost': float('nan'), 'prompt_tokens': None, 'completion_tokens': 'unknown'}}]
        json.dumps(client.stats(), allow_nan=False)
        self.assertFalse(client.stats()['cost_complete'])

    def test_interrupted_http_body_is_retried_within_budget(self):
        client = OpenRouterClient('fixture-key', max_requests=2, retries=1)
        with patch('urllib.request.urlopen', side_effect=http.client.IncompleteRead(b'partial')), patch('time.sleep'):
            with self.assertRaisesRegex(RuntimeError, 'connection failed'):
                client.request('chat/completions', {'model': 'fixture/model'})
        self.assertEqual(client.attempts, 2)
        self.assertFalse(client.stats()['cost_complete'])

    def test_silent_transcription_prompt_preserves_json_object_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp)/'fixture.wav'
            audio.write_bytes(b'fixture')
            client = OpenRouterClient('fixture-key')
            backend = OpenRouterBackend(client, ArtifactCache(tmp))
            with patch.object(client, 'chat', return_value={'segments': []}) as chat:
                self.assertEqual(backend.transcribe(str(audio), 1), [])
            self.assertIn('Return {"segments":[]} if no intelligible speech.', chat.call_args.args[1])
