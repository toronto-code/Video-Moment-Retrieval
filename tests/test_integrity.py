import contextlib
import io
import json
import sqlite3
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from video_moment_retrieval.cache import ArtifactCache
from video_moment_retrieval.cli import main
from video_moment_retrieval.demo import DemoBackend, DemoEncoder, build_demo
from video_moment_retrieval.evaluation import evaluate
from video_moment_retrieval.pipeline import Indexer, IndexConfig, validate_extraction, validate_segments
from video_moment_retrieval.providers import OpenRouterBackend, OpenRouterClient
from video_moment_retrieval.search import SearchEngine, SearchConfig
from video_moment_retrieval.store import Store
from video_moment_retrieval.types import Candidate, Evidence, QueryPlan, Record, Verdict, Verification


class Backend:
    identity = 'fixture-v1'
    def extract(self, path, duration, transcript, context=None):
        return {'summary': 'person', 'observations': [{'kind': 'appearance', 'start': 0, 'end': 1,
            'text': self.identity, 'subject': 'P', 'attributes': {'clothing': 'red'}}]}
    def reconcile(self, records):
        return []


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root/'source.mp4'
        self.source.write_bytes(b'fixture source')
        self.store = Store(self.root/'index.sqlite')
        self.addCleanup(self.store.close)
        self.backend = Backend()
        self.indexer = Indexer(self.store, ArtifactCache(self.root/'cache'), self.root/'work',
            self.backend, self.backend, self.backend, DemoEncoder())
        self.config = IndexConfig(window_seconds=2, overlap_seconds=0, enable_ocr=False)
        self.enterContext(patch('video_moment_retrieval.pipeline.media.probe', return_value={'duration': 6, 'has_audio': False}))
        self.enterContext(patch('video_moment_retrieval.pipeline.media.clip', side_effect=lambda source, target, *args: str(target)))

    def index(self):
        return self.indexer.index(str(self.source), self.config)

    def test_read_only_legacy_database_coverage_needs_no_migration(self):
        self.index()
        self.store.db.execute("DROP TABLE index_attempts")
        self.store.db.commit()
        path = Path(self.store.path)
        original = path.read_bytes()
        readonly = Store(path, read_only=True)
        try:
            coverage = readonly.coverage()
            self.assertTrue(coverage['videos'])
            self.assertEqual(coverage['attempts'], [])
        finally:
            readonly.close()
        self.assertEqual(path.read_bytes(), original)

    def test_failed_reindex_preserves_records_vectors_and_published_coverage(self):
        first = self.index()
        records = [r.to_dict() for r in self.store.records()]
        coverage = self.store.coverage()['videos']
        revision = self.store.get_meta('index_revision')
        self.backend.identity = 'new-model'
        with patch.object(self.backend, 'extract', side_effect=RuntimeError('provider offline')):
            failed = self.index()
        self.assertEqual(failed['publication'], 'retained_previous')
        self.assertEqual([r.to_dict() for r in self.store.records()], records)
        self.assertEqual(self.store.coverage()['videos'], coverage)
        self.assertEqual(self.store.get_meta('index_revision'), revision)
        self.assertTrue(failed['errors'])
        self.assertTrue(Path(failed['attempt_report']).is_file())
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM vectors').fetchone()[0], len(records))
        recovered = self.index()
        self.assertEqual(recovered['publication'], 'published')
        self.assertTrue(any(r.text == 'new-model' for r in self.store.records()))
        self.assertNotEqual(self.store.get_meta('index_revision'), revision)

    def test_interruption_preserves_last_snapshot_and_marks_attempt(self):
        self.index()
        before = [r.to_dict() for r in self.store.records()]
        self.backend.identity = 'interrupted-model'
        with patch.object(self.backend, 'extract', side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            self.index()
        self.assertEqual([r.to_dict() for r in self.store.records()], before)
        self.assertEqual(self.store.coverage()['attempts'][0]['status'], 'interrupted')

    def test_publish_failure_rolls_back_entire_snapshot(self):
        self.index()
        before = [r.to_dict() for r in self.store.records()]
        revision = self.store.get_meta('index_revision')
        self.store.db.execute("CREATE TRIGGER fail_publish BEFORE INSERT ON coverage BEGIN SELECT RAISE(ABORT, 'disk fixture'); END")
        self.backend.identity = 'new-model'
        with self.assertRaises(sqlite3.IntegrityError):
            self.index()
        self.assertEqual([r.to_dict() for r in self.store.records()], before)
        self.assertEqual(self.store.get_meta('index_revision'), revision)

    def test_failed_neighbor_does_not_become_complete_reconciliation(self):
        original = self.backend.extract
        def extract(path, *args, **kwargs):
            if 'visual-2.0' in path:
                raise RuntimeError('middle failed')
            return original(path, *args, **kwargs)
        with patch.object(self.backend, 'extract', side_effect=extract), patch.object(self.backend, 'reconcile') as reconcile:
            report = self.index()
        stage = report['attempt_coverage']['videos'][0]['stages']['reconciliation']
        self.assertEqual(stage['processed_seconds'], 0)
        self.assertEqual(stage['statuses'], ['skipped'])
        reconcile.assert_not_called()

    def test_malformed_response_is_a_stage_failure_not_a_crash(self):
        with patch.object(self.backend, 'extract', return_value={'summary':'person','observations':[
            {'start':0,'end':1,'kind':'appearance','text':'person','attributes':None}]}):
            report = self.index()
        self.assertTrue(report['errors'])
        self.assertEqual({e['stage'] for e in report['errors']}, {'visual'})

    def test_ingestion_reserves_attempts_for_embeddings_and_restores_client_limit(self):
        client = OpenRouterClient('fixture', max_requests=3, retries=0)
        class LimitedBackend(Backend):
            def ensure_budget(self):
                self.client.ensure_budget()
            def extract(self, *args, **kwargs):
                self.ensure_budget(); self.client.attempts += 1
                return super().extract(*args, **kwargs)
            def reconcile(self, records):
                self.ensure_budget(); self.client.attempts += 1
                return []
        class LimitedEncoder(DemoEncoder):
            def encode(self, texts):
                self.client.ensure_budget(); self.client.attempts += 1
                return super().encode(texts)
        backend, encoder = LimitedBackend(), LimitedEncoder()
        backend.client = encoder.client = client
        indexer = Indexer(self.store, ArtifactCache(self.root/'limited-cache'), self.root/'limited-work',
            backend, backend, backend, encoder)
        report = indexer.index(str(self.source), self.config)
        self.assertTrue(report['errors'])
        self.assertGreater(self.store.db.execute('SELECT count(*) FROM vectors').fetchone()[0], 0)
        self.assertGreaterEqual(client.attempts, 2)
        self.assertLessEqual(client.attempts, 3)
        self.assertIsNone(client.stage_limit)
        self.assertGreaterEqual(report['reserved_requests_by_stage']['embedding'], 1)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.clip = self.root/'clip.mp4'; self.clip.write_bytes(b'fixture')
        self.audio = self.root/'audio.wav'; self.audio.write_bytes(b'fixture')
        self.store = Store(self.root/'index.sqlite'); self.addCleanup(self.store.close)
        self.engine = SearchEngine(self.store, ArtifactCache(self.root/'cache'), self.root, DemoEncoder(), None, None)
        self.candidate = Candidate(Record('r','v','appearance',1,2,'person',[Evidence('e','visual',1,2,str(self.clip))]),1,[])

    def value(self, binding):
        v = Verdict('supported','visible',1,2,['media:visual'],
            [{'criterion':'same person','status':'supported','evidence_ids':['media:visual']}], {'subject_binding':binding})
        return {'verification':Verification([v],True).to_dict(), 'inspection':{'start':0,'end':3,'clip':str(self.clip)}}

    def test_visual_identity_requires_no_speech_and_accepts_visual_proof(self):
        plan = QueryPlan('red shirt handcuffed',['red'], modalities=['visual'],criteria=['same person'],requires_subject_link=True)
        self.assertEqual(plan.subject_link_kind, 'visual')
        self.assertEqual(plan.modalities, ['visual'])
        self.assertNotIn('speech', plan.subject_criterion)
        value = self.value({'person':'P','method':'visual_tracking','reason':'same tracked person', 'evidence_ids':['media:visual']})
        self.engine.validate_verification(value, plan, self.candidate)

    def test_nested_binding_cannot_invent_audio_or_any_other_reference(self):
        plan = QueryPlan('same person',['person'],criteria=['same person'],requires_subject_link=True)
        for refs in (['media:visual','invented'], ['media:visual','media:audio']):
            value = self.value({'person':'P','method':'visual_tracking','reason':'tracked', 'evidence_ids':refs})
            with self.subTest(refs=refs), self.assertRaisesRegex(ValueError,'supplied media'):
                self.engine.validate_verification(value, plan, self.candidate)

    def test_speaker_binding_requires_both_inputs_and_explanation(self):
        plan = QueryPlan('red shirt shouting',['red'],criteria=['same person'],subject_link_kind='speaker_visual')
        self.assertIn('audio',plan.modalities)
        binding = {'person':'P','speaker':'S','method':'visible_synchronized_speech', 'reason':'synchronized speech',
                   'evidence_ids':['media:visual','media:audio']}
        value = self.value(binding)
        with self.assertRaises(ValueError):
            self.engine.validate_verification(value, plan, self.candidate)
        value['inspection']['audio'] = str(self.audio)
        value['verification']['verdicts'][0]['evidence_ids'].append('media:audio')
        self.engine.validate_verification(value, plan, self.candidate)
        del value['verification']['verdicts'][0]['details']['subject_binding']['reason']
        with self.assertRaises(ValueError):
            self.engine.validate_verification(value, plan, self.candidate)

    def test_ocr_box_frame_and_interval_are_validated_by_live_adapter(self):
        frame = self.root/'frame.png'; frame.write_bytes(b'fixture')
        client = OpenRouterClient('fixture')
        backend = OpenRouterBackend(client, ArtifactCache(self.root/'cache'))
        plan = QueryPlan('read plate',['plate'],modalities=['ocr'],criteria=['read plate'])
        good = {'text':'ABC123','legibility':'readable','bbox':[.1,.1,.5,.5],'frame_id':'media:frame:0'}
        for detail, ok in ((good, True), ({k:v for k,v in good.items() if k!='bbox'},False),
                           ({**good,'bbox':[.5,.5,.1,.1]},False), ({**good,'frame_id':'media:frame:9'},False)):
            response = {'verdicts':[{'status':'supported','reason':'visible','start':1,'end':3,
                'evidence_ids':['media:frame:0'], 'criteria':[{'criterion':'read plate','status':'supported','evidence_ids':['media:frame:0']}],
                'details':deepcopy(detail)}],'complete':True}
            with self.subTest(detail=detail), patch.object(client,'chat',return_value=response):
                if ok:
                    result = backend.verify(plan,self.candidate,{'start':10,'end':15,'frames':[{'time':12,'path':str(frame)}]})
                    self.assertEqual(result.verdicts[0].start,11)
                else:
                    with self.assertRaises(ValueError):
                        backend.verify(plan,self.candidate,{'start':10,'end':15,'frames':[{'time':12,'path':str(frame)}]})

    def test_nested_model_schemas_fail_with_value_errors(self):
        for kwargs in ({'constraints':None},{'keywords':None},{'modalities':[None]}):
            params={'query':'test','keywords':['test'],**kwargs}
            with self.subTest(params=params), self.assertRaises(ValueError): QueryPlan(**params)
        with self.assertRaises(ValueError):
            validate_segments([{'start':0,'end':1,'text':'test','words':[None]}],2)
        with self.assertRaises(ValueError):
            Verdict('supported',None,details=None).validate(0,1,set(),[])
        with self.assertRaises(ValueError): Verification.from_dict({'verdicts':[None],'complete':True})


class SearchFailureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = Store(self.root/'index.sqlite'); self.addCleanup(self.store.close)
        build_demo(self.store,self.root/'labels.json')
        self.backend = DemoBackend()
        self.engine = SearchEngine(self.store,ArtifactCache(self.root/'cache'),self.root,DemoEncoder(),self.backend,self.backend,True)

    def test_failure_is_not_a_correct_negative_evaluation(self):
        labels=json.loads((self.root/'labels.json').read_text())
        labels['queries']=[q for q in labels['queries'] if not q['matches']]
        with patch.object(self.backend,'verify',side_effect=RuntimeError('HTTP 503')):
            report=evaluate(self.engine,labels,SearchConfig(),['combined+verification'])
        query=report['policies'][0]['queries'][0]
        self.assertFalse(query['evaluation_valid'])
        self.assertIsNone(query['no_match_correct'])
        self.assertTrue(report['operational_errors'])
        self.assertEqual(report['policies'][0]['failed_query_count'],1)
        self.assertEqual(report['policies'][0]['valid_query_count'],0)

    def test_cli_returns_nonzero_for_verification_outage(self):
        with patch.object(DemoBackend,'verify',side_effect=RuntimeError('HTTP 503')), contextlib.redirect_stdout(io.StringIO()):
            code=main(['--data-dir',str(self.root),'--env-file',str(self.root/'absent'), '--demo','search','handcuffed'])
        self.assertEqual(code,2)

    def test_model_abstention_is_separate_from_operational_failure(self):
        with patch.object(self.backend,'verify',return_value=Verification([Verdict('unresolved','occluded')],False)):
            response=self.engine.search('handcuffed',SearchConfig())
        self.assertEqual(response['operational_errors'],[])
        self.assertEqual(response['decision'],'abstained')
        self.assertFalse(response['pagination']['all_tasks_verified'])

    def test_enumeration_enriches_only_page_and_reuses_retrieval_queue(self):
        config=SearchConfig(enumerate_all=True,verify=False,assess=False,top_k=1)
        with patch.object(self.store,'context',wraps=self.store.context) as context:
            first=self.engine.search('handcuffed',config)
            self.assertEqual(context.call_count,1)
        with patch('video_moment_retrieval.search.retrieve',side_effect=AssertionError('retrieval repeated')), \
                patch.object(self.store,'context',wraps=self.store.context) as context:
            second=self.engine.search('handcuffed',SearchConfig(enumerate_all=True,verify=False,assess=False,top_k=1,
                offset=1,snapshot=first['pagination']['snapshot']))
            self.assertEqual(context.call_count,1)
        self.assertNotEqual(first['candidates'][0]['record_id'],second['candidates'][0]['record_id'])

    def test_enumeration_revision_invalidates_saved_queue(self):
        first=self.engine.search('handcuffed',SearchConfig(enumerate_all=True,verify=False,top_k=1))
        records=self.store.records();records[0].text='changed'
        self.store.replace_records(records[0].video_id,records,DemoEncoder().encode([r.text for r in records]),DemoEncoder.identity)
        with self.assertRaisesRegex(ValueError,'settings changed'):
            self.engine.search('handcuffed',SearchConfig(enumerate_all=True,verify=False,top_k=1,offset=1,snapshot=first['pagination']['snapshot']))


class BudgetTests(unittest.TestCase):
    def test_reservations_include_retries_and_unused_attempts_roll_forward(self):
        from types import SimpleNamespace
        from video_moment_retrieval.budget import StageBudgets
        client = OpenRouterClient('fixture',max_requests=10)
        provider = SimpleNamespace(client=client)
        budget = StageBudgets([('visual',provider,35),('ocr',provider,25),('embedding',provider,10)])
        budget.enter('visual')
        visual_limit = client.stage_limit
        client.attempts = visual_limit
        with patch('urllib.request.urlopen',side_effect=AssertionError('reserved attempt spent')):
            with self.assertRaisesRegex(RuntimeError,'reserved'):
                client.request('embeddings',{'model':'fixture'})
        budget.enter('ocr')
        self.assertGreater(client.stage_limit,visual_limit)
        self.assertEqual(client.stage_limit,10-budget.allocations['embedding'])
        budget.enter('embedding')
        self.assertEqual(client.stage_limit,10)
        client.ensure_budget()
        budget.restore()
        self.assertIsNone(client.stage_limit)
