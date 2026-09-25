"""Regression coverage for the independent follow-up review; no provider calls."""
import contextlib
import io
import json
import os
import sqlite3
import stat
import tempfile
import unittest
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from video_moment_retrieval import media
from video_moment_retrieval.cache import ArtifactCache, atomic_json
from video_moment_retrieval.cli import main
from video_moment_retrieval.demo import DemoBackend, DemoEncoder, build_demo
from video_moment_retrieval.download import entries
from video_moment_retrieval.pipeline import IndexConfig, Indexer
from video_moment_retrieval.retrieval import retrieve
from video_moment_retrieval.search import SearchConfig, SearchEngine
from video_moment_retrieval.speech import WebRTCSpeechDetector
from video_moment_retrieval.store import Store
from video_moment_retrieval.types import Candidate, Evidence, Record, Verdict, Verification
from .test_integrity import Backend


class PipelineReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root/'source.mp4'
        self.source.write_bytes(b'fixture')
        self.store = Store(self.root/'index.sqlite')
        self.addCleanup(self.store.close)
        self.cache = ArtifactCache(self.root/'cache')
        self.backend = Backend()
        self.backend.transcribe = lambda *args: []
        self.detector = SimpleNamespace(identity='vad-fixture', detect=lambda path: [{'start': 0, 'end': 1}])
        self.indexer = Indexer(self.store, self.cache, self.root/'work', self.backend,
                              self.backend, self.backend, DemoEncoder(), speech_detector=self.detector)
        self.info = {'duration': 50, 'has_audio': False}
        self.enterContext(patch('video_moment_retrieval.pipeline.media.probe', return_value=self.info))
        self.enterContext(patch('video_moment_retrieval.pipeline.media.clip', side_effect=lambda source, target, *args: str(target)))
        self.enterContext(patch('video_moment_retrieval.pipeline.media.audio', return_value='fixture.wav'))
        self.config = IndexConfig(enable_ocr=False)

    def index(self):
        return self.indexer.index(str(self.source), self.config)

    def test_ocr_clusters_span_entire_window_and_reuse_valid_cache(self):
        seen = []
        def frames(inputs):
            seen.append([f['time'] for f in inputs])
            return [{'frame_index': 4, 'text': 'LATE', 'text_type': 'license_plate',
                     'bbox': [.1,.1,.5,.5], 'legibility': 'readable'}]
        def crop(source, target, box):
            target.write_bytes(b'fixture crop')
            return str(target)
        self.backend.read_frames = frames
        self.backend.read_crop = lambda path: {'text': 'LATE', 'legibility': 'readable'}
        self.config.enable_ocr = True
        with patch.object(media, 'frame', side_effect=lambda source, target, time, height=720: str(target)), \
                patch.object(media, 'crop', side_effect=crop), patch.object(media, 'sharpness', return_value=1):
            report = self.index()
            self.assertEqual(report['errors'], [])
            self.assertEqual(len(seen), 5)
            self.assertEqual(seen[-1], [41.8,42.,42.2,44.8,45.,45.2,47.8,48.,48.2])
            self.assertTrue(any(r.kind == 'ocr' and r.start == 45 and 'LATE' in r.text for r in self.store.records()))
            self.index()
            self.assertEqual(len(seen), 5)  # unchanged sampling reuses valid artifacts

    def test_ocr_samples_stay_inside_very_short_final_window(self):
        self.info['duration'] = 10.001
        seen = []
        self.backend.read_frames = lambda inputs: seen.append([f['time'] for f in inputs]) or []
        self.config.enable_ocr = True
        with patch.object(media, 'frame', side_effect=lambda source, target, time, height=720: str(target)):
            report = self.index()
        self.assertEqual(report['errors'], [])
        self.assertEqual(len(seen[-1]), 9)
        self.assertTrue(all(10 < t < 10.001 for t in seen[-1]))

    def test_ocr_cache_is_bounded_by_frame_height(self):
        calls = []
        self.backend.read_frames = lambda inputs: calls.append(1) or []
        self.config.enable_ocr = True
        with patch.object(media, 'frame', side_effect=lambda source, target, time, height: str(target)):
            self.assertEqual(self.index()['errors'], [])
            self.index()
            self.assertEqual(len(calls), 5)   # same height reuses OCR artifacts
            self.config.height = 640
            self.index()
            self.assertEqual(len(calls), 10)  # a new frame height recomputes them

    def test_malformed_vad_becomes_failed_stage_without_aborting_index(self):
        self.info['has_audio'] = True
        for malformed in ([{'start': 0}], [None], None):
            with self.subTest(malformed=malformed):
                self.detector.detect = lambda path: malformed
                report = self.index()
                self.assertTrue(any(e['stage'] == 'vad' for e in report['errors']))
                stages = report['attempt_coverage']['videos'][0]['stages']
                self.assertEqual(stages['vad']['statuses'], ['failed'])
                self.assertEqual(stages['audio']['statuses'], ['skipped'])
                self.assertEqual(stages['embedding']['statuses'], ['complete'])

    def test_invalid_vad_cache_is_recomputed(self):
        self.info['has_audio'] = True
        with patch.object(media, 'energy_observations', return_value=[]):
            self.assertEqual(self.index()['errors'], [])
            entry = next((self.cache.root/'vad').glob('*.json'))
            entry.write_text('[{"start": 0}]')
            with patch.object(self.detector, 'detect', wraps=self.detector.detect) as detect:
                self.assertEqual(self.index()['errors'], [])
                detect.assert_called_once()
            self.assertEqual(json.loads(entry.read_text()), [{'start': 0, 'end': 1}])

    def test_energy_failure_preserves_completed_vad_and_discards_partial_energy(self):
        self.info['has_audio'] = True
        malformed = [{'start': 0, 'end': 1, 'speaker': None}, {'start': 2}]
        with patch.object(media, 'energy_observations', return_value=malformed):
            report = self.index()
        stages = report['attempt_coverage']['videos'][0]['stages']
        self.assertEqual(stages['vad']['statuses'], ['complete'])
        self.assertEqual(stages['audio']['statuses'], ['failed'])
        self.assertFalse(any(r.kind == 'audio' for r in self.store.records()))

    def test_retained_reindex_keeps_published_sidecars_and_new_attempt_is_separate(self):
        first = self.index()
        selected = self.store.video(first['video_id'])['metadata']['artifacts']
        original = {k: Path(p).read_bytes() for k,p in selected.items()}
        self.backend.identity = 'new-version'
        with patch.object(self.backend, 'extract', side_effect=RuntimeError('outage')):
            failed = self.index()
        self.assertEqual(failed['publication'], 'retained_previous')
        self.assertEqual(self.store.video(first['video_id'])['metadata']['artifacts'], selected)
        self.assertNotEqual(failed['artifacts'], selected)
        self.assertTrue(json.loads(Path(failed['artifacts']['index_report']).read_text())['errors'])
        for name, path in selected.items():
            self.assertEqual(Path(path).read_bytes(), original[name])
        recovered = self.index()
        self.assertEqual(self.store.video(first['video_id'])['metadata']['artifacts'], recovered['artifacts'])
        self.assertCountEqual(json.loads(Path(recovered['artifacts']['observations']).read_text()), [r.to_dict() for r in self.store.records()])

    def test_failed_publication_keeps_previous_sidecar_selection(self):
        first = self.index()
        before = self.store.video(first['video_id'])['metadata']['artifacts']
        original = {k: Path(v).read_bytes() for k,v in before.items()}
        self.store.db.execute("CREATE TRIGGER reject_publish BEFORE INSERT ON coverage BEGIN SELECT RAISE(ABORT, 'fixture failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.index()
        self.assertEqual(self.store.video(first['video_id'])['metadata']['artifacts'], before)
        self.assertEqual({k: Path(v).read_bytes() for k,v in before.items()}, original)

    def test_publish_uses_column_names_and_rejects_incompatible_schema(self):
        first = self.index()
        self.store.db.executescript('''ALTER TABLE coverage RENAME TO coverage_old;
          CREATE TABLE coverage(detail TEXT NOT NULL, status TEXT NOT NULL, end REAL NOT NULL,
            start REAL NOT NULL, stage TEXT NOT NULL, video_id TEXT NOT NULL REFERENCES videos(id),
            PRIMARY KEY(video_id,stage,start,end));
          INSERT INTO coverage SELECT detail,status,end,start,stage,video_id FROM coverage_old;
          DROP TABLE coverage_old;''')
        expected = self.store.coverage()['videos']
        self.assertEqual(self.index()['publication'], 'published')
        self.assertEqual(self.store.coverage()['videos'], expected)
        records = [r.to_dict() for r in self.store.records()]
        self.store.db.execute('ALTER TABLE coverage ADD COLUMN unsupported TEXT')
        with self.assertRaisesRegex(ValueError, 'Incompatible.*schema'):
            self.index()
        self.assertEqual([r.to_dict() for r in self.store.records()], records)


class BoundaryReviewTests(unittest.TestCase):
    def test_windows_preserve_positive_intervals_and_exact_final_duration(self):
        for duration,size,overlap in [(1e-9,1,0),(.0000007,.0000003,0),(1.0000001,1,0),(2.35,.3,.1)]:
            with self.subTest(duration=duration):
                spans = media.windows(duration,size,overlap)
                self.assertEqual(spans[0][0], 0)
                self.assertEqual(spans[-1][1], duration)
                for i,(start,end) in enumerate(spans):
                    self.assertTrue(0 <= start < end <= duration)
                    if i: self.assertLessEqual(start,spans[i-1][1])

    def test_dangling_download_output_flag_has_json_cli_error(self):
        with tempfile.TemporaryDirectory() as root:
            script = Path(root)/'download.sh'
            script.write_text('curl https://storage.googleapis.com/fixture -o\n')
            with self.assertRaisesRegex(ValueError,'missing a filename'):
                entries(script)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main(['download',str(script),'--list'])
            self.assertEqual(code,2)
            self.assertEqual(json.loads(stderr.getvalue())['type'],'ValueError')

    def test_vad_accepts_original_distribution_metadata(self):
        module = SimpleNamespace(Vad=lambda mode: object())
        with patch.dict('sys.modules', {'webrtcvad':module}), \
                patch('video_moment_retrieval.speech.version',side_effect=[PackageNotFoundError('wheels'),'2.0.10']):
            self.assertIn('2.0.10',WebRTCSpeechDetector().identity)

    def test_vad_without_distribution_metadata_uses_module_version_or_clear_error(self):
        module = SimpleNamespace(Vad=lambda mode: object(),__version__='2.0.custom')
        with patch.dict('sys.modules', {'webrtcvad':module}), \
                patch('video_moment_retrieval.speech.version',side_effect=PackageNotFoundError('fixture')):
            self.assertIn('2.0.custom',WebRTCSpeechDetector().identity)
            del module.__version__
            with self.assertRaisesRegex(RuntimeError,'Cannot determine'):
                WebRTCSpeechDetector()

    def test_atomic_json_syncs_contents_before_rename_and_directory_after(self):
        events = []
        original_sync, original_replace = os.fsync, os.replace
        def sync(fd):
            events.append('directory' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file')
            return original_sync(fd)
        def replace(*args):
            events.append('replace')
            return original_replace(*args)
        with tempfile.TemporaryDirectory() as root:
            target = Path(root)/'state.json'
            with patch('video_moment_retrieval.cache.os.fsync',side_effect=sync), \
                    patch('video_moment_retrieval.cache.os.replace',side_effect=replace):
                atomic_json(target,{'page':2})
            self.assertEqual(json.loads(target.read_text()),{'page':2})
            self.assertEqual(events,['file','replace','directory'] if os.name=='posix' else ['file','replace'])

    def test_failed_file_sync_preserves_previous_state_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root)/'state.json'
            target.write_text('{"page":1}')
            with patch('video_moment_retrieval.cache.os.fsync',side_effect=OSError('disk error')):
                with self.assertRaises(OSError):
                    atomic_json(target,{'page':2})
            self.assertEqual(json.loads(target.read_text()),{'page':1})
            self.assertEqual(list(Path(root).iterdir()),[target])


class RetrievalReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.store = Store(root/'index.sqlite')
        self.addCleanup(self.store.close)
        build_demo(self.store,root/'labels.json')
        self.backend = DemoBackend()
        self.engine = SearchEngine(self.store,ArtifactCache(root/'cache'),root/'verification',
                                   DemoEncoder(),self.backend,self.backend,synthetic=True)

    def test_one_sqlite_channel_failure_keeps_other_channels_and_marks_degraded(self):
        with patch.object(Store,'lexical',side_effect=sqlite3.OperationalError('database is locked')):
            response = self.engine.search('handcuffed',SearchConfig(verify=False,assess=False))
        self.assertTrue(response['results'])
        self.assertEqual(response['outcome'],'degraded')
        self.assertIn('lexical',response['retrieval']['channel_errors'])
        self.assertTrue(response['retrieval']['channel_hits']['dense'])

    def test_all_sqlite_channel_failures_have_explicit_failure(self):
        with patch.object(Store,'lexical',side_effect=sqlite3.OperationalError('locked')), \
                patch.object(Store,'structured',side_effect=sqlite3.OperationalError('locked')), \
                patch.object(Store,'semantic',side_effect=sqlite3.OperationalError('locked')):
            with self.assertRaisesRegex(RuntimeError,'All retrieval channels failed'):
                retrieve(self.store,DemoEncoder(),self.backend.plan('handcuffed'),10)

    def test_ranked_pagination_requires_snapshot_and_rejects_changed_index(self):
        config = SearchConfig(verify=False,assess=False,top_k=1)
        first = self.engine.search('handcuffed',config)
        with patch.object(self.backend,'plan',side_effect=AssertionError('must fail before planning')):
            with self.assertRaisesRegex(ValueError,'Pagination requires'):
                self.engine.search('handcuffed',SearchConfig(verify=False,offset=1))
        next_config = SearchConfig(verify=False,assess=False,top_k=1,offset=1,snapshot=first['pagination']['snapshot'])
        second = self.engine.search('handcuffed',next_config)
        self.assertNotEqual(first['results'][0]['record_id'],second['results'][0]['record_id'])
        video = self.store.video(first['results'][0]['video_id'])
        self.store.add_video(video['id'],video['path'],video['duration'],bool(video['has_audio']))
        with self.assertRaisesRegex(ValueError,'settings changed'):
            self.engine.search('handcuffed',next_config)

    def test_supported_prefix_match_has_explicit_partial_inspection(self):
        video_id = self.store.db.execute('SELECT id FROM videos LIMIT 1').fetchone()[0]
        record = Record('long',video_id,'window',0,80,'handcuffed',
                        [Evidence('long:e','visual',0,80,'fixture')])
        def verify(plan,candidate,prepared):
            return Verification([Verdict('supported','fixture match',1,2,['long:e'],
                [{'criterion':c,'status':'supported','evidence_ids':['long:e']} for c in plan.criteria])],True)
        with patch('video_moment_retrieval.search.retrieve',return_value=([Candidate(record,1,[])],{})), \
                patch.object(self.backend,'verify',side_effect=verify):
            result = self.engine.search('handcuffed',SearchConfig(assess=False,max_clip_seconds=5))
        self.assertTrue(result['results'])
        self.assertTrue(result['partial_inspection'])
        self.assertFalse(result['results'][0]['inspection_complete'])
        self.assertTrue(result['results'][0]['clip_inspection_complete'])
        self.assertFalse(result['pagination']['all_tasks_verified'])


class MediaCeilingTests(unittest.TestCase):
    def test_heights_above_the_720_ceiling_are_rejected(self):
        for bad in (721, 1080, 2160):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, '720'):
                    IndexConfig(height=bad)
                with self.assertRaisesRegex(ValueError, '720'):
                    SearchConfig(height=bad)
        self.assertEqual((IndexConfig().height, SearchConfig().height), (720, 720))

    def test_media_helpers_reject_heights_above_the_ceiling(self):
        with self.assertRaises(ValueError):
            media.frame('source', Path('/tmp/frame.png'), 0, 721)
        with self.assertRaises(ValueError):
            media.clip('source', Path('/tmp/clip.mp4'), 0, 1, 1080)
