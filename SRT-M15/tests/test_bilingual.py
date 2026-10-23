"""Regression checks for bilingual SRT-M15 without external network requests."""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from srt_m15.chat import SRTChatEngine, detect_text_language
from srt_m15.supervised_language import (
    _pair_bytes, ReviewedQAStore, LanguageLearningController,
    LanguageTrainConfig, SAMPLE_BILINGUAL_QA,
)
from srt_m15.web_learning import WebKnowledgeStore


class BilingualLearningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.sources = WebKnowledgeStore(self.root)
        self.store = ReviewedQAStore(self.root, self.sources)

    def approved_pairs(self):
        imported = self.store.add_bilingual_demo_drafts()
        self.assertEqual(imported['new_drafts'], 48)
        for row in self.store.recent():
            self.store.review(row['id'], True)
        return self.store.split(self.store.approved())

    def test_utf8_and_parallel_prompts(self):
        chinese, ans_zh = _pair_bytes('你好', '你好。', 'zh')
        english, ans_en = _pair_bytes('Hello', 'Hi.', 'en')
        self.assertEqual(bytes(chinese).decode(), '问：你好\n答：')
        self.assertEqual(bytes(english).decode(), 'Q: Hello\nA: ')
        self.assertEqual(bytes(ans_en).decode(), 'Hi.\n')
        self.assertEqual(bytes(ans_zh).decode(), '你好。\n')
        for bad in ('fr', 'both', 'auto'):
            with self.assertRaises(ValueError):
                _pair_bytes('Bonjour', 'Salut', bad)

    def test_demo_import_is_unapproved_idempotent_and_pair_isolation(self):
        self.assertEqual(len(SAMPLE_BILINGUAL_QA), 24)
        self.assertEqual(self.store.add_bilingual_demo_drafts()['new_drafts'], 48)
        self.assertEqual(self.store.add_bilingual_demo_drafts()['new_drafts'], 0)
        self.assertEqual(self.store.status()['approved'], 0)
        train, val = self.store.split(self.store.recent())
        self.assertEqual(len(train), 40)
        self.assertEqual(len(val), 8)
        self.assertEqual(self.store._counts(train), {'zh': 20, 'en': 20})
        self.assertEqual(self.store._counts(val), {'zh': 4, 'en': 4})
        for items in (train, val):
            counts = {}
            for item in items:
                counts[item['group_key']] = counts.get(item['group_key'], 0) + 1
            self.assertTrue(all(n == 2 for n in counts.values()))
        self.assertFalse({x['group_key'] for x in train} & {x['group_key'] for x in val})

    def test_bilingual_training_requires_both_languages(self):
        for row in self.store.add_bilingual_demo_drafts(),:
            self.assertTrue(row['all_require_manual_approval'])
        for row in self.store.recent():
            if row['language'] == 'zh':
                self.store.review(row['id'], True)
        with self.assertRaisesRegex(ValueError, '双语训练'):
            LanguageLearningController(self.root, self.store).start({'language_mode': 'bilingual'})
        with self.assertRaises(ValueError):
            LanguageTrainConfig.from_request({'language_mode': 'unknown'})

    def test_validation_regression_in_one_language_rejects_update(self):
        self.approved_pairs()
        learner = LanguageLearningController(self.root, self.store)
        # Score calls: train zh, train en, val zh, val en, then same after train.
        with patch('srt_m15.supervised_language._evaluate',
                   side_effect=[5., 5., 4., 4., 3.5, 3.0, 4.2, 3.8]):
            learner.start({'steps': 2, 'max_runtime_seconds': 35,
                           'language_mode': 'bilingual'})
            learner.worker.join(timeout=35)
        self.assertFalse(learner.status()['running'])
        self.assertEqual(learner.last_result['state'], 'rejected')
        self.assertFalse(learner.checkpoint.exists())
        self.assertGreater(learner.last_result['per_language']['zh']['validation_after'],
                           learner.last_result['per_language']['zh']['validation_before'])

    def test_bilingual_candidate_accepted_and_rollback(self):
        self.approved_pairs()
        learner = LanguageLearningController(self.root, self.store)
        orig = self.root / 'checkpoints/srt_m15_lm.pt'
        orig.parent.mkdir(parents=True, exist_ok=True)
        orig.write_bytes(b'preserve_original_english_and_chinese')
        with patch('srt_m15.supervised_language._evaluate',
                   side_effect=[5., 5., 4., 4., 3.5, 3.2, 3.9, 3.8]):
            learner.start({'steps': 2, 'max_runtime_seconds': 35,
                           'language_mode': 'bilingual'})
            learner.worker.join(timeout=35)
        self.assertEqual(learner.last_result['state'], 'accepted')
        self.assertEqual(learner.last_result['language_mode'], 'bilingual')
        self.assertEqual(orig.read_bytes(), b'preserve_original_english_and_chinese')
        checkpoint = torch.load(learner.checkpoint, map_location='cpu', weights_only=True)
        self.assertEqual(checkpoint['language_mode'], 'bilingual')
        learner.rollback()
        self.assertFalse(learner.checkpoint.exists())

    def test_db_legacy_zh_migration_preserves_rows(self):
        old_db = self.root / 'language_learning/reviewed_qa.sqlite3'
        old_db.parent.mkdir(parents=True, exist_ok=True)
        old_db.unlink()
        with sqlite3.connect(old_db) as con:
            con.execute('''CREATE TABLE qa (id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL UNIQUE, answer TEXT NOT NULL,
                source_note_id INTEGER, source_url TEXT NOT NULL,
                source_title TEXT NOT NULL, status TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL)''')
            con.execute('''INSERT INTO qa (question, answer, source_url, source_title,
                status, created_at, updated_at) VALUES (?, ?, '', '', 'approved', '', '')''',
                        ('你好', '你好。'))
        restored = ReviewedQAStore(self.root, self.sources)
        self.assertEqual(restored.approved()[0]['language'], 'zh')
        self.assertEqual(restored.approved()[0]['group_key'], '')
        self.assertEqual(restored.status()['approved'], 1)

    def test_english_source_draft_has_correct_language(self):
        sample = {
            'url': 'https://en.wikipedia.org/wiki/Neural_network',
            'title': 'Neural network', 'summary': 'A neural network is a computational model. ' * 3,
            'topic': 'Neural network', 'language': 'en', 'fetched_at': '2026-10-23T10:00:00Z'
        }
        self.sources.save([sample])
        note_id = self.sources.recent()[0]['id']
        self.sources.review(note_id, True)
        draft = self.store.suggest_from_source(note_id)
        self.assertEqual(draft['language'], 'en')
        self.assertTrue(draft['question'].startswith('Briefly describe'))
        with self.assertRaises(ValueError):
            self.store.add_draft('What is a network?', 'A network is a graph.', note_id, language='zh')
        entry = self.store.add_draft('What is a network?', 'A network is a graph.', note_id, language='en')
        self.assertEqual(entry['language'], 'en')

    def test_language_detection_and_manual_override(self):
        self.assertEqual(detect_text_language('Hello, how are you?'), 'en')
        self.assertEqual(detect_text_language('你好 English'), 'zh')
        self.assertEqual(detect_text_language('Bonjour', 'zh'), 'zh')
        with self.assertRaises(ValueError):
            detect_text_language('Hello', 'fr')
        self.assertEqual(SRTChatEngine(self.root).reply('Hi', language='en').mode, 'notice')

    def test_same_web_source_and_different_pair_groups_do_not_cross_split(self):
        self.sources.save([{
            'url':'https://en.wikipedia.org/wiki/Natural_language_processing',
            'title':'Natural language processing', 'summary':'A field of machine learning. '*4,
            'topic':'NLP','language':'en','fetched_at':'2026-10-23T10:00:00Z'
        }])
        note_id=self.sources.recent()[0]['id'];self.sources.review(note_id, True)
        items=[]
        for i in range(4):
            items.append({'question':f'What is NLP {i}?','answer':'A field of machine learning.',
                          'language':'en','source_note_id':note_id,'group_key':f'pair-{i}'})
        train, val = ReviewedQAStore.split(items)
        self.assertIn(len(train), (0,4))
        self.assertIn(len(val), (0,4))


if __name__ == '__main__':
    unittest.main()
