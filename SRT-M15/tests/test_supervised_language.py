"""Deterministic tests: never access actual Internet or large model APIs."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from srt_m15.supervised_language import (
    Utf8ByteVocabulary, LanguageTrainConfig, ReviewedQAStore,
    LanguageLearningController, _pair_bytes, TOKENIZER_NAME,
)
from srt_m15.web_learning import WebKnowledgeStore
from srt_m15.chat import SRTChatEngine


def fake_note(n=1, title='人工智能'):
    return {
        'url': f'https://zh.wikipedia.org/wiki/{title}_{n}', 'title': title,
        'summary': '人工智能系统通过数据处理、数学模型以及验证来完成部分任务。' * 3,
        'topic': '人工智能', 'language': 'zh', 'fetched_at': '2026-10-23T12:00:00Z',
    }


class LanguageUpgradeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.sources = WebKnowledgeStore(self.root)
        self.qa = ReviewedQAStore(self.root, self.sources)

    def sample_ready(self):
        # Generate distinct short supervised Q/A with stable split.
        i = 0
        while True:
            i += 1
            q = f'你好{i}号'
            ans = f'我是实验模型{i}号'
            row = self.qa.add_draft(q, ans)
            self.qa.review(row['id'], True)
            train, val = self.qa.split(self.qa.approved())
            if len(train) >= 5 and len(val) >= 3:
                break
            if i > 60:
                self.fail('unable to produce deterministic validation examples')
        return train, val

    def test_unicode_byte_vocab_roundtrip_arbitrary_characters(self):
        vocab = Utf8ByteVocabulary()
        for text in ('你好！1+9=多少？', '📚🌏天気AI', '\u200b中文！'):
            self.assertEqual(vocab.decode(vocab.encode(text)), text)
        self.assertEqual(len(set(vocab.encode('abc'))), 3)

    def test_validate_length_and_resource_limits(self):
        with self.assertRaises(ValueError):
            _pair_bytes('这是一个非常复杂的问题' * 6, '非常详细的答案' * 6)
        for request in ({'steps': 999}, {'steps': 1.2}, {'max_runtime_seconds': 1},
                        {'learning_rate': 2}, {'unknown_flag': True}, {'steps': True}):
            with self.subTest(request=request), self.assertRaises(ValueError):
                LanguageTrainConfig.from_request(request)

    def test_unreviewed_web_source_cannot_generate_training_labels(self):
        self.assertEqual(self.sources.save([fake_note()]), 1)
        note_id = self.sources.recent()[0]['id']
        with self.assertRaises(ValueError):
            self.qa.suggest_from_source(note_id)
        with self.assertRaises(ValueError):
            self.qa.add_draft('什么是人工智能', '人工智能是一个技术领域', note_id)
        self.sources.review(note_id, True)
        suggestion = self.qa.suggest_from_source(note_id)
        self.assertEqual(suggestion['note_id'], note_id)
        self.assertIn('草稿', suggestion['notice'])

    def test_qa_needs_separate_manual_approval_and_keeps_source(self):
        self.sources.save([fake_note()])
        note_id = self.sources.recent()[0]['id']
        self.sources.review(note_id, True)
        drafted = self.qa.add_draft('人工智能是什么', '人工智能是一个研究领域', note_id)
        self.assertEqual(self.qa.status()['approved'], 0)
        self.assertEqual(self.qa.approved(), [])
        self.qa.review(drafted['id'], True)
        row = self.qa.approved()[0]
        self.assertEqual(row['source_note_id'], note_id)
        self.assertIn('wikipedia.org', row['source_url'])
        with self.assertRaises(ValueError):
            self.qa.review(drafted['id'], True)
        self.qa.review(drafted['id'], False)
        self.assertEqual(self.qa.approved(), [])

    def test_edit_requires_new_manual_approval(self):
        row = self.qa.add_draft('你是谁', '我是实验模型')
        self.qa.review(row['id'], True)
        updated = self.qa.revise(row['id'], '你是谁', '我是一个小模型')
        self.assertEqual(updated['status'], 'draft')
        self.assertEqual(self.qa.approved(), [])
        self.qa.review(row['id'], True)
        self.assertEqual(self.qa.approved()[0]['answer'], '我是一个小模型')

    def test_rejects_retracted_source_on_approval(self):
        self.sources.save([fake_note()])
        note_id = self.sources.recent()[0]['id']
        self.sources.review(note_id, True)
        drafted = self.qa.add_draft('人工智能是什么', '人工智能是一个研究领域', note_id)
        self.sources.review(note_id, False)
        with self.assertRaises(ValueError):
            self.qa.review(drafted['id'], True)
        self.assertEqual(self.qa.approved(), [])
        self.qa.review(drafted['id'], False)
        self.assertEqual(self.qa.approved(), [])

    def test_stable_holdout_grouped_by_source(self):
        self.sources.save([fake_note(1), fake_note(2)])
        ids = [x['id'] for x in self.sources.recent()]
        for note_id in ids:
            self.sources.review(note_id, True)
        for note_id in ids:
            for j in range(2):
                row = self.qa.add_draft(f'请介绍{note_id}概念{j}', '这是一个需要独立验证的测试答案', note_id)
                self.qa.review(row['id'], True)
        training, holdout = self.qa.split(self.qa.approved())
        train_sources = {x['source_note_id'] for x in training}
        validation_sources = {x['source_note_id'] for x in holdout}
        self.assertFalse(train_sources & validation_sources)
        self.assertEqual(len(training)+len(holdout), 4)

    def test_no_training_before_sufficient_review(self):
        controller = LanguageLearningController(self.root, self.qa)
        with self.assertRaises(ValueError):
            controller.start({'steps': 2})
        row = self.qa.add_draft('人工智能是什么', '这是一个人工智能实验')
        with self.assertRaises(ValueError):
            controller.start({'steps': 2})
        self.assertEqual(self.qa.status()['draft'], 1)

    def test_training_in_new_byte_checkpoint_without_overwrite(self):
        self.sample_ready()
        old_file = self.root / 'checkpoints/srt_m15_lm.pt'
        old_file.parent.mkdir(parents=True, exist_ok=True)
        old_file.write_bytes(b'original-language-weights-test-marker')
        controller = LanguageLearningController(self.root, self.qa)
        controller.start({'steps': 3, 'max_runtime_seconds': 30})
        controller.worker.join(timeout=45)
        self.assertFalse(controller.status()['running'])
        self.assertIn(controller.status()['last_result']['state'], ('rejected', 'accepted'))
        self.assertEqual(old_file.read_bytes(), b'original-language-weights-test-marker')
        if controller.checkpoint.is_file():
            loaded = torch.load(controller.checkpoint, map_location='cpu', weights_only=True)
            self.assertEqual(loaded['tokenizer'], TOKENIZER_NAME)

    def test_candidate_accepted_after_positive_metrics_then_rollback(self):
        self.sample_ready()
        controller = LanguageLearningController(self.root, self.qa)
        # Metrics come from a deterministic test double. Actual training
        # and filesystem persistence still execute.
        with patch('srt_m15.supervised_language._evaluate', side_effect=[4., 3.9, 3., 3.85]):
            controller.start({'steps': 1, 'max_runtime_seconds': 30})
            controller.worker.join(timeout=30)
        self.assertEqual(controller.status()['last_result']['state'], 'accepted')
        self.assertTrue(controller.checkpoint.is_file())
        engine = SRTChatEngine(self.root)
        answer = engine.reply('你好，机器人！')
        self.assertEqual(answer.mode, 'text')
        controller2 = LanguageLearningController(self.root, self.qa, engine)
        self.assertTrue(controller2.status()['trained_checkpoint'])
        self.assertEqual(controller2.status()['accepted_updates'], 1)
        controller2.rollback()
        self.assertFalse(controller2.checkpoint.exists())
        self.assertEqual(controller2.status()['last_result']['state'], 'rolled_back')

    def test_candidate_rejection_never_overwrites_current(self):
        self.sample_ready()
        controller = LanguageLearningController(self.root, self.qa)
        controller.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        controller.checkpoint.write_bytes(b'do-not-overwrite')
        # In this case a corrupted previous checkpoint is caught; rejection
        # leaves original file untouched even on exception.
        controller.start({'steps': 1, 'max_runtime_seconds': 30})
        controller.worker.join(timeout=30)
        self.assertEqual(controller.checkpoint.read_bytes(), b'do-not-overwrite')
        self.assertEqual(controller.status()['last_result']['state'], 'error')

    def test_holdout_regression_rejects_candidate(self):
        self.sample_ready()
        controller = LanguageLearningController(self.root, self.qa)
        with patch('srt_m15.supervised_language._evaluate', side_effect=[4., 3.9, 2., 4.3]):
            controller.start({'steps': 1, 'max_runtime_seconds': 30})
            controller.worker.join(timeout=30)
        self.assertEqual(controller.status()['last_result']['state'], 'rejected')
        self.assertFalse(controller.checkpoint.exists())

    def test_pause_no_automatic_restart(self):
        self.sample_ready()
        controller = LanguageLearningController(self.root, self.qa)
        controller.stop()
        self.assertFalse(controller.status()['running'])
        controller2 = LanguageLearningController(self.root, self.qa)
        self.assertFalse(controller2.status()['running'])


if __name__ == '__main__':
    unittest.main()
