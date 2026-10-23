"""Bounded supervised language learning for SRT-M15.

A small byte-token language model, trained ONLY on individually human-approved,
short Q/A pairs. Wikipedia excerpts remain reviewable SOURCE MATERIAL, not truth
labels. Split is stable and by source document to reduce hold-out leakage.
All candidate model changes require training improvement, held-out loss no worse,
finite weights, and a checkpoint read-back before atomic activation.

No browser fetches or external model APIs occur here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import copy
import json
import math
import os
import random
import shutil
import sqlite3
import tempfile
import threading
import time
import unicodedata

import torch
import torch.nn.functional as F

from .language import SRTM15LanguageModel
from .model import ModelConfig


TOKENIZER_NAME = 'utf8-bytes-v1'
MAX_SEQUENCE = 160
MAX_QA_COUNT = 160
LANGUAGES = ('zh', 'en')


class Utf8ByteVocabulary:
    """Fixed vocabulary: each input UTF-8 byte is a model token, no OOV."""
    def encode(self, text: str) -> list[int]:
        return list(text.encode('utf-8'))

    def decode(self, ids) -> str:
        return bytes(int(x) for x in ids).decode('utf-8', errors='replace')


@dataclass(frozen=True)
class LanguageTrainConfig:
    steps: int = 30
    max_runtime_seconds: int = 100
    learning_rate: float = 0.001
    language_mode: str = 'auto'  # auto, zh, en, bilingual

    @classmethod
    def from_request(cls, options):
        if not isinstance(options, dict) or set(options) - set(cls.__dataclass_fields__):
            raise ValueError('训练设置必须是指定字段的 JSON 对象')
        cfg = cls(**options)
        if isinstance(cfg.steps, bool) or not isinstance(cfg.steps, int) or not 1 <= cfg.steps <= 100:
            raise ValueError('训练步数允许 1～100')
        if isinstance(cfg.max_runtime_seconds, bool) or not isinstance(cfg.max_runtime_seconds, int) or not 15 <= cfg.max_runtime_seconds <= 240:
            raise ValueError('训练时长允许 15～240 秒')
        if isinstance(cfg.learning_rate, bool) or not isinstance(cfg.learning_rate, (int, float)) or not math.isfinite(cfg.learning_rate) or not 0.00005 <= cfg.learning_rate <= 0.002:
            raise ValueError('学习率允许 0.00005～0.002')
        if cfg.language_mode not in ('auto', 'zh', 'en', 'bilingual'):
            raise ValueError('训练模式须为 auto、zh、en 或 bilingual')
        return cfg


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _atomic_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _valid_text(s, label, maximum):
    if not isinstance(s, str):
        raise ValueError(f'{label}必须是文本')
    s = unicodedata.normalize('NFKC', s).strip()
    if not 2 <= len(s) <= maximum or any(ord(c) < 32 or ord(c) == 127 for c in s):
        raise ValueError(f'{label}需要 2～{maximum} 字符，不含换行或控制字符')
    return s


def _language(language):
    if language not in LANGUAGES:
        raise ValueError('问答语言只允许 zh 或 en')
    return language


def _group_key(key):
    if key is None or key == '':
        return ''
    if not isinstance(key, str) or len(key) > 64 or not key.isascii() or any(
        c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in key
    ):
        raise ValueError('平行问答分组只允许 1～64 位英文、数字、连字符或下划线')
    return key


def prompt_prefix(question, language='zh'):
    """Keep Chinese prefix backward-compatible; use Q/A tags in English."""
    _language(language)
    return ('问：' + question + '\n答：') if language == 'zh' else ('Q: ' + question + '\nA: ')


def _pair_bytes(question, answer, language='zh'):
    """Teacher-forced answer-only loss. No silent truncation of the answer."""
    tok = Utf8ByteVocabulary()
    prefix = tok.encode(prompt_prefix(question, language))
    ans = tok.encode(answer + '\n')
    if len(prefix) + len(ans) - 1 > MAX_SEQUENCE:
        raise ValueError(f'问答合计 UTF-8 长度超过 {MAX_SEQUENCE} 字节，请缩短内容')
    return prefix, ans


SAMPLE_QA = [
    ('你是谁', '我是SRT-M15实验模型。'),
    ('你能做什么', '我能尝试个位数加法和有限文本续写。'),
    ('你能联网吗', '联网资料收集需要先获得用户授权。'),
    ('你的回答可靠吗', '不能保证，需要通过可靠来源核实。'),
    ('什么是参数', '参数是训练时调整的数值。'),
    ('什么是训练', '训练通过优化参数来减少误差。'),
    ('什么是推理', '推理是利用当前参数计算预测。'),
    ('什么是验证集', '验证集检查模型更新的表现。'),
    ('什么是测试集', '测试集用来评估泛化表现。'),
    ('为什么要注明来源', '来源可以帮助检查信息的可靠性。'),
    ('什么是词表', '词表帮助将文字映射为模型输入。'),
    ('你能独立写代码吗', '目前不能可靠完成开放式编程任务。'),
    ('记忆和学习相同吗', '不相同。保存信息不等于更新参数。'),
    ('什么是神经网络', '神经网络是具有可训练参数的计算系统。'),
    ('联网后会自动训练吗', '不会。问答还需要人工确认。'),
    ('什么是回滚', '回滚是恢复之前保存的模型版本。'),
    ('你需要人类监督吗', '训练答案和更新结果需要独立验证。'),
    ('什么是错误反馈', '错误反馈指出预测与目标的差异。'),
    ('你的知识完整吗', '不完整。我仍是很小的研究原型。'),
    ('你会保存来源吗', '联网资料会保存来源和审核状态。'),
]


# Paired first-party examples. Every zh/en translation pair uses the SAME split
# group, so that training on the Chinese phrasing cannot leak into English val.
# All are imported as DRAFTS and require human approval, including these demos.
SAMPLE_BILINGUAL_QA = [
    ('请简短介绍自己', '我是SRT-M15实验模型。', 'Introduce yourself briefly.', 'I am the SRT-M15 experimental model.'),
    ('你正在学习什么语言', '我正在学习中文和英语。', 'Which languages are you learning?', 'I am learning Chinese and English.'),
    ('你能可靠回答所有问题吗', '不能，我仍是小型实验模型。', 'Can you answer every question reliably?', 'No. I am still a small research model.'),
    ('人工智能研究什么', '研究让机器完成部分智能任务。', 'What does AI study?', 'It studies machines performing intelligent tasks.'),
    ('什么是数据集', '数据集是用于学习或评估的数据集合。', 'What is a dataset?', 'A dataset is a set of training or testing examples.'),
    ('什么是参数更新', '参数更新是根据训练损失调整数值。', 'What is a parameter update?', 'It changes numeric weights using a training signal.'),
    ('模型如何学习', '通过训练样本和优化算法更新参数。', 'How does a model learn?', 'It updates parameters using examples and an optimizer.'),
    ('什么是预测', '预测是模型根据输入计算结果。', 'What is a prediction?', 'It is an output computed from an input.'),
    ('什么是验证器', '验证器使用独立规则检查结果。', 'What is a verifier?', 'A verifier checks results using independent rules.'),
    ('错误答案能直接训练吗', '不能，应先获得可靠的正确标签。', 'Can wrong answers train the model directly?', 'No. Training requires a reliable target label.'),
    ('模型为什么会遗忘', '学习新任务可能干扰旧任务。', 'Why might a model forget?', 'Training on new tasks may interfere with old tasks.'),
    ('什么是回滚功能', '回滚可以恢复之前保存的权重。', 'What does rollback mean?', 'Rollback restores a previous saved checkpoint.'),
    ('什么是训练集', '训练集用于计算梯度和更新模型。', 'What is a training set?', 'Training data provides examples to update weights.'),
    ('什么是检验集', '检验集用于检查更新是否退步。', 'What is a validation set?', 'A validation set detects regressions after training.'),
    ('为什么保留测试集', '测试集用于最后评估泛化能力。', 'Why keep a test set separate?', 'It is reserved for a final generalization check.'),
    ('什么是过拟合', '过拟合是记住训练样本却难以泛化。', 'What is overfitting?', 'It means fitting training data but generalizing poorly.'),
    ('什么是词元', '词元是模型处理的离散输入单位。', 'What is a token?', 'A token is a discrete unit processed by a model.'),
    ('什么是字节词表', '字节词表使用UTF-8编码文字。', 'What is a byte vocabulary?', 'It encodes text as UTF-8 bytes.'),
    ('模型有记忆吗', '可以保存资料，但记忆不等于训练。', 'Can a model store memories?', 'It can store notes, but memory is not training.'),
    ('你可以上网吗', '需要明确授权才能联网检索。', 'Can you access the internet?', 'Internet research requires explicit permission.'),
    ('联网信息都是真的吗', '不一定，必须检查来源和证据。', 'Is everything found online true?', 'No. Sources and evidence must be checked.'),
    ('为什么审核训练资料', '审核可以减少错误信息进入训练。', 'Why review training examples?', 'Review helps reduce erroneous training labels.'),
    ('如何停止训练', '可以点击暂停并保留旧模型。', 'How can training be stopped?', 'Use pause and retain the previous model.'),
    ('你支持简单加法吗', '数学网络可以实验个位数加法。', 'Do you support simple addition?', 'The toy arithmetic model tests single-digit addition.'),
]


class ReviewedQAStore:
    """Separate from web notes. Reviewed ARTICLE != approved MODEL LABEL."""

    def __init__(self, root: Path, source_store=None):
        self.root = Path(root)
        self.path = self.root / 'language_learning' / 'reviewed_qa.sqlite3'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.source_store = source_store
        with self._db() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS qa (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL UNIQUE,
                answer TEXT NOT NULL,
                source_note_id INTEGER,
                source_url TEXT NOT NULL,
                source_title TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'draft' CHECK(status IN ('draft','approved','rejected')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )''')
            columns = {row['name'] for row in db.execute('PRAGMA table_info(qa)').fetchall()}
            # Preserve all existing drafts/approvals when upgrading older DBs.
            if 'language' not in columns:
                db.execute("ALTER TABLE qa ADD COLUMN language TEXT NOT NULL DEFAULT 'zh'")
            if 'group_key' not in columns:
                db.execute("ALTER TABLE qa ADD COLUMN group_key TEXT NOT NULL DEFAULT ''")

    @contextmanager
    def _db(self):
        con = sqlite3.connect(self.path, timeout=5)
        con.row_factory = sqlite3.Row
        try:
            with con:
                yield con
        finally:
            con.close()

    def suggest_from_source(self, note_id: int):
        if not isinstance(note_id, int) or isinstance(note_id, bool) or note_id <= 0:
            raise ValueError('资料 ID 必须是正整数')
        if self.source_store is None:
            raise ValueError('尚未启用网页资料库')
        # Use only the official approved-note accessor, never a public URL fetch.
        note = self.source_store.approved_note(note_id)
        if note is None:
            raise ValueError('资料必须先在联网知识面板人工标记审核')
        lang = _language(note.get('language', 'zh'))
        title = note['title'][:28]
        question = (('请简要介绍' + title) if lang == 'zh' else ('Briefly describe ' + title))[:60]
        draft = note['summary'].strip()[:55].rsplit('。' if lang == 'zh' else '.', 1)[0].strip()
        if not draft:
            draft = note['summary'].strip()[:38]
        return {'question': question, 'answer': draft, 'language': lang, 'note_id': note_id,
                'source_title': note['title'], 'source_url': note['url'],
                'notice': '仅是草稿，不是已验证答案。请阅读来源、修改答案，再分别点击保存与人工批准。'}

    def add_demo_drafts(self):
        """First-party project facts; explicit import; ALL rows remain drafts."""
        created = 0
        for q, a in SAMPLE_QA:
            try:
                self.add_draft(q, a)
                created += 1
            except ValueError as exc:
                if '已经存在' not in str(exc):
                    raise
        return {'new_drafts': created, 'all_require_manual_approval': True}

    def add_bilingual_demo_drafts(self):
        """24 matched zh/en pairs; not approved until individually reviewed."""
        created = 0
        for idx, (zq, za, eq, ea) in enumerate(SAMPLE_BILINGUAL_QA):
            for lang, question, answer in (('zh', zq, za), ('en', eq, ea)):
                try:
                    self.add_draft(question, answer, language=lang, group_key=f'demo-{idx:02d}')
                    created += 1
                except ValueError as exc:
                    if '已经存在' not in str(exc):
                        raise
        return {'new_drafts': created, 'pairs': len(SAMPLE_BILINGUAL_QA),
                'all_require_manual_approval': True}

    def add_draft(self, question, answer, note_id=None, language='zh', group_key=None):
        question = _valid_text(question, '问题', 60)
        answer = _valid_text(answer, '答案', 95)
        language = _language(language)
        group_key = _group_key(group_key)
        _pair_bytes(question, answer, language)
        source_url, source_title = '', '用户人工编写'
        if note_id is not None:
            if isinstance(note_id, bool) or not isinstance(note_id, int) or note_id < 1:
                raise ValueError('无效资料 ID')
            if self.source_store is None:
                raise ValueError('来源资料不可用')
            note = self.source_store.approved_note(note_id)
            if note is None:
                raise ValueError('只有人工已审核的来源才能建立训练问答')
            source_url, source_title = note['url'], note['title']
            if note.get('language', 'zh') != language:
                raise ValueError('来源资料语言与问答语言不一致，翻译数据请手动建立并核验')
        with self._db() as db:
            if db.execute('SELECT COUNT(*) FROM qa').fetchone()[0] >= MAX_QA_COUNT:
                raise ValueError('已达到问答数量上限')
            try:
                cur = db.execute('''INSERT INTO qa(question,answer,source_note_id,source_url,
                        source_title,status,created_at,updated_at,language,group_key)
                        VALUES(?,?,?,?,?,'draft',?,?,?,?)''',
                        (question, answer, note_id, source_url, source_title, _now(), _now(),
                         language, group_key))
            except sqlite3.IntegrityError as exc:
                raise ValueError('同一问题已经存在，请先审核现有条目') from exc
            return {'id': cur.lastrowid, 'status': 'draft', 'language': language}

    def revise(self, qa_id, question, answer, language=None, group_key=None):
        if isinstance(qa_id, bool) or not isinstance(qa_id, int) or qa_id < 1:
            raise ValueError('无效问答 ID')
        question = _valid_text(question, '问题', 60)
        answer = _valid_text(answer, '答案', 95)
        with self._db() as db:
            old = db.execute('SELECT * FROM qa WHERE id=?', (qa_id,)).fetchone()
            if old is None or old['status'] == 'rejected':
                raise ValueError('问答不存在或已被拒绝')
            language = _language(old['language'] if language is None else language)
            group_key = _group_key(old['group_key'] if group_key is None else group_key)
            _pair_bytes(question, answer, language)
            if old['source_note_id'] is not None and (
                self.source_store is None or self.source_store.approved_note(old['source_note_id']) is None
            ):
                raise ValueError('来源已经失去人工审核资格')
            if old['source_note_id'] is not None and (
                self.source_store.approved_note(old['source_note_id']).get('language', 'zh') != language
            ):
                raise ValueError('来源语言与训练标签不一致')
            try:
                db.execute("UPDATE qa SET question=?,answer=?,language=?,group_key=?,status='draft',updated_at=? WHERE id=?",
                           (question, answer, language, group_key, _now(), qa_id))
            except sqlite3.IntegrityError as exc:
                raise ValueError('同一问题已经存在') from exc
        return {'id': qa_id, 'status': 'draft',
                'notice': '内容已修改，须重新人工批准才会进入训练池'}

    def review(self, qa_id, approved: bool):
        if isinstance(qa_id, bool) or not isinstance(qa_id, int) or qa_id < 1 or not isinstance(approved, bool):
            raise ValueError('无效问答审核请求')
        with self._db() as db:
            row = db.execute('SELECT * FROM qa WHERE id=?', (qa_id,)).fetchone()
            if row is None:
                raise ValueError('没有找到问答草稿')
            if row['status'] == 'rejected' or (row['status'] == 'approved' and approved):
                raise ValueError('这条问答已经完成审核')
            if approved:
                _pair_bytes(row['question'], row['answer'], row['language'])
                if row['source_note_id'] is not None and (
                    self.source_store is None or self.source_store.approved_note(row['source_note_id']) is None
                ):
                    raise ValueError('来源已失去审核资格')
            status = 'approved' if approved else 'rejected'
            db.execute('UPDATE qa SET status=?,updated_at=? WHERE id=?', (status, _now(), qa_id))
        return {'id': qa_id, 'status': status}

    def approved(self):
        with self._db() as db:
            rows = db.execute("SELECT * FROM qa WHERE status='approved' ORDER BY id").fetchall()
        return [dict(r) for r in rows if r['source_note_id'] is None or
                (self.source_store is not None and self.source_store.approved_note(r['source_note_id']) is not None)]

    def recent(self, limit=MAX_QA_COUNT):
        with self._db() as db:
            rows = db.execute('SELECT * FROM qa ORDER BY id DESC LIMIT ?',
                              (min(max(limit, 1), MAX_QA_COUNT),)).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def split(items):
        """Group every source document AND each cross-language translation pair.

        Connected groups prevent split leakage even if multiple pair keys share
        one Wikipedia note. The grouping is reproducible for unchanged data.
        """
        parents = list(range(len(items)))

        def root(i):
            while parents[i] != i:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i

        def unite(a, b):
            x, y = root(a), root(b)
            if x != y:
                parents[max(x, y)] = min(x, y)

        key_owners = {}
        def keys(item):
            result = []
            if item.get('source_note_id') is not None:
                result.append('web-note-' + str(item['source_note_id']))
            if item.get('group_key'):
                result.append('parallel-' + item['group_key'])
            if not result:
                # Retain the original key for already-trained legacy Q/A data;
                # otherwise upgrading would reshuffle the old validation pool.
                result = ['manual-' + item['question']]
            return result

        for index, item in enumerate(items):
            for key in keys(item):
                if key in key_owners:
                    unite(index, key_owners[key])
                key_owners[key] = index
        connected = {}
        for index, item in enumerate(items):
            connected.setdefault(root(index), []).append(item)

        training, holdout = [], []
        for members in connected.values():
            digest = int.from_bytes(sha256(min(k for item in members for k in keys(item)).encode('utf-8')).digest()[:8], 'big')
            (holdout if digest % 5 == 0 else training).extend(members)
        return training, holdout

    @staticmethod
    def _counts(items):
        return {language: sum(row.get('language', 'zh') == language for row in items)
                for language in LANGUAGES}

    def status(self):
        with self._db() as db:
            rows = db.execute('SELECT status,COUNT(*) AS n FROM qa GROUP BY status').fetchall()
        counts = {r['status']: r['n'] for r in rows}
        approved = self.approved()
        train, holdout = self.split(approved)
        return {'total': sum(counts.values()), 'draft': counts.get('draft', 0),
                'approved': counts.get('approved', 0), 'rejected': counts.get('rejected', 0),
                'training_samples': len(train), 'validation_samples': len(holdout),
                'approved_by_language': self._counts(approved),
                'train_by_language': self._counts(train),
                'holdout_by_language': self._counts(holdout),
                'recent': self.recent()}


def _loss_for_pair(model, sample):
    prefix, answer = _pair_bytes(sample['question'], sample['answer'], sample.get('language', 'zh'))
    seq = torch.tensor(prefix + answer, dtype=torch.long)
    # Input t predicts target t+1. Only the answer targets count.
    result = model(seq[:-1].unsqueeze(0))['logits']
    answer_logits = result[:, len(prefix)-1:, :].reshape(-1, 256)
    target = seq[len(prefix):]
    return F.cross_entropy(answer_logits, target)


def _evaluate(model, examples, stop_event=None):
    model.eval()
    losses = []
    with torch.inference_mode():
        for item in examples[:8]:
            if stop_event is not None and stop_event.is_set():
                raise InterruptedError('训练已被暂停')
            losses.append(float(_loss_for_pair(model, item)))
    return sum(losses) / len(losses)


class LanguageLearningController:
    """Explicit training on approved Q/A, never automated web-to-weight writes."""

    def __init__(self, root, qa_store: ReviewedQAStore, engine=None, engine_lock=None):
        self.root = Path(root)
        self.store = qa_store
        self.engine = engine
        self.engine_lock = engine_lock or threading.RLock()
        self.dir = self.root / 'language_learning'
        self.checkpoint = self.root / 'checkpoints/srt_m15_byte_qa.pt'
        self.previous = self.root / 'checkpoints/srt_m15_byte_qa_previous.pt'
        self.state_file = self.dir / 'training_state.json'
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.worker = None
        self.running = False
        self.attempts = 0
        self.accepted_updates = 0
        self.last_result = {'state': 'not_started'}
        try:
            state = json.loads(self.state_file.read_text(encoding='utf-8'))
            self.attempts = int(state.get('attempts', 0))
            self.accepted_updates = int(state.get('accepted_updates', 0))
            self.last_result = state.get('last_result', {'state': 'not_started'})
        except (FileNotFoundError, ValueError, OSError, TypeError):
            pass

    def _save(self):
        _atomic_json(self.state_file, {'attempts': self.attempts,
                     'accepted_updates': self.accepted_updates, 'last_result': self.last_result,
                     'last_update': _now()})

    def status(self):
        with self.lock:
            return {**self.store.status(), 'running': self.running, 'attempts': self.attempts,
                    'accepted_updates': self.accepted_updates, 'last_result': self.last_result,
                    'trained_checkpoint': self.checkpoint.is_file(),
                    'can_rollback': self.checkpoint.is_file()}

    def start(self, options):
        cfg = LanguageTrainConfig.from_request(options)
        with self.lock:
            if self.running:
                raise ValueError('语言训练已经在运行中')
            train, holdout = self.store.split(self.store.approved())
            if cfg.language_mode == 'bilingual':
                selected = LANGUAGES
            elif cfg.language_mode == 'auto':
                selected = tuple(lang for lang in LANGUAGES
                                 if any(x['language'] == lang for x in train + holdout))
            else:
                selected = (cfg.language_mode,)
                train = [x for x in train if x['language'] in selected]
                holdout = [x for x in holdout if x['language'] in selected]
            if len(train) < 4 or len(holdout) < 2:
                raise ValueError('至少需要 4 个训练问答和 2 个独立检验问答；均须逐条人工批准')
            train_counts, holdout_counts = self.store._counts(train), self.store._counts(holdout)
            if cfg.language_mode == 'bilingual' and any(
                train_counts[lang] < 3 or holdout_counts[lang] < 2 for lang in LANGUAGES
            ):
                raise ValueError('双语训练要求中英文各至少3条训练与2条独立检验样本')
            if any(train_counts[lang] == 0 or holdout_counts[lang] == 0 for lang in selected):
                raise ValueError('每个目标语言都需要独立的训练与验证样本')
            # Validate all pairs up front. Changes approved during training are not used.
            for item in train + holdout:
                _pair_bytes(item['question'], item['answer'], item['language'])
            self.running = True
            self.stop_event.clear()
            self.attempts += 1
            self.last_result = {'state': 'training', 'language_mode': cfg.language_mode,
                                'started_at': _now()}
            self._save()
            self.worker = threading.Thread(target=self._train_worker,
                                           args=(cfg, list(train), list(holdout)), daemon=True)
            self.worker.start()
        return self.status()

    def stop(self):
        self.stop_event.set()
        return self.status()

    def _build_or_load(self):
        if self.checkpoint.is_file():
            saved = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
            if saved.get('tokenizer') != TOKENIZER_NAME:
                raise ValueError('模型检查点并非 UTF-8 字节词表')
            cfg = ModelConfig(**saved['config'])
            model = SRTM15LanguageModel(cfg)
            model.load_state_dict(saved['model'])
            return model, True
        cfg = ModelConfig(vocab_size=256, num_classes=256, dim=48,
                          max_length=MAX_SEQUENCE, reasoning_steps=1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261023)
            model = SRTM15LanguageModel(cfg)
        return model, False

    def _train_worker(self, cfg, train, holdout):
        started = time.monotonic()
        result = {'state': 'error', 'timestamp': _now()}
        tmp = self.checkpoint.with_suffix('.candidate.tmp')
        try:
            torch.set_num_threads(min(torch.get_num_threads(), 2))
            model, is_incremental = self._build_or_load()
            # On subsequent runs, only fast adapters and language output heads update.
            if is_incremental:
                for name, param in model.named_parameters():
                    param.requires_grad_(('fast_down.' in name or 'fast_up.' in name or
                                          name.startswith('combine.') or name.startswith('expert_head.')))
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate)
            languages = tuple(lang for lang in LANGUAGES if any(x['language'] == lang for x in train))

            def scores(items):
                return {lang: _evaluate(model, [x for x in items if x['language'] == lang], self.stop_event)
                        for lang in languages}

            before_train = scores(train)
            before_val = scores(holdout)
            rand = random.Random(1729 + self.attempts)
            pools = {lang: [r for r in train if r['language'] == lang] for lang in languages}
            steps_done = 0
            model.train()
            for step in range(cfg.steps):
                if self.stop_event.is_set():
                    raise InterruptedError('用户已暂停训练')
                if time.monotonic() - started >= cfg.max_runtime_seconds:
                    raise TimeoutError('达到训练时长上限')
                # Round-robin across languages; an English minority does not
                # get ignored by a large Chinese training corpus (and vice versa).
                item = rand.choice(pools[languages[step % len(languages)]])
                loss = _loss_for_pair(model, item)
                if not bool(torch.isfinite(loss)):
                    raise ValueError('损失不是有限数')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                steps_done += 1
            after_train = scores(train)
            after_val = scores(holdout)
            baseline_train = sum(before_train.values()) / len(languages)
            candidate_train = sum(after_train.values()) / len(languages)
            baseline_val = sum(before_val.values()) / len(languages)
            candidate_val = sum(after_val.values()) / len(languages)
            finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
            improvement = baseline_train - candidate_train
            accepted = (finite and improvement >= 0.005 and
                        all(after_train[lang] <= before_train[lang] + 0.000001
                            and after_val[lang] <= before_val[lang] + 0.000001 for lang in languages) and
                        not self.stop_event.is_set() and time.monotonic()-started < cfg.max_runtime_seconds)
            result = {'state': 'accepted' if accepted else 'rejected',
                      'reason': '目标语言独立检验均通过' if accepted else '某个语言未达到训练或验证门槛',
                      'train_before': round(baseline_train, 5), 'train_after': round(candidate_train, 5),
                      'validation_before': round(baseline_val, 5), 'validation_after': round(candidate_val, 5),
                      'per_language': {lang: {
                          'train_before': round(before_train[lang], 5),
                          'train_after': round(after_train[lang], 5),
                          'validation_before': round(before_val[lang], 5),
                          'validation_after': round(after_val[lang], 5)} for lang in languages},
                      'language_mode': cfg.language_mode,
                      'steps': steps_done, 'elapsed_seconds': round(time.monotonic() - started, 2),
                      'training_count': len(train), 'holdout_count': len(holdout),
                      'timestamp': _now()}
            if accepted:
                # Source revocation or label rejection during training invalidates
                # the candidate before any checkpoint can be activated.
                valid_ids = {r['id'] for r in self.store.approved()}
                if any(r['id'] not in valid_ids for r in train + holdout):
                    accepted = False
                    result['state'] = 'rejected'
                    result['reason'] = '训练期间有问答被撤销批准或来源被拒绝'
            if accepted:
                self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({'model': model.state_dict(), 'config': asdict(model.cfg),
                            'tokenizer': TOKENIZER_NAME, 'steps': steps_done,
                            'source': 'human-reviewed-QA', 'language_mode': cfg.language_mode,
                            'saved_at': _now()}, tmp)
                # Verify integrity before activation.
                candidate_state = torch.load(tmp, map_location='cpu', weights_only=True)
                if candidate_state['tokenizer'] != TOKENIZER_NAME:
                    raise ValueError('候选模型词表格式错误')
                verifier = SRTM15LanguageModel(ModelConfig(**candidate_state['config']))
                verifier.load_state_dict(candidate_state['model'])
                if self.stop_event.is_set():
                    raise InterruptedError('保存候选参数前已收到暂停指令')
                still_eligible = {r['id'] for r in self.store.approved()}
                if any(r['id'] not in still_eligible for r in train + holdout):
                    raise InterruptedError('训练样本审核状态发生变化，候选已取消')
                if self.checkpoint.exists():
                    shutil.copy2(self.checkpoint, self.previous)
                os.replace(tmp, self.checkpoint)
                with self.engine_lock:
                    if self.engine is not None:
                        self.engine.clear_language_cache()
                with self.lock:
                    self.accepted_updates += 1
        except InterruptedError as exc:
            result = {'state': 'cancelled', 'reason': str(exc), 'timestamp': _now()}
        except TimeoutError as exc:
            result = {'state': 'timeout', 'reason': str(exc), 'timestamp': _now()}
        except Exception as exc:
            result = {'state': 'error', 'reason': type(exc).__name__+': '+str(exc)[:150], 'timestamp': _now()}
        finally:
            if tmp.exists():
                tmp.unlink()
            with self.lock:
                self.running = False
                self.last_result = result
                self._save()

    def rollback(self):
        with self.lock:
            if self.running:
                raise ValueError('训练期间不能回滚')
            if not self.checkpoint.is_file():
                raise ValueError('没有语言模型候选更新可以回滚')
            if self.previous.is_file():
                # Swap current and previous, permitting undo/re-do one version.
                tmp = self.checkpoint.with_suffix('.rollback.tmp')
                shutil.copy2(self.checkpoint, tmp)
                os.replace(self.previous, self.checkpoint)
                os.replace(tmp, self.previous)
                mode = 'previous_checkpoint'
            else:
                self.checkpoint.unlink()
                mode = 'original_char_language_model'
            with self.engine_lock:
                if self.engine is not None:
                    self.engine.clear_language_cache()
            self.last_result = {'state': 'rolled_back', 'reason': mode, 'timestamp': _now()}
            self._save()
        return self.status()

    def shutdown(self):
        self.stop_event.set()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=3)
