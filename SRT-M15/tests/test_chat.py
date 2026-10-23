import unittest
from pathlib import Path
from srt_m15.chat import SRTChatEngine, route_question


class ChatRoutingTests(unittest.TestCase):
    def test_addition_dispatch(self):
        self.assertEqual(route_question('1+9=\u591a\u5c11'), ('addition', 1, 9))
        self.assertEqual(route_question('\u8bf7\u95ee 3\u52a06 \u7b49\u4e8e\u591a\u5c11'), ('addition', 3, 6))
        self.assertEqual(route_question('14+9'), ('math_limit', 14, 9))
        self.assertEqual(route_question('1+9+2'), ('unsupported',))
        self.assertEqual(route_question('\u4f60\u597d'), ('text',))

    def test_existing_math_checkpoint(self):
        if not Path('checkpoints/srt_m15_toy.pt').is_file():
            self.skipTest('weights not installed')
        engine = SRTChatEngine()
        reply = engine.reply('1+9=\u591a\u5c11')
        self.assertEqual(reply.mode, 'math')
        self.assertIn('10', reply.text)
        self.assertIn('\u795e\u7ecf\u7f51\u7edc', reply.text)

    def test_unseen_text_does_not_crash(self):
        if not Path('checkpoints/srt_m15_lm.pt').is_file():
            self.skipTest('weights not installed')
        engine = SRTChatEngine()
        reply = engine.reply('\u9633\u5149\u660e\u5a9a\u7684\u6668\u66e6\u8ba9\u6211\u5f88\u9ad8\u5174')
        self.assertIn(reply.mode, ('vocab', 'text'))


if __name__ == '__main__':
    unittest.main()
