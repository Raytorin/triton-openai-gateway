# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

import time
import unittest

from gateway.sanitizer import sanitize_generated_text


class SanitizerTests(unittest.TestCase):
    def test_role_marker_stops_at_next_message(self):
        text, stopped = sanitize_generated_text("Answer text\n\n [assistant] :\nExtra text")

        self.assertEqual(text, "Answer text")
        self.assertTrue(stopped)

    def test_many_blank_lines_do_not_stall_gateway(self):
        generated = "Answer text" + "\n" * 2000 + "end"

        started = time.monotonic()
        text, stopped = sanitize_generated_text(generated)

        self.assertEqual(text, generated)
        self.assertFalse(stopped)
        self.assertLess(time.monotonic() - started, 2.0)


class StreamingSanitizerTests(unittest.TestCase):
    def test_incremental_scans_preserve_chunk_boundary_behavior(self):
        from gateway import sanitizer
        self.assertTrue(hasattr(sanitizer, 'StreamingSanitizer'))
        samples = [
            ('Answer text\n\n [assistant] :\nExtra text', 'Answer text', True),
            ('Answer text<|im_end|>extra', 'Answer text', True),
            ('assistant: Answer text', 'Answer text', False),
            ('<think>private reasoning</think>Answer text', 'Answer text', False),
            ('Answer text\nuser request\n\n /no_think extra', 'Answer text', True),
            ('A' * 100 + '\n\n' + 'A' * 40, 'A' * 100, True),
            ('A' * 100 + '\n\n' + 'B' * 100 + '\n\n' + 'A' * 40, 'A' * 100 + '\n\n' + 'B' * 100, True),
            ('Answer text\n' + ' ' * 500 + '\nuser\nextra', 'Answer text', True),
        ]
        for raw, expected, stopped in samples:
            for size in (1, 3, 11, 64):
                with self.subTest(raw=raw[:30], size=size):
                    stream = sanitizer.StreamingSanitizer()
                    for end in range(size, len(raw), size):
                        # Stateless processing is the established compatibility oracle.
                        self.assertEqual(stream.sanitize(raw[:end]), sanitize_generated_text(raw[:end], streaming=True))
                    self.assertEqual(stream.sanitize(raw), (expected, stopped))

    def test_revision_of_buffer_resets_incremental_scans(self):
        from gateway import sanitizer
        self.assertTrue(hasattr(sanitizer, 'StreamingSanitizer'))
        stream = sanitizer.StreamingSanitizer()
        stream.sanitize('old answer' * 100)
        self.assertEqual(stream.sanitize('new answer\nuser\nextra'), ('new answer', True))

    def test_many_paragraphs_do_not_copy_the_whole_answer_per_separator(self):
        class CountedText(str):
            copied = 0
            def __getitem__(self, key):
                value = super().__getitem__(key)
                if isinstance(key, slice):
                    type(self).copied += len(value)
                return value
            def rstrip(self, chars=None):
                return type(self)(super().rstrip(chars))
        from gateway.sanitizer import _trim_repeated_completion
        raw = CountedText(''.join(f'Paragraph {i}: ' + 'content ' * 10 + '\n\n' for i in range(1000)))
        result, stopped = _trim_repeated_completion(raw)
        self.assertEqual(result, raw)
        self.assertFalse(stopped)
        self.assertLess(CountedText.copied, len(raw) * 4)
