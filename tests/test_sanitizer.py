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
