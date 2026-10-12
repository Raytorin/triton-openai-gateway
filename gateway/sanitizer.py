# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re


DEFAULT_CHAT_STOP_SEQUENCES = [
    "<|im_end|>",
    "<|im_start|>",
    "\nuser\n",
    "\nUser\n",
    "\nassistant\n",
    "\nAssistant\n",
    "\n[user]\n",
    "\n[User]\n",
    "\n[assistant]\n",
    "\n[Assistant]\n",
]
THINK_OPEN_TAGS = ("<think>", "<thinking>")
THINK_CLOSE_TAGS = ("</think>", "</thinking>")
CHAT_ROLE_MARKER_RE = re.compile(
    r"^[ \t]*\[?[ \t]*(?:user|assistant|asistant|system)[ \t]*\]?[ \t]*:?[ \t]*\n",
    re.IGNORECASE | re.MULTILINE,
)
PROMPT_LEAK_MARKER_RE = re.compile(
    r"(?:^|\n)\s*user[^\n]{0,160}\n\s*/no_think\b",
    re.IGNORECASE,
)
LEADING_ROLE_NOISE_RE = re.compile(
    r"^\s*(?:[A-Za-z][A-Za-z0-9_-]{0,31}\s*\n)?"
    r"\[?\s*(?:assistant|asistant)\s*\]?\s*:?\s*",
    re.IGNORECASE,
)
REASONING_CUE_RE = re.compile(
    r"^\s*(?:хорошо|нужно|надо|пользователь|провер|думаю|"
    r"the user|we need|i need|let me|okay)",
    re.IGNORECASE,
)


def strip_prompt_echo(prompt: str, generated_text: str) -> str:
    if generated_text.startswith(prompt):
        return generated_text[len(prompt) :]
    return generated_text


def _looks_like_reasoning(text: str) -> bool:
    return bool(REASONING_CUE_RE.search(text[:500]))


def _contains_thinking_tag(text: str) -> bool:
    return any(tag in text for tag in (*THINK_OPEN_TAGS, *THINK_CLOSE_TAGS))


def _find_first_tag(text: str, tags: tuple[str, ...], start: int = 0) -> tuple[int, str]:
    result_index = -1
    result_tag = ""
    for tag in tags:
        index = text.find(tag, start)
        if index != -1 and (result_index == -1 or index < result_index):
            result_index = index
            result_tag = tag
    return result_index, result_tag


def _strip_leading_role_noise(text: str) -> str:
    previous = None
    current = text
    while previous != current:
        previous = current
        current = LEADING_ROLE_NOISE_RE.sub("", current, count=1)
    return current


def _remove_thinking_blocks(text: str) -> tuple[str, bool]:
    result = text
    while True:
        open_index, open_tag = _find_first_tag(result, THINK_OPEN_TAGS)
        close_index, close_tag = _find_first_tag(result, THINK_CLOSE_TAGS)

        if close_index != -1 and (open_index == -1 or close_index < open_index):
            before_close = result[:close_index].rstrip()
            after_close = result[close_index + len(close_tag) :].lstrip()
            if before_close and not _looks_like_reasoning(before_close):
                return before_close, True
            result = after_close
            continue

        if open_index == -1:
            return result, False

        close_index, close_tag = _find_first_tag(
            result,
            THINK_CLOSE_TAGS,
            open_index + len(open_tag),
        )
        if close_index == -1:
            return result[:open_index].rstrip(), False

        result = (
            result[:open_index]
            + result[close_index + len(close_tag) :].lstrip()
        )


_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_NONSPACE_RE = re.compile(r"\S")
_NEWLINE_RE = re.compile(r"\n")


def _trim_repeated_completion(text: str, state: StreamingSanitizer | None = None) -> tuple[str, bool]:
    normalized = text.rstrip()
    start = state.repeat_start if state is not None else 0
    next_start = len(normalized)
    for match in _PARAGRAPH_RE.finditer(normalized, start):
        # Compare offsets and at most 96 characters. Slicing both the entire
        # head and tail at every paragraph makes one scan quadratic.
        head_end = match.start()
        while head_end and normalized[head_end - 1].isspace():
            head_end -= 1
        if head_end < 80:
            continue
        tail_match = _NONSPACE_RE.search(normalized, match.end())
        tail_start = tail_match.start() if tail_match else len(normalized)
        tail_length = len(normalized) - tail_start
        if tail_length < 32:
            next_start = min(next_start, match.start())
            continue
        prefix_length = min(head_end, tail_length, 96)
        if normalized.startswith(normalized[:prefix_length], tail_start):
            return normalized[:head_end], True
    if state is not None:
        state.repeat_start = next_start
    return text, False


class StreamingSanitizer:
    """Remember completed scans; recheck incomplete lines and paragraph tails.

    Thinking removal and leading role cleanup retain their stateless semantics.
    If either revises the visible prefix, all scan cursors reset automatically.
    """
    def __init__(self):
        self.previous = ""
        self.repeat_start = 0
        self.line_start = 0
        self.previous_line_start = 0
        self.stopped: tuple[str, bool] | None = None

    def sanitize(self, text: str) -> tuple[str, bool]:
        return sanitize_generated_text(text, streaming=True, _state=self)

    def prepare(self, text: str) -> None:
        if not text.startswith(self.previous):
            self.__init__()

    def marker_index(self, text: str) -> int | None:
        old_length = len(self.previous)
        # Literal markers can straddle an arbitrary backend chunk boundary.
        overlap = max(map(len, DEFAULT_CHAT_STOP_SEQUENCES)) - 1
        start = max(0, old_length - overlap)
        cut_at = None
        for marker in DEFAULT_CHAT_STOP_SEQUENCES:
            index = text.find(marker, start)
            if index != -1 and (cut_at is None or index < cut_at):
                cut_at = index

        # Role markers require a completed line; prompt-leak markers require
        # /no_think. Search only when that new input can complete a match.
        has_newline = text.find("\n", old_length) != -1
        if has_newline:
            match = CHAT_ROLE_MARKER_RE.search(text, self.line_start)
            if match and (cut_at is None or match.start() < cut_at):
                cut_at = match.start()
        if "/no_think" in text[max(0, old_length - 9):].lower():
            match = PROMPT_LEAK_MARKER_RE.search(text, self.previous_line_start)
            if match and (cut_at is None or match.start() < cut_at):
                cut_at = match.start()

        for newline in _NEWLINE_RE.finditer(text, old_length):
            # Retain the previous non-empty line through arbitrarily many
            # blank lines: user ... \n <whitespace> /no_think is still a marker.
            if _NONSPACE_RE.search(text, self.line_start, newline.start()):
                self.previous_line_start = max(0, self.line_start - 1)
            self.line_start = newline.end()
        return cut_at


def _find_chat_marker_index(text: str) -> int | None:
    cut_at = None
    for marker in DEFAULT_CHAT_STOP_SEQUENCES:
        index = text.find(marker)
        if index != -1 and (cut_at is None or index < cut_at):
            cut_at = index

    match = CHAT_ROLE_MARKER_RE.search(text)
    if match and (cut_at is None or match.start() < cut_at):
        cut_at = match.start()

    match = PROMPT_LEAK_MARKER_RE.search(text)
    if match and (cut_at is None or match.start() < cut_at):
        cut_at = match.start()

    return cut_at


def sanitize_generated_text(text: str, *, streaming: bool = False, _state: StreamingSanitizer | None = None) -> tuple[str, bool]:
    if streaming and not _contains_thinking_tag(text):
        if _looks_like_reasoning(text):
            return "", False

    text, stopped_at_thinking = _remove_thinking_blocks(text)
    text = _strip_leading_role_noise(text)
    if _state is not None:
        _state.prepare(text)
        if _state.stopped is not None:
            return _state.stopped
    original_text = text
    text, stopped_at_repeat = _trim_repeated_completion(text, _state)
    cut_at = _find_chat_marker_index(text) if _state is None else _state.marker_index(text)
    if _state is not None:
        _state.previous = original_text
        if cut_at is not None or stopped_at_thinking or stopped_at_repeat:
            _state.stopped = (text[:cut_at].rstrip() if cut_at is not None else text, True)
    if cut_at is None:
        return text, stopped_at_thinking or stopped_at_repeat
    return text[:cut_at].rstrip(), True
