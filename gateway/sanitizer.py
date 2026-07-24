# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

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
    r"(?:^|\n)\s*\[?\s*(?:user|assistant|asistant|system)\s*\]?\s*:?\s*\n",
    re.IGNORECASE,
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


def _trim_repeated_completion(text: str) -> tuple[str, bool]:
    normalized = text.rstrip()
    for match in re.finditer(r"\n\s*\n", normalized):
        head = normalized[: match.start()].rstrip()
        tail = normalized[match.end() :].lstrip()
        if len(head) < 80 or len(tail) < 32:
            continue

        duplicate_prefix_len = min(len(head), len(tail), 96)
        if duplicate_prefix_len < 32:
            continue

        if tail.startswith(head[:duplicate_prefix_len]):
            return head, True

    return text, False


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


def sanitize_generated_text(text: str, *, streaming: bool = False) -> tuple[str, bool]:
    if streaming and not _contains_thinking_tag(text):
        if _looks_like_reasoning(text):
            return "", False

    text, stopped_at_thinking = _remove_thinking_blocks(text)
    text = _strip_leading_role_noise(text)
    text, stopped_at_repeat = _trim_repeated_completion(text)
    cut_at = _find_chat_marker_index(text)
    if cut_at is None:
        return text, stopped_at_thinking or stopped_at_repeat
    return text[:cut_at].rstrip(), True
