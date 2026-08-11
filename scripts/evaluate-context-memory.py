#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

"""Evaluate downstream fact retention after gateway context compaction."""

from __future__ import annotations

import argparse
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


EXPECTED_FACTS = {
    "model": "Qwen3-32B-FP8",
    "config_path": "/srv/models/qwen/model.json",
    "gateway_port": "8080",
    "triton_grpc_port": "8001",
    "max_num_seqs": "45",
    "ticket": "INC-4821",
    "ticket_status": "waiting_for_vendor",
    "constraint": "не удалять старый контекст",
}


def build_messages(filler_turns: int) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = [
        {
            "role": "user",
            "content": (
                "Для текущей задачи используй модель Qwen3-32B-FP8. "
                "Точный путь конфигурации: /srv/models/qwen/model.json."
            ),
        },
        {
            "role": "assistant",
            "content": "Модель и точный путь зафиксированы.",
        },
        {
            "role": "user",
            "content": (
                "Gateway слушает порт 8080, Triton gRPC слушает порт 8001. "
                "Критичное требование: не удалять старый контекст."
            ),
        },
        {
            "role": "assistant",
            "content": "Порты и критичное требование приняты.",
        },
        {
            "role": "assistant",
            "content": "Предлагаю установить max_num_seqs=128.",
        },
        {
            "role": "user",
            "content": (
                "Исправление: max_num_seqs должен быть ровно 45, а не 128. "
                "Считай значение 45 актуальным."
            ),
        },
        {
            "role": "assistant",
            "content": "Принято: актуальное значение max_num_seqs равно 45.",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_ticket_status",
                    "type": "function",
                    "function": {
                        "name": "get_ticket_status",
                        "arguments": "{\"ticket\":\"INC-4821\"}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "get_ticket_status",
            "tool_call_id": "call_ticket_status",
            "content": (
                "{\"ticket\":\"INC-4821\","
                "\"status\":\"waiting_for_vendor\"}"
            ),
        },
    ]
    filler = (
        "Это нерелевантная техническая заметка для заполнения истории; "
        "она не изменяет ранее зафиксированные параметры и решения. "
    )
    for index in range(filler_turns):
        messages.extend(
            [
                {
                    "role": "user",
                    "content": f"Заметка {index}: " + filler * 3,
                },
                {
                    "role": "assistant",
                    "content": f"Заметка {index} принята без изменения параметров.",
                },
            ]
        )
    messages.append(
        {
            "role": "user",
            "content": (
                "Восстанови точные данные из нашей истории. Верни краткий JSON с "
                "полями model, config_path, gateway_port, triton_grpc_port, "
                "max_num_seqs, ticket, ticket_status и constraint. Не заменяй "
                "точные строки пересказом."
            ),
        }
    )
    return messages


def call_gateway(
    url: str,
    model: str,
    filler_turns: int,
    max_tokens: int,
) -> dict[str, object]:
    payload = json.dumps(
        {
            "model": model,
            "messages": build_messages(filler_turns),
            "temperature": 0.0,
            "max_tokens": max_tokens,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=600) as response:
        return json.load(response)


def evaluate(response: dict[str, object]) -> dict[str, object]:
    choices = response.get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    answer = str(message.get("content") or "")
    normalized = answer.casefold()
    checks = {
        name: value.casefold() in normalized
        for name, value in EXPECTED_FACTS.items()
    }
    matched = sum(checks.values())
    return {
        "score": matched / len(checks),
        "matched": matched,
        "total": len(checks),
        "checks": checks,
        "context_status": response.get("context_status"),
        "usage": response.get("usage"),
        "answer": answer,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--filler-turns", type=int, default=22)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--minimum-score", type=float, default=0.75)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    try:
        result = evaluate(
            call_gateway(
                args.url,
                args.model,
                max(args.filler_turns, 0),
                max(args.max_tokens, 1),
            )
        )
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"Gateway returned HTTP {exc.code}: {detail}", file=sys.stderr)
        return 2
    except URLError as exc:
        print(f"Unable to reach gateway: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.strict:
        return 0

    status = result.get("context_status") or {}
    compacted = status.get("action") == "summarize"
    return 0 if compacted and result["score"] >= args.minimum_score else 1


if __name__ == "__main__":
    raise SystemExit(main())
