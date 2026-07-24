# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from fastapi import HTTPException

from .schemas import RerankRequest


def build_rerank_documents(request: RerankRequest) -> list[str]:
    if not request.documents:
        raise HTTPException(status_code=400, detail="Rerank documents must not be empty")

    documents: list[str] = []
    for document in request.documents:
        if isinstance(document, str):
            documents.append(document)
            continue

        if isinstance(document, dict):
            text = document.get("text")
            if isinstance(text, str):
                documents.append(text)
                continue

            content = document.get("content")
            if isinstance(content, str):
                documents.append(content)
                continue

        documents.append(str(document))

    return documents


def build_rerank_response(
    request: RerankRequest,
    documents: list[str],
    scores: list[float],
) -> dict[str, Any]:
    if len(scores) != len(documents):
        raise HTTPException(
            status_code=502,
            detail=(
                "Rerank backend returned an unexpected number of scores: "
                f"{len(scores)} for {len(documents)} documents"
            ),
        )

    ranked = sorted(
        (
            {
                "index": index,
                "relevance_score": float(score),
                "document": {"text": documents[index]},
            }
            for index, score in enumerate(scores)
        ),
        key=lambda item: item["relevance_score"],
        reverse=True,
    )

    if request.top_n is not None:
        if request.top_n <= 0:
            raise HTTPException(status_code=400, detail="top_n must be greater than 0")
        ranked = ranked[: request.top_n]

    if not request.return_documents:
        for item in ranked:
            item.pop("document", None)

    return {
        "id": "rerank",
        "results": ranked,
        "meta": {
            "api_version": {
                "version": "1",
            }
        },
    }
