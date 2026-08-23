# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import heapq
import json
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import triton_python_backend_utils as pb_utils
from transformers import AutoModel, AutoTokenizer


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        self.logger = pb_utils.Logger
        self.model_path = os.path.join(
            args["model_repository"],
            args["model_version"],
        )
        self.max_length = self._config_positive_int("MAX_LENGTH", 8192)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.logger.log_info(
            f"[python-backend][bge-m3-hybrid] loading model from {self.model_path}"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=False,
            local_files_only=True,
        )
        self.encoder = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=False,
            local_files_only=True,
            dtype=self.dtype,
        ).to(self.device)
        self.encoder.eval()

        sparse_head_path = os.path.join(self.model_path, "sparse_linear.pt")
        if not os.path.isfile(sparse_head_path):
            raise RuntimeError(
                "BGE-M3 sparse head is missing: sparse_linear.pt must be stored "
                "next to the model weights"
            )
        self.sparse_linear = torch.nn.Linear(
            self.encoder.config.hidden_size,
            1,
            dtype=self.dtype,
        )
        sparse_state = torch.load(
            sparse_head_path,
            map_location="cpu",
            weights_only=True,
        )
        self.sparse_linear.load_state_dict(sparse_state)
        self.sparse_linear.to(self.device)
        self.sparse_linear.eval()

        self.special_token_ids = {
            int(token_id)
            for token_id in (
                self.tokenizer.cls_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
                self.tokenizer.unk_token_id,
            )
            if token_id is not None
        }

    def execute(self, requests):
        responses: list[Any | None] = [None] * len(requests)
        prepared: list[dict[str, Any]] = []

        for request_index, request in enumerate(requests):
            try:
                payload = self._get_embedding_request(request)
                prepared.append(self._prepare_request(request_index, payload))
            except Exception as exc:
                responses[request_index] = self._error_response(exc)

        if prepared:
            try:
                batch_responses = self._embed_batch(prepared)
            except Exception as exc:
                self.logger.log_error(
                    "[python-backend][bge-m3-hybrid] batch inference failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                for item in prepared:
                    responses[item["request_index"]] = self._error_response(exc)
            else:
                for item, response in zip(prepared, batch_responses):
                    responses[item["request_index"]] = response

        return [
            response
            if response is not None
            else self._error_response(RuntimeError("missing embedding response"))
            for response in responses
        ]

    def finalize(self):
        self.logger.log_info("[python-backend][bge-m3-hybrid] finalizing model")

    def _prepare_request(self, request_index: int, payload: dict[str, Any]):
        model_input = payload.get("input")
        if isinstance(model_input, str):
            if not model_input:
                raise ValueError("embedding_request.input must not be empty")
        elif isinstance(model_input, list):
            if not model_input or not all(
                isinstance(token_id, int) and not isinstance(token_id, bool)
                for token_id in model_input
            ):
                raise ValueError(
                    "embedding_request.input token ids must be a non-empty integer array"
                )
        else:
            raise ValueError(
                "embedding_request.input must be a string or list of token ids"
            )

        pooling_params = payload.get("pooling_params") or {}
        if not isinstance(pooling_params, dict):
            raise ValueError("embedding_request.pooling_params must be an object")
        max_length = self._positive_int(
            pooling_params.get("max_length", self.max_length),
            "pooling_params.max_length",
        )
        if max_length > self.max_length:
            raise ValueError(
                f"max_length={max_length} exceeds configured limit {self.max_length}"
            )

        dimensions = self._parse_dimensions(pooling_params.get("dimensions"))
        if (
            dimensions is not None
            and dimensions != int(self.encoder.config.hidden_size)
        ):
            raise ValueError(
                "BAAI/bge-m3 does not support Matryoshka dimension truncation; "
                f"dimensions must be {self.encoder.config.hidden_size} or omitted"
            )

        explicit_output_types = "output_types" in payload
        output_types = payload.get("output_types", ["dense"])
        if not isinstance(output_types, list) or not output_types:
            raise ValueError("embedding_request.output_types must be a non-empty array")
        output_types = list(dict.fromkeys(str(item).lower() for item in output_types))
        unknown = set(output_types) - {"dense", "sparse"}
        if unknown:
            raise ValueError(
                "unsupported embedding output types: " + ", ".join(sorted(unknown))
            )
        if payload.get("sparse_format", "indices_values") != "indices_values":
            raise ValueError("only sparse_format=indices_values is supported")

        sparse_top_k = payload.get("sparse_top_k")
        if sparse_top_k is not None:
            sparse_top_k = self._positive_int(sparse_top_k, "sparse_top_k")
            if "sparse" not in output_types:
                raise ValueError("sparse_top_k requires sparse output")

        return {
            "request_index": request_index,
            "input": model_input,
            "max_length": max_length,
            "output_types": output_types,
            "sparse_top_k": sparse_top_k,
            "legacy_dense_response": not explicit_output_types,
        }

    def _embed_batch(self, prepared: list[dict[str, Any]]):
        features = [self._tokenize(item) for item in prepared]
        encoded = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        need_dense = any("dense" in item["output_types"] for item in prepared)
        need_sparse = any("sparse" in item["output_types"] for item in prepared)

        with torch.inference_mode():
            hidden_state = self.encoder(**encoded, return_dict=True).last_hidden_state
            dense_vectors = (
                F.normalize(hidden_state[:, 0], p=2, dim=-1)
                if need_dense
                else None
            )
            token_weights = (
                torch.relu(self.sparse_linear(hidden_state)).squeeze(-1)
                if need_sparse
                else None
            )

        responses = []
        for row, item in enumerate(prepared):
            result: dict[str, Any] = {}
            if "dense" in item["output_types"]:
                assert dense_vectors is not None
                result["dense"] = (
                    dense_vectors[row].detach().to(torch.float32).cpu().tolist()
                )
            if "sparse" in item["output_types"]:
                assert token_weights is not None
                result["sparse"] = self._build_sparse_embedding(
                    encoded["input_ids"][row],
                    encoded["attention_mask"][row],
                    token_weights[row],
                    item["sparse_top_k"],
                )

            response_payload: Any = (
                result["dense"] if item["legacy_dense_response"] else result
            )
            num_input_tokens = int(encoded["attention_mask"][row].sum().item())
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor(
                            "text_output",
                            np.asarray(
                                [json.dumps(response_payload, ensure_ascii=False)],
                                dtype=object,
                            ),
                        ),
                        pb_utils.Tensor(
                            "num_input_tokens",
                            np.asarray([num_input_tokens], dtype=np.uint32),
                        ),
                        pb_utils.Tensor(
                            "num_output_tokens",
                            np.asarray([0], dtype=np.uint32),
                        ),
                    ]
                )
            )
        return responses

    def _tokenize(self, item: dict[str, Any]) -> dict[str, list[int]]:
        model_input = item["input"]
        if isinstance(model_input, str):
            return self.tokenizer(
                model_input,
                truncation=True,
                max_length=item["max_length"],
                return_token_type_ids=False,
            )

        input_ids = model_input[: item["max_length"]]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }

    def _build_sparse_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_weights: torch.Tensor,
        sparse_top_k: int | None,
    ) -> dict[str, list[int] | list[float]]:
        best_weights: dict[int, float] = {}
        for token_id, mask, weight in zip(
            input_ids.detach().cpu().tolist(),
            attention_mask.detach().cpu().tolist(),
            token_weights.detach().to(torch.float32).cpu().tolist(),
        ):
            token_id = int(token_id)
            weight = float(weight)
            if not mask or token_id in self.special_token_ids or weight <= 0:
                continue
            best_weights[token_id] = max(best_weights.get(token_id, 0.0), weight)

        pairs = list(best_weights.items())
        if sparse_top_k is not None and len(pairs) > sparse_top_k:
            pairs = heapq.nlargest(sparse_top_k, pairs, key=lambda pair: pair[1])
        pairs.sort(key=lambda pair: pair[0])
        return {
            "indices": [token_id for token_id, _ in pairs],
            "values": [weight for _, weight in pairs],
        }

    def _get_embedding_request(self, request) -> dict[str, Any]:
        tensor = pb_utils.get_input_tensor_by_name(request, "embedding_request")
        if tensor is None:
            raise ValueError("missing required input: embedding_request")
        value = tensor.as_numpy().reshape(-1)[0]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        payload = json.loads(str(value))
        if not isinstance(payload, dict):
            raise ValueError("embedding_request must be a JSON object")
        return payload

    def _config_positive_int(self, key: str, default: int) -> int:
        parameters = self.model_config.get("parameters") or {}
        raw = parameters.get(key) or {}
        return self._positive_int(raw.get("string_value", default), key)

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive integer")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
        if parsed <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return parsed

    @staticmethod
    def _parse_dimensions(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, list):
            if not value:
                return None
            value = value[0]
        return TritonPythonModel._positive_int(value, "dimensions")

    @staticmethod
    def _error_response(exc: Exception):
        return pb_utils.InferenceResponse(
            error=pb_utils.TritonError(f"{type(exc).__name__}: {exc}")
        )
