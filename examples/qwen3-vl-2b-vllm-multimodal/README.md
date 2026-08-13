# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

# Qwen3-VL-2B-Instruct test profile

Small production-path profile for functional image, video and PDF tests through
`vllm_multimodal`. It uses the official Apache-2.0 model
`Qwen/Qwen3-VL-2B-Instruct` and the same backend contract as larger Qwen3-VL
models.

Copy `config.pbtxt` next to the Triton model version and copy `model.json` plus
`gateway.json` into the version directory. Change `model` in `model.json` to the
actual weights directory when the watcher does not rewrite it automatically.

The profile requires an NVIDIA GPU and the pinned NVIDIA Triton/vLLM image
declared by the root `Dockerfile.triton-gateway`.

Qwen3-VL accepts image and video, so PDF pages can also be rendered and analyzed
by the gateway. It does not natively consume audio. Configure a local ASR model
in `vllm_multimodal.audio_asr_model` before testing audio, or use a model whose
vLLM architecture explicitly supports audio.

Start conservatively with this profile. Increase `max_model_len`, media limits,
`max_num_seqs` and admission limits only after measuring KV-cache capacity and
latency on the target GPU.
