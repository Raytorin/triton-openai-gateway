# Third-Party Notices

Triton OpenAI Gateway includes code derived from the
[NVIDIA Triton vLLM Backend](https://github.com/triton-inference-server/vllm_backend).
The affected files are under `backends/vllm_multimodal/` and retain their
original copyright and license headers.

## NVIDIA Triton NGC Base Image

`Dockerfile.triton-gateway` builds on the NVIDIA Triton NGC container. Use and
redistribution of the resulting image are additionally governed by the
[NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/),
the [Product-Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/),
and the licenses of software shipped in the base image.

NVIDIA attribution notice included by this project:

> This software contains source code provided by NVIDIA Corporation.

NVIDIA and Triton names are used only to identify upstream software and
compatibility. This project is independent and is not endorsed by NVIDIA.

## NVIDIA Triton vLLM Backend

Copyright 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of NVIDIA CORPORATION nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The `backends/vllm_multimodal/` files are distributed under BSD-3-Clause as
indicated by their source headers. Other original project code and documentation
are distributed under the repository's Apache-2.0 license.

## NVIDIA DCGM Exporter Helm Chart

The archives at `helm/*/charts/dcgm-exporter-4.8.2.tgz` contain version `4.8.2`
of the [NVIDIA DCGM Exporter Helm chart](https://github.com/NVIDIA/dcgm-exporter).
The chart is distributed under the Apache License 2.0. Its templates retain
their NVIDIA copyright and license headers.
