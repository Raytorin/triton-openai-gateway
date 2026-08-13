# Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import queue
import threading
from typing import Dict, List, Optional, Union

import triton_python_backend_utils as pb_utils
from vllm.config import VllmConfig
from vllm.v1.metrics.loggers import StatLoggerBase, build_1_2_5_buckets
from vllm.v1.metrics.stats import IterationStats, MultiModalCacheStats, SchedulerStats


class RequestTokenMetrics:
    """Stable token counters independent of vLLM's internal stats API."""

    def __init__(self, labels: Dict[str, str]):
        self._prompt_family = pb_utils.MetricFamily(
            name="triton_vllm_multimodal_prompt_tokens_total",
            description="Prompt tokens observed by the vllm_multimodal backend.",
            kind=pb_utils.MetricFamily.COUNTER,
        )
        self._generation_family = pb_utils.MetricFamily(
            name="triton_vllm_multimodal_generation_tokens_total",
            description="Generation tokens observed by the vllm_multimodal backend.",
            kind=pb_utils.MetricFamily.COUNTER,
        )
        self._prompt = self._prompt_family.Metric(labels=labels)
        self._generation = self._generation_family.Metric(labels=labels)

    def increment(self, prompt_tokens: int, generation_tokens: int) -> None:
        if prompt_tokens > 0:
            self._prompt.increment(prompt_tokens)
        if generation_tokens > 0:
            self._generation.increment(generation_tokens)


class RequestTokenAccumulator:
    """Counts cumulative vLLM request outputs exactly once per token."""

    def __init__(self):
        self.prompt_tokens = 0
        self.generation_tokens = 0
        self._prompt_recorded = False
        self._output_lengths: List[int] = []

    def observe(self, request_output) -> None:
        if not self._prompt_recorded:
            prompt_token_ids = getattr(request_output, "prompt_token_ids", None)
            if prompt_token_ids is not None:
                self.prompt_tokens = len(prompt_token_ids)
                self._prompt_recorded = True

        outputs = getattr(request_output, "outputs", None) or []
        current_lengths = [
            len(getattr(output, "token_ids", None) or []) for output in outputs
        ]
        for index, current_length in enumerate(current_lengths):
            previous_length = (
                self._output_lengths[index]
                if index < len(self._output_lengths)
                else 0
            )
            self.generation_tokens += max(current_length - previous_length, 0)
        self._output_lengths = current_lengths


class TritonMetrics:
    def __init__(self, labels: List[str], max_model_len: int):
        # Initialize metric families
        # Iteration stats
        self.counter_prompt_tokens_family = pb_utils.MetricFamily(
            name="vllm:prompt_tokens_total",
            description="Number of prefill tokens processed.",
            kind=pb_utils.MetricFamily.COUNTER,
        )
        self.counter_generation_tokens_family = pb_utils.MetricFamily(
            name="vllm:generation_tokens_total",
            description="Number of generation tokens processed.",
            kind=pb_utils.MetricFamily.COUNTER,
        )
        self.histogram_time_to_first_token_family = pb_utils.MetricFamily(
            name="vllm:time_to_first_token_seconds",
            description="Histogram of time to first token in seconds.",
            kind=pb_utils.MetricFamily.HISTOGRAM,
        )
        self.histogram_time_per_output_token_family = pb_utils.MetricFamily(
            name="vllm:time_per_output_token_seconds",
            description="Histogram of time per output token in seconds.",
            kind=pb_utils.MetricFamily.HISTOGRAM,
        )
        # Request stats
        #   Latency
        self.histogram_e2e_time_request_family = pb_utils.MetricFamily(
            name="vllm:e2e_request_latency_seconds",
            description="Histogram of end to end request latency in seconds.",
            kind=pb_utils.MetricFamily.HISTOGRAM,
        )
        #   Metadata
        self.histogram_num_prompt_tokens_request_family = pb_utils.MetricFamily(
            name="vllm:request_prompt_tokens",
            description="Number of prefill tokens processed.",
            kind=pb_utils.MetricFamily.HISTOGRAM,
        )
        self.histogram_num_generation_tokens_request_family = pb_utils.MetricFamily(
            name="vllm:request_generation_tokens",
            description="Number of generation tokens processed.",
            kind=pb_utils.MetricFamily.HISTOGRAM,
        )
        self.histogram_n_request_family = pb_utils.MetricFamily(
            name="vllm:request_params_n",
            description="Histogram of the n request parameter.",
            kind=pb_utils.MetricFamily.HISTOGRAM,
        )
        self.gauge_running_requests_family = pb_utils.MetricFamily(
            name="vllm:num_requests_running",
            description="Number of requests currently running in the vLLM scheduler.",
            kind=pb_utils.MetricFamily.GAUGE,
        )
        self.gauge_waiting_requests_family = pb_utils.MetricFamily(
            name="vllm:num_requests_waiting",
            description="Number of requests currently waiting in the vLLM scheduler.",
            kind=pb_utils.MetricFamily.GAUGE,
        )
        self.gauge_kv_cache_usage_family = pb_utils.MetricFamily(
            name="vllm:kv_cache_usage_ratio",
            description="Fraction of the vLLM KV cache currently in use.",
            kind=pb_utils.MetricFamily.GAUGE,
        )
        self.counter_prefix_cache_queries_family = pb_utils.MetricFamily(
            name="vllm:prefix_cache_queries_total",
            description="Prompt tokens queried in the vLLM prefix cache.",
            kind=pb_utils.MetricFamily.COUNTER,
        )
        self.counter_prefix_cache_hits_family = pb_utils.MetricFamily(
            name="vllm:prefix_cache_hits_total",
            description="Prompt tokens served from the vLLM prefix cache.",
            kind=pb_utils.MetricFamily.COUNTER,
        )
        self.counter_preemptions_family = pb_utils.MetricFamily(
            name="vllm:preemptions_total",
            description="Requests preempted by the vLLM scheduler.",
            kind=pb_utils.MetricFamily.COUNTER,
        )
        self.counter_kv_cache_evictions_family = pb_utils.MetricFamily(
            name="vllm:kv_cache_evictions_total",
            description="KV cache eviction events reported by vLLM.",
            kind=pb_utils.MetricFamily.COUNTER,
        )
        self.counter_mm_cache_queries_family = pb_utils.MetricFamily(
            name="vllm:mm_cache_queries_total",
            description="Items queried in the vLLM multimodal cache.",
            kind=pb_utils.MetricFamily.COUNTER,
        )
        self.counter_mm_cache_hits_family = pb_utils.MetricFamily(
            name="vllm:mm_cache_hits_total",
            description="Items served from the vLLM multimodal cache.",
            kind=pb_utils.MetricFamily.COUNTER,
        )
        self.histogram_queue_time_family = pb_utils.MetricFamily(
            name="vllm:request_queue_time_seconds",
            description="Time requests spent queued in the vLLM scheduler.",
            kind=pb_utils.MetricFamily.HISTOGRAM,
        )
        self.histogram_prefill_time_family = pb_utils.MetricFamily(
            name="vllm:request_prefill_time_seconds",
            description="Time vLLM spent prefilling finished requests.",
            kind=pb_utils.MetricFamily.HISTOGRAM,
        )
        self.histogram_decode_time_family = pb_utils.MetricFamily(
            name="vllm:request_decode_time_seconds",
            description="Time vLLM spent decoding finished requests.",
            kind=pb_utils.MetricFamily.HISTOGRAM,
        )
        self.histogram_inference_time_family = pb_utils.MetricFamily(
            name="vllm:request_inference_time_seconds",
            description="Total scheduled inference time for finished requests.",
            kind=pb_utils.MetricFamily.HISTOGRAM,
        )
        self.histogram_cached_prompt_tokens_family = pb_utils.MetricFamily(
            name="vllm:request_cached_prompt_tokens",
            description="Prompt tokens reused from prefix cache per finished request.",
            kind=pb_utils.MetricFamily.HISTOGRAM,
        )

        # Initialize metrics
        # Iteration stats
        self.counter_prompt_tokens = self.counter_prompt_tokens_family.Metric(
            labels=labels
        )
        self.counter_generation_tokens = self.counter_generation_tokens_family.Metric(
            labels=labels
        )
        # Use the same bucket boundaries from vLLM sample metrics as an example.
        # https://github.com/vllm-project/vllm/blob/21313e09e3f9448817016290da20d0db1adf3664/vllm/engine/metrics.py#L81-L96
        self.histogram_time_to_first_token = (
            self.histogram_time_to_first_token_family.Metric(
                labels=labels,
                buckets=[
                    0.001,
                    0.005,
                    0.01,
                    0.02,
                    0.04,
                    0.06,
                    0.08,
                    0.1,
                    0.25,
                    0.5,
                    0.75,
                    1.0,
                    2.5,
                    5.0,
                    7.5,
                    10.0,
                ],
            )
        )
        self.histogram_time_per_output_token = (
            self.histogram_time_per_output_token_family.Metric(
                labels=labels,
                buckets=[
                    0.01,
                    0.025,
                    0.05,
                    0.075,
                    0.1,
                    0.15,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.75,
                    1.0,
                    2.5,
                ],
            )
        )
        # Request stats
        #   Latency
        self.histogram_e2e_time_request = self.histogram_e2e_time_request_family.Metric(
            labels=labels,
            buckets=[1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        )
        #   Metadata
        self.histogram_num_prompt_tokens_request = (
            self.histogram_num_prompt_tokens_request_family.Metric(
                labels=labels,
                buckets=build_1_2_5_buckets(max_model_len),
            )
        )
        self.histogram_num_generation_tokens_request = (
            self.histogram_num_generation_tokens_request_family.Metric(
                labels=labels,
                buckets=build_1_2_5_buckets(max_model_len),
            )
        )
        self.histogram_n_request = self.histogram_n_request_family.Metric(
            labels=labels,
            buckets=[1, 2, 5, 10, 20],
        )
        self.gauge_running_requests = self.gauge_running_requests_family.Metric(
            labels=labels
        )
        self.gauge_waiting_requests = self.gauge_waiting_requests_family.Metric(
            labels=labels
        )
        self.gauge_kv_cache_usage = self.gauge_kv_cache_usage_family.Metric(
            labels=labels
        )
        self.counter_prefix_cache_queries = (
            self.counter_prefix_cache_queries_family.Metric(labels=labels)
        )
        self.counter_prefix_cache_hits = self.counter_prefix_cache_hits_family.Metric(
            labels=labels
        )
        self.counter_preemptions = self.counter_preemptions_family.Metric(labels=labels)
        self.counter_kv_cache_evictions = (
            self.counter_kv_cache_evictions_family.Metric(labels=labels)
        )
        self.counter_mm_cache_queries = self.counter_mm_cache_queries_family.Metric(
            labels=labels
        )
        self.counter_mm_cache_hits = self.counter_mm_cache_hits_family.Metric(
            labels=labels
        )
        latency_buckets = [
            0.001,
            0.005,
            0.01,
            0.025,
            0.05,
            0.1,
            0.25,
            0.5,
            1.0,
            2.5,
            5.0,
            10.0,
            30.0,
            60.0,
            120.0,
            300.0,
            600.0,
        ]
        self.histogram_queue_time = self.histogram_queue_time_family.Metric(
            labels=labels, buckets=latency_buckets
        )
        self.histogram_prefill_time = self.histogram_prefill_time_family.Metric(
            labels=labels, buckets=latency_buckets
        )
        self.histogram_decode_time = self.histogram_decode_time_family.Metric(
            labels=labels, buckets=latency_buckets
        )
        self.histogram_inference_time = self.histogram_inference_time_family.Metric(
            labels=labels, buckets=latency_buckets
        )
        self.histogram_cached_prompt_tokens = (
            self.histogram_cached_prompt_tokens_family.Metric(
                labels=labels,
                buckets=build_1_2_5_buckets(max_model_len),
            )
        )


# Create a partially initialized callable that adapts VllmStatLogger to StatLoggerFactory interface
class VllmStatLoggerFactory:
    def __init__(self, labels, log_logger):
        self._labels = labels
        self._log_logger = log_logger
        self._instances_list = []

    def __call__(self, vllm_config, engine_index):
        stat_logger = VllmStatLogger(
            self._labels, self._log_logger, vllm_config, engine_index
        )
        self._instances_list.append(stat_logger)
        return stat_logger

    def finalize(self):
        for stat_logger in self._instances_list:
            if stat_logger is not None:
                stat_logger.finalize()


class VllmStatLogger(StatLoggerBase):
    """StatLogger is used as an adapter between vLLM stats collector and Triton metrics provider."""

    def __init__(
        self, labels: Dict, log_logger, vllm_config: VllmConfig, engine_index: int
    ) -> None:
        # Tracked stats over current local logging interval.
        # local_interval not used here. It's for vLLM logs to stdout.
        super().__init__(vllm_config=vllm_config, engine_index=engine_index)
        self.metrics = TritonMetrics(
            labels=labels, max_model_len=vllm_config.model_config.max_model_len
        )
        self.log_logger = log_logger

        # Starting the metrics thread. It allows vLLM to keep making progress
        # while reporting metrics to triton metrics service.
        queue_size = max(
            int(os.environ.get("VLLM_MULTIMODAL_METRICS_QUEUE_SIZE", "4096")),
            1,
        )
        self._logger_queue = queue.Queue(maxsize=queue_size)
        self._dropped_metrics = 0
        self._logger_thread = threading.Thread(target=self._logger_loop)
        self._logger_thread.start()

    def _log_counter(self, counter, data: Union[int, float]) -> None:
        """Convenience function for logging to counter.

        Args:
            counter: A counter metric instance.
            data: An int or float to increment the count metric.

        Returns:
            None
        """
        if data != 0:
            self._enqueue_metric((counter, "increment", data))

    def _log_histogram(self, histogram, data: Union[List[int], List[float]]) -> None:
        """Convenience function for logging list to histogram.

        Args:
            histogram: A histogram metric instance.
            data: A list of int or float data to observe into the histogram metric.

        Returns:
            None
        """
        for datum in data:
            self._enqueue_metric((histogram, "observe", datum))

    def _set_gauge(self, gauge, value: Union[int, float]) -> None:
        self._enqueue_metric((gauge, "set", value))

    def _enqueue_metric(self, item) -> None:
        try:
            self._logger_queue.put_nowait(item)
        except queue.Full:
            # Metrics must never stall inference. Dropping a sample is safer
            # than allowing this auxiliary queue to grow without a bound.
            self._dropped_metrics += 1
            if self._dropped_metrics == 1 or self._dropped_metrics % 1000 == 0:
                self.log_logger.log_warn(
                    "[vllm] Metrics queue is full; dropped "
                    f"{self._dropped_metrics} sample(s)"
                )

    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int = 0,
    ) -> None:
        """Report stats to Triton metrics server.

        Args:
            stats: Created by LLMEngine for use by VllmStatLogger.

        Returns:
            None
        """
        if scheduler_stats is not None:
            self._set_gauge(
                self.metrics.gauge_running_requests,
                scheduler_stats.num_running_reqs,
            )
            self._set_gauge(
                self.metrics.gauge_waiting_requests,
                scheduler_stats.num_waiting_reqs,
            )
            self._set_gauge(
                self.metrics.gauge_kv_cache_usage,
                scheduler_stats.kv_cache_usage,
            )
            prefix_stats = scheduler_stats.prefix_cache_stats
            self._log_counter(
                self.metrics.counter_prefix_cache_queries,
                prefix_stats.queries,
            )
            self._log_counter(
                self.metrics.counter_prefix_cache_hits,
                prefix_stats.hits,
            )
            self._log_counter(
                self.metrics.counter_kv_cache_evictions,
                len(scheduler_stats.kv_cache_eviction_events),
            )

        if mm_cache_stats is not None:
            self._log_counter(
                self.metrics.counter_mm_cache_queries,
                mm_cache_stats.queries,
            )
            self._log_counter(
                self.metrics.counter_mm_cache_hits,
                mm_cache_stats.hits,
            )

        if iteration_stats is None:
            return

        # Parse finished request stats into lists
        e2e_latency: List[float] = []
        num_prompt_tokens: List[int] = []
        num_generation_tokens: List[int] = []
        queue_times: List[float] = []
        prefill_times: List[float] = []
        decode_times: List[float] = []
        inference_times: List[float] = []
        cached_prompt_tokens: List[int] = []
        for finished_req in iteration_stats.finished_requests:
            e2e_latency.append(finished_req.e2e_latency)
            num_prompt_tokens.append(finished_req.num_prompt_tokens)
            num_generation_tokens.append(finished_req.num_generation_tokens)
            queue_times.append(finished_req.queued_time)
            prefill_times.append(finished_req.prefill_time)
            decode_times.append(finished_req.decode_time)
            inference_times.append(finished_req.inference_time)
            cached_prompt_tokens.append(finished_req.num_cached_tokens)

        # The list of vLLM metrics reporting to Triton is also documented here.
        # https://github.com/triton-inference-server/vllm_backend/blob/main/README.md#triton-metrics
        counter_metrics = [
            (self.metrics.counter_prompt_tokens, iteration_stats.num_prompt_tokens),
            (
                self.metrics.counter_generation_tokens,
                iteration_stats.num_generation_tokens,
            ),
            (self.metrics.counter_preemptions, iteration_stats.num_preempted_reqs),
        ]
        histogram_metrics = [
            (
                self.metrics.histogram_time_to_first_token,
                iteration_stats.time_to_first_tokens_iter,
            ),
            (
                self.metrics.histogram_time_per_output_token,
                iteration_stats.inter_token_latencies_iter,
            ),
            (self.metrics.histogram_e2e_time_request, e2e_latency),
            (
                self.metrics.histogram_num_prompt_tokens_request,
                num_prompt_tokens,
            ),
            (
                self.metrics.histogram_num_generation_tokens_request,
                num_generation_tokens,
            ),
            (self.metrics.histogram_n_request, iteration_stats.n_params_iter),
            (self.metrics.histogram_queue_time, queue_times),
            (self.metrics.histogram_prefill_time, prefill_times),
            (self.metrics.histogram_decode_time, decode_times),
            (self.metrics.histogram_inference_time, inference_times),
            (self.metrics.histogram_cached_prompt_tokens, cached_prompt_tokens),
        ]
        for metric, data in counter_metrics:
            self._log_counter(metric, data)
        for metric, data in histogram_metrics:
            self._log_histogram(metric, data)

    def log_engine_initialized(self) -> None:
        pass

    def _logger_loop(self):
        while True:
            item = self._logger_queue.get()
            # To signal shutdown a None item will be added to the queue.
            if item is None:
                break
            metric, command, data = item
            if command == "increment":
                metric.increment(data)
            elif command == "observe":
                metric.observe(data)
            elif command == "set":
                metric.set(data)
            else:
                self.log_logger.log_error(f"Undefined command name: {command}")

    def finalize(self):
        # Shutdown the logger thread.
        self._logger_queue.put(None)
        if self._logger_thread is not None:
            self._logger_thread.join()
            self._logger_thread = None
