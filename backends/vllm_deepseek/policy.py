# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-only admission, bounded startup and generation request contract."""
import asyncio
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import re
import threading
import time

import triton_python_backend_utils as pb_utils

_spec = importlib.util.spec_from_file_location('deepseek_checkpoint_preflight', Path(__file__).with_name('checkpoint_preflight.py'))
_checkpoint = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_checkpoint)


class GenerationOnlyModel:
    """Narrow facade; legacy Qwen lifecycle and configuration stay independent."""

    def _preflight_model(self, args):
        config = _checkpoint.read_json(Path(pb_utils.get_model_dir())/'model.json')
        self.checkpoint_report = _checkpoint.validate_checkpoint(config)
        if config.get('language_model_only') is not True or config.get('enable_mm_embeds',False) is not False:
            raise ValueError('vllm_deepseek requires language_model_only:true and enable_mm_embeds:false')
        if config.get('tokenizer_mode','deepseek_v41') != 'deepseek_v41':
            raise ValueError('vllm_deepseek requires tokenizer_mode:deepseek_v41')
        if config.get('trust_remote_code',False) is not False:
            raise ValueError('vllm_deepseek requires trust_remote_code:false')
        _checkpoint.check_runtime()

    def initialize(self, args):
        self.args = args
        self._deepseek_initialized = False
        self._deepseek_finalized = False
        self._startup_timeout = 1800.0
        self._shutdown_timeout = 30.0
        try:
            parameters = json.loads(args.get('model_config','{}')).get('parameters',{})
            for name, attribute in (('STARTUP_TIMEOUT_SECONDS','_startup_timeout'),('SHUTDOWN_TIMEOUT_SECONDS','_shutdown_timeout')):
                value = float(parameters.get(name,{}).get('string_value',getattr(self,attribute)))
                if not math.isfinite(value) or value<=0:
                    raise ValueError(f'{name} must be a finite positive number')
                setattr(self,attribute,value)
            self._preflight_model(args)
            if not re.fullmatch(r'sm[0-9]{2,3}[af]?', os.environ.get('DEEPSEEK_CACHE_ARCH','')):
                raise RuntimeError('Set DEEPSEEK_CACHE_ARCH to the validated GPU architecture before engine startup')
            super().initialize(args)
            if 'generate' not in self.supported_tasks:
                raise ValueError('vllm_deepseek requires a generation-capable model')
            self.supported_tasks = ('generate',)
            self._deepseek_initialized = True
        except BaseException as error:
            self._cleanup_after_error(error)
            raise

    def _cleanup_after_error(self, error):
        try:
            self._cleanup_partial_startup()
        except Exception as cleanup_error:
            error.add_note(f'DeepSeek startup cleanup failed: {cleanup_error}')

    def _init_engine(self):
        self._llm_engine = None
        self._ongoing_request_count = 0
        self._llm_engine_start_cv = threading.Condition()
        self._llm_engine_shutdown_event = asyncio.Event()
        self._startup_aborted = threading.Event()
        self._startup_done = False
        self._startup_error = None
        self._event_loop = None
        self._engine_task = None
        # The stub/process boundary is required to terminate a native constructor
        # that ignores cancellation. Never attempt to kill a Python thread.
        self._event_thread = threading.Thread(target=asyncio.run, args=(self._run_llm_engine(),), daemon=True)
        deadline = time.monotonic()+self._startup_timeout
        self._event_thread.start()
        with self._llm_engine_start_cv:
            while not self._startup_done:
                remaining = deadline-time.monotonic()
                if remaining<=0:
                    self._startup_aborted.set()
                    self._startup_error = TimeoutError(f'DeepSeek startup exceeded {self._startup_timeout:g}s for {self.args.get("model_name","")}; inspect first engine cause and snapshot preflight')
                    break
                self._llm_engine_start_cv.wait(remaining)
        if self._startup_error is not None:
            error = self._startup_error
            logger = getattr(self,'logger',None)
            if logger is not None:
                logger.log_error(f'DeepSeek engine startup failed: {type(error).__name__}: {error}')
            self._cleanup_after_error(error)
            raise error

    async def _run_llm_engine(self):
        self._event_loop = asyncio.get_running_loop()
        self._engine_task = asyncio.current_task()
        try:
            from utils.vllm_backend_utils import build_async_engine_client_from_engine_args
            async with build_async_engine_client_from_engine_args(
                engine_args=self._aync_engine_args, logger=self.logger,
                disable_frontend_multiprocessing=self._enable_metrics,
                stat_loggers=self._vllm_metrics,
            ) as engine:
                try:
                    # Includes the task query in the same startup deadline and cleanup
                    # scope. A late factory return must never publish an engine.
                    if self._startup_aborted.is_set():
                        return
                    supported = await engine.get_supported_tasks()
                    if self._startup_aborted.is_set():
                        return
                    with self._llm_engine_start_cv:
                        if self._startup_aborted.is_set():
                            return
                        self.supported_tasks = supported
                        self._llm_engine = engine
                        self._startup_done = True
                        self._llm_engine_start_cv.notify_all()
                    await self._llm_engine_shutdown_event.wait()
                    while self._ongoing_request_count>0:
                        await asyncio.sleep(0.1)
                except BaseException as error:
                    # Publish the first cause before __aexit__: cleanup can raise
                    # another error or wait past the startup/shutdown deadlines.
                    with self._llm_engine_start_cv:
                        if self._startup_error is None:
                            self._startup_error = error
                        self._startup_done = True
                        self._llm_engine_start_cv.notify_all()
                    raise
        except BaseException as error:
            with self._llm_engine_start_cv:
                if self._startup_error is None:
                    self._startup_error = error
                elif error is not self._startup_error and not isinstance(error,asyncio.CancelledError):
                    self._startup_error.add_note(f'DeepSeek engine context cleanup failed: {type(error).__name__}: {error}')
                self._startup_done = True
                self._llm_engine_start_cv.notify_all()
            # Preserve the original exception for the initialize caller; asyncio
            # cancellation also passes through the context manager's cleanup.
        finally:
            with self._llm_engine_start_cv:
                self._llm_engine = None
                if not self._startup_done:
                    self._startup_done = True
                    if self._startup_error is None:
                        self._startup_error = RuntimeError('DeepSeek event thread exited before startup completed')
                self._llm_engine_start_cv.notify_all()

    def _cleanup_partial_startup(self):
        timeout = getattr(self,'_shutdown_timeout',30.0)
        thread = getattr(self,'_event_thread',None)
        loop = getattr(self,'_event_loop',None)
        shutdown = getattr(self,'_llm_engine_shutdown_event',None)
        aborted = getattr(self,'_startup_aborted',None)
        if aborted is not None:
            aborted.set()
        if thread is not None:
            if loop is not None and loop.is_running():
                def signal():
                    if shutdown is not None:
                        shutdown.set()
                    task = getattr(self,'_engine_task',None)
                    if task is not None and not getattr(self,'_startup_done',False):
                        task.cancel()
                try:
                    loop.call_soon_threadsafe(signal)
                except RuntimeError:
                    pass # loop finished between is_running and signal
            thread.join(timeout=timeout)
            if thread.is_alive():
                if loop is not None and loop.is_running():
                    task = getattr(self,'_engine_task',None)
                    if task is not None:
                        loop.call_soon_threadsafe(task.cancel)
                raise RuntimeError('engine thread did not stop within shutdown deadline; native constructor requires owning stub/process cleanup')
            self._event_thread = None
        response_thread = getattr(self,'_response_thread',None)
        if response_thread is not None:
            try:
                self._response_queue.put(None,timeout=timeout)
            except queue.Full as error:
                raise RuntimeError('response queue did not drain within shutdown deadline') from error
            response_thread.join(timeout=timeout)
            if response_thread.is_alive():
                raise RuntimeError('response thread did not stop within shutdown deadline')
            self._response_thread = None
        factories = getattr(self,'_vllm_metrics',())
        self._vllm_metrics = []
        for factory in factories:
            factory.finalize()
        self._deepseek_finalized = True

    def _validate_request_task_name(self, request):
        for name in ('embedding_request','image','video','audio','pdf','media_parameters'):
            if pb_utils.get_input_tensor_by_name(request,name) is not None:
                raise ValueError(f'vllm_deepseek is generation-only; {name} is unsupported')
        return 'generate'

    def finalize(self):
        if not getattr(self,'_deepseek_finalized',False):
            self._cleanup_partial_startup()
