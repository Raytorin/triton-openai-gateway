"""The real policy owns a real event loop/thread; only GPU factory is replaced."""
import asyncio
from contextlib import asynccontextmanager
import importlib.util
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import sys
import pytest
from test_deepseek_runtime import load_policy


def model(monkeypatch, factory, startup=0.1, shutdown=0.3):
    policy=load_policy(monkeypatch)
    monkeypatch.setitem(sys.modules,'utils.vllm_backend_utils',SimpleNamespace(build_async_engine_client_from_engine_args=factory))
    m=policy();m.args={'model_name':'selected'};m._startup_timeout=startup;m._shutdown_timeout=shutdown
    m._aync_engine_args=object();m._enable_metrics=False;m._vllm_metrics=[]
    m.logger=SimpleNamespace(log_error=lambda msg:None)
    m._deepseek_finalized=False
    return m


def shared_engine_factory(monkeypatch, engine):
    """Import the production builder, replacing unavailable GPU/runtime imports."""
    modules = {
        'vllm.engine.arg_utils': SimpleNamespace(AsyncEngineArgs=object),
        'vllm.engine.protocol': SimpleNamespace(EngineClient=object),
        'vllm.sampling_params': SimpleNamespace(SamplingParams=object, StructuredOutputsParams=object),
        'vllm.usage.usage_lib': SimpleNamespace(UsageContext=SimpleNamespace(OPENAI_API_SERVER='openai')),
        'vllm.v1.metrics.loggers': SimpleNamespace(StatLoggerFactory=object),
        'vllm.v1.engine.async_llm': SimpleNamespace(AsyncLLM=SimpleNamespace(from_vllm_config=lambda **kwargs:engine)),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[1]/'backends/vllm_multimodal/utils/vllm_backend_utils.py'
    spec = importlib.util.spec_from_file_location('shared_engine_builder_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_async_engine_client_from_engine_args


@pytest.mark.parametrize('primary_failure', ['reset', 'body', None])
def test_shared_builder_preserves_primary_failure_when_shutdown_raises(monkeypatch, primary_failure):
    failure = ValueError('first engine error')
    shutdown_error = RuntimeError('engine shutdown failed')
    events = []

    class Engine:
        async def reset_mm_cache(self):
            events.append('reset')
            if primary_failure == 'reset':
                raise failure

        def shutdown(self):
            events.append('shutdown')
            raise shutdown_error

    engine = Engine()
    factory = shared_engine_factory(monkeypatch, engine)
    args = SimpleNamespace(create_engine_config=lambda **kwargs:object(),
                           enable_log_requests=False, aggregate_engine_logging=False,
                           disable_log_stats=False)

    async def use_engine():
        async with factory(args, SimpleNamespace()) as client:
            assert client is engine
            events.append('body')
            if primary_failure == 'body':
                raise failure

    expected = failure if primary_failure else shutdown_error
    with pytest.raises(type(expected), match=str(expected)) as raised:
        asyncio.run(use_engine())
    assert raised.value is expected
    assert events == (['reset', 'shutdown'] if primary_failure == 'reset'
                      else ['reset', 'body', 'shutdown'])
    if primary_failure:
        assert any('engine shutdown failed' in note for note in failure.__notes__)


@pytest.mark.parametrize('failure',[ValueError('causal constructor error'),SystemExit('event thread died')])
def test_constructor_first_cause_survives_cleanup(monkeypatch,failure):
    @asynccontextmanager
    async def factory(**kwargs):
        raise failure
        yield
    m=model(monkeypatch,factory)
    with pytest.raises(type(failure),match=str(failure)):
        m._init_engine()
    m.finalize();m.finalize()
    assert m._event_thread is None


def test_task_query_failure_closes_engine(monkeypatch):
    closed=[]
    class Engine:
        async def get_supported_tasks(self):raise ValueError('task query failed')
    @asynccontextmanager
    async def factory(**kwargs):
        try:yield Engine()
        finally:closed.append(True)
    m=model(monkeypatch,factory)
    with pytest.raises(ValueError,match='task query failed'):m._init_engine()
    assert closed==[True] and m._event_thread is None


@pytest.mark.parametrize('cleanup', ['raises', 'slow', 'waits_for_cancellation'])
def test_task_query_first_cause_survives_context_cleanup(monkeypatch, cleanup):
    failure = ValueError('first task query error')
    cleanup_finished = threading.Event()

    class Engine:
        async def get_supported_tasks(self):
            raise failure

    @asynccontextmanager
    async def factory(**kwargs):
        try:
            yield Engine()
        finally:
            try:
                if cleanup == 'raises':
                    raise RuntimeError('context shutdown failed')
                if cleanup == 'slow':
                    await asyncio.sleep(0.06)
                if cleanup == 'waits_for_cancellation':
                    await asyncio.Future()
            finally:
                cleanup_finished.set()

    m = model(monkeypatch, factory, startup=0.02,
              shutdown=0.02 if cleanup == 'waits_for_cancellation' else 0.3)
    started = time.monotonic()
    with pytest.raises(ValueError, match='first task query error') as raised:
        m._init_engine()
    assert raised.value is failure
    assert time.monotonic() - started < 0.8
    assert cleanup_finished.wait(1)
    if cleanup == 'raises':
        assert any('context shutdown failed' in note for note in failure.__notes__)
    elif cleanup == 'waits_for_cancellation':
        assert any('shutdown deadline' in note for note in failure.__notes__)
    m.finalize()
    assert m._event_thread is None


def test_cooperative_startup_deadline_cancels_and_cleans(monkeypatch):
    closed=threading.Event()
    class Engine:
        async def get_supported_tasks(self):await asyncio.Future()
    @asynccontextmanager
    async def factory(**kwargs):
        try:yield Engine()
        finally:closed.set()
    m=model(monkeypatch,factory,startup=0.02)
    start=time.monotonic()
    with pytest.raises(TimeoutError,match='startup'):m._init_engine()
    assert time.monotonic()-start<0.8 and closed.is_set() and m._event_thread is None
    assert m._llm_engine is None


def test_late_synchronous_factory_cannot_publish_ready(monkeypatch):
    closed=threading.Event()
    class Engine:
        async def get_supported_tasks(self):return ('generate',)
    @asynccontextmanager
    async def factory(**kwargs):
        time.sleep(0.15) # intentional uncooperative constructor boundary
        try:yield Engine()
        finally:closed.set()
    m=model(monkeypatch,factory,startup=0.01,shutdown=0.01)
    with pytest.raises(TimeoutError,match='startup'):m._init_engine()
    assert m._llm_engine is None
    assert closed.wait(1)
    m.finalize()
    assert m._llm_engine is None and m._event_thread is None


def test_successful_unload_and_partial_finalize_are_idempotent(monkeypatch):
    closed=[];metrics=[]
    class Engine:
        async def get_supported_tasks(self):return ('generate',)
    @asynccontextmanager
    async def factory(**kwargs):
        try:yield Engine()
        finally:closed.append(True)
    m=model(monkeypatch,factory)
    m._vllm_metrics=[SimpleNamespace(finalize=lambda:metrics.append(True))]
    m._init_engine();assert m.supported_tasks==('generate',)
    m.finalize();m.finalize()
    assert closed==[True] and metrics==[True] and m._event_thread is None
    load_policy(monkeypatch)().finalize()


def test_backend_preflight_validates_before_shared_constructor(monkeypatch,tmp_path):
    policy=load_policy(monkeypatch)
    monkeypatch.setattr(sys.modules['triton_python_backend_utils'],'get_model_dir',lambda:str(tmp_path),raising=False)
    (tmp_path/'model.json').write_text('{"model":"org/unmaterialized"}')
    class Engine:
        def initialize(self,args):raise AssertionError('shared constructor must remain untouched')
    class Model(policy,Engine):pass
    with pytest.raises(ValueError,match='absolute'):Model().initialize({})


def test_engine_factory_import_failure_surfaces_original_cause(monkeypatch):
    m=model(monkeypatch,object())
    monkeypatch.setitem(sys.modules,'utils.vllm_backend_utils',SimpleNamespace())
    with pytest.raises(ImportError,match='build_async_engine_client_from_engine_args'):m._init_engine()


from test_deepseek_checkpoint import snapshot

@pytest.mark.parametrize('missing',['architecture','tokenizer',None])
def test_backend_native_registry_guard_and_text_only_config_forwarding(monkeypatch,tmp_path,snapshot,missing):
    import json
    policy=load_policy(monkeypatch)
    config=json.loads((Path(__file__).resolve().parents[1]/'examples/deepseek-v4.1-flash/1/model.json').read_text())
    config['model']=str(snapshot)
    (tmp_path/'model.json').write_text(json.dumps(config))
    monkeypatch.setattr(sys.modules['triton_python_backend_utils'],'get_model_dir',lambda:str(tmp_path),raising=False)
    monkeypatch.setitem(sys.modules,'vllm',SimpleNamespace(__version__='0.32.0+cu129'))
    monkeypatch.setitem(sys.modules,'vllm.model_executor.models',SimpleNamespace(ModelRegistry=SimpleNamespace(get_supported_archs=lambda:[] if missing=='architecture' else ['DeepseekV41ForCausalLM'])))
    monkeypatch.setitem(sys.modules,'vllm.tokenizers.registry',SimpleNamespace(TokenizerRegistry=SimpleNamespace(tokenizers={} if missing=='tokenizer' else {'deepseek_v41':('registered','class')})))
    observed=[]
    class Engine:
        def initialize(self,args):
            observed.append(json.loads((tmp_path/'model.json').read_text()))
            self.supported_tasks=('generate',)
    class Model(policy,Engine):pass
    instance=Model()
    if missing:
        with pytest.raises(ValueError,match='registration'):instance.initialize({})
        assert observed==[]
    else:
        instance.initialize({})
        assert observed[0]['language_model_only'] is True and observed[0]['enable_mm_embeds'] is False
        assert observed[0]['engram_config']['cpu_offload'] is True
        assert observed[0]['safetensors_load_strategy']=='lazy'
        assert observed[0]==config
        instance.finalize()


def test_response_worker_and_metrics_stop_after_success(monkeypatch):
    import queue
    finalized=[]
    @asynccontextmanager
    async def factory(**kwargs):
        yield SimpleNamespace(get_supported_tasks=tasks)
    async def tasks():return ('generate',)
    m=model(monkeypatch,factory)
    m._init_engine();m._response_queue=queue.Queue(maxsize=1)
    def respond():
        while m._response_queue.get() is not None:pass
    m._response_thread=threading.Thread(target=respond);m._response_thread.start()
    m._vllm_metrics=[SimpleNamespace(finalize=lambda:finalized.append(True))]
    m.finalize();m.finalize()
    assert m._response_thread is None and finalized==[True]
