"""Model-local effort aliases preserve the selected template's native contract."""
import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from jinja2.sandbox import ImmutableSandboxedEnvironment

from gateway.prompt import RequestTokenizer, render_chat_prompt
from gateway.reasoning import load_reasoning_settings

FIXTURES = Path(__file__).parent / 'fixtures'


class Tokenizer:
    def __init__(self, template):
        self.chat_template = template

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True,
                            chat_template=None, **kwargs):
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
            extensions=['jinja2.ext.loopcontrols'])
        env.globals['raise_exception'] = lambda text: (_ for _ in ()).throw(ValueError(text))
        return env.from_string(chat_template or self.chat_template).render(
            messages=messages, add_generation_prompt=add_generation_prompt, **kwargs)


def model(tmp_path, *, kind='deepseek_v41', mapping=None):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / 'config.json').write_text(json.dumps({'model_type': kind}))
    settings = {'mode': 'separate'}
    if mapping is not None:
        settings['effort_map'] = mapping
    (tmp_path / 'gateway.json').write_text(json.dumps({'reasoning': settings}))
    return tmp_path


def deepseek_tokenizer():
    return Tokenizer((FIXTURES / 'deepseek/v41-flash/chat_template.jinja').read_text())


def prompt(tokenizer, settings):
    wrapped = RequestTokenizer(tokenizer, settings.template_kwargs, strict=True,
                               chat_template=settings.resolved_chat_template)
    return render_chat_prompt(wrapped, [{'role': 'user', 'content': 'What is 19 + 23?'}], None)


def test_mapped_xhigh_matches_existing_independent_high_golden_and_max_stays_100(tmp_path):
    path = model(tmp_path, mapping={'xhigh': 'high'})
    tokenizer = deepseek_tokenizer()
    golden = next(case for case in json.loads((FIXTURES / 'deepseek/v41-flash/token-reference.json').read_text())['cases']
                  if case['name'] == 'thinking-high')['expected_prompt']
    settings = load_reasoning_settings(path, tokenizer=tokenizer, reasoning_effort='xhigh')
    assert settings.template_kwargs['reasoning_effort'] == 'high'
    assert prompt(tokenizer, settings) == golden
    maximum = load_reasoning_settings(path, tokenizer=tokenizer, reasoning_effort='max')
    assert 'Reasoning Effort: 100 ' in prompt(tokenizer, maximum)


@pytest.mark.parametrize('mapping', [None, [], 'high', True, {'xhigh': 75}, {'': 'high'}, {'xhigh': ''}, {'xhigh': 'HIGH'}, {'xhigh': 'missing'}])
def test_bad_mapping_is_server_error_even_without_requested_effort(tmp_path, mapping):
    path = model(tmp_path)
    (path / 'gateway.json').write_text(json.dumps({'reasoning': {'mode': 'separate', 'effort_map': mapping}}))
    with pytest.raises(HTTPException) as error:
        load_reasoning_settings(path, tokenizer=deepseek_tokenizer())
    assert error.value.status_code == 500


def test_mapping_is_one_hop_and_does_not_mutate_shared_tokenizer(tmp_path):
    tokenizer = deepseek_tokenizer()
    original = dict(tokenizer.__dict__)
    mapped = model(tmp_path / 'mapped', mapping={'xhigh': 'high', 'high': 'low'})
    plain = model(tmp_path / 'plain')
    assert 'Reasoning Effort: 75 ' in prompt(tokenizer, load_reasoning_settings(
        mapped, tokenizer=tokenizer, reasoning_effort='xhigh'))
    assert 'Reasoning Effort: 50 ' in prompt(tokenizer, load_reasoning_settings(
        mapped, tokenizer=tokenizer, reasoning_effort='high'))
    assert 'Reasoning Effort: 75 ' in prompt(tokenizer, load_reasoning_settings(
        plain, tokenizer=tokenizer, reasoning_effort='high'))
    with pytest.raises(HTTPException) as error:
        load_reasoning_settings(plain, tokenizer=tokenizer, reasoning_effort='xhigh')
    assert error.value.status_code == 400
    assert tokenizer.__dict__ == original


def test_qwen_native_xhigh_is_unchanged_without_model_map(tmp_path):
    path = model(tmp_path, kind='qwen3_8')
    tokenizer = Tokenizer((FIXTURES / 'qwen/qwen38/chat_template.jinja').read_text())
    settings = load_reasoning_settings(path, tokenizer=tokenizer, reasoning_effort='xhigh')
    assert settings.template_kwargs['reasoning_effort'] == 'xhigh'
    assert 'Reasoning effort is set to xhigh.' in prompt(tokenizer, settings)


CUSTOM = "{% if add_generation_prompt %}{{ '<think>' if enable_thinking else '<think></think>' }}{% endif %}" + "{% if reasoning_effort is defined %}{% if reasoning_effort in ['native75', 'low'] %}{{ reasoning_effort }}{% else %}{{ raise_exception('unsupported') }}{% endif %}{% endif %}"


def test_configured_native_target_is_probed_and_cache_is_model_local(tmp_path):
    tokenizer = Tokenizer(CUSTOM)
    plain = model(tmp_path / 'plain', kind='custom')
    mapped = model(tmp_path / 'mapped', kind='custom', mapping={'xhigh': 'native75'})
    load_reasoning_settings(plain, tokenizer=tokenizer)
    settings = load_reasoning_settings(mapped, tokenizer=tokenizer, reasoning_effort='xhigh')
    assert settings.template_kwargs['reasoning_effort'] == 'native75'
    assert 'native75' in prompt(tokenizer, settings)
    with pytest.raises(HTTPException) as error:
        load_reasoning_settings(plain, tokenizer=tokenizer, reasoning_effort='native75')
    assert error.value.status_code == 400


def test_mapping_target_is_checked_against_selected_tools_template(tmp_path):
    class NamedTokenizer(Tokenizer):
        def get_chat_template(self, *, tools=None):
            return CUSTOM.replace('native75', 'other') if tools else CUSTOM

    tokenizer = NamedTokenizer(CUSTOM)
    path = model(tmp_path, kind='custom', mapping={'xhigh': 'native75'})
    assert load_reasoning_settings(path, tokenizer=tokenizer, reasoning_effort='xhigh')
    with pytest.raises(HTTPException) as error:
        load_reasoning_settings(path, tokenizer=tokenizer, tools=[{'type': 'function', 'function': {'name': 'lookup'}}])
    assert error.value.status_code == 500


@pytest.mark.parametrize('effort', ['XHIGH', ' xhigh', 'xhigh '])
def test_input_effort_is_never_silently_normalized(tmp_path, effort):
    path = model(tmp_path, mapping={'xhigh': 'high'})
    with pytest.raises(HTTPException) as error:
        load_reasoning_settings(path, tokenizer=deepseek_tokenizer(), reasoning_effort=effort)
    assert error.value.status_code == 400


@pytest.mark.parametrize('options', [{'enable_thinking': False}, {'force_disable': True}])
def test_mapped_effort_still_obeys_thinking_policy(tmp_path, options):
    path = model(tmp_path, mapping={'xhigh': 'high'})
    with pytest.raises(HTTPException) as error:
        load_reasoning_settings(path, tokenizer=deepseek_tokenizer(), reasoning_effort='xhigh', **options)
    assert error.value.status_code == 400
