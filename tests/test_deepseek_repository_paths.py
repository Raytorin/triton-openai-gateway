"""Resolve S3 repository aliases without hiding bad explicit checkpoint paths."""
from copy import deepcopy
import pytest

from test_deepseek_checkpoint import helper, snapshot


@pytest.fixture
def repository(tmp_path):
    version = tmp_path / 'folderMaterialized' / '1'
    version.mkdir(parents=True)
    return version, tmp_path / 'models-active'


def resolve(config, version, active, **context):
    return helper().resolve_repository_config(
        config,
        model_dir=version,
        model_name=context.get('model_name', 'selected'),
        model_version=context.get('model_version', '1'),
        active_dir=active,
    )


@pytest.mark.parametrize('suffix', ['selected', 'selected/1'])
def test_unpublished_alias_uses_materialized_version(repository, suffix):
    version, active = repository
    config = {'model': str(active / suffix), 'tokenizer': str(active / suffix)}

    result = resolve(config, version, active)

    assert result == {'model': str(version), 'tokenizer': str(version)}
    assert not active.exists()


def test_default_active_directory_rebases_the_same_model(repository):
    version, _ = repository

    result = helper().resolve_repository_config(
        {'model': '/tmp/models-active/selected/1'},
        model_dir=version,
        model_name='selected',
        model_version='1',
    )

    assert result == {'model': str(version)}


def test_custom_active_directory_is_used_verbatim(repository, tmp_path):
    version, _ = repository
    active = tmp_path / 'custom-active'

    result = resolve({'model': str(active / 'selected/1')}, version, active)

    assert result == {'model': str(version)}


@pytest.mark.parametrize('suffix', ['selected', 'selected/1'])
def test_stale_active_symlink_cannot_select_the_previous_copy(repository, tmp_path, suffix):
    version, active = repository
    previous = tmp_path / 'folderPrevious' / '1'
    previous.mkdir(parents=True)
    active.mkdir()
    (active / 'selected').symlink_to(previous, target_is_directory=True)

    result = resolve({'model': str(active / suffix)}, version, active)

    assert result == {'model': str(version)}
    assert (active / 'selected').resolve() == previous


@pytest.mark.parametrize('alias_field', ['model', 'tokenizer'])
def test_model_and_tokenizer_are_resolved_independently(repository, tmp_path, alias_field):
    version, active = repository
    other_field = 'tokenizer' if alias_field == 'model' else 'model'
    external = tmp_path / 'explicit-external-snapshot'
    config = {alias_field: str(active / 'selected/1'), other_field: str(external)}

    result = resolve(config, version, active)

    assert result[alias_field] == str(version)
    assert result[other_field] == str(external)
    assert not external.exists()


def test_absent_tokenizer_keeps_vllm_model_default(repository):
    version, active = repository

    result = resolve({'model': str(active / 'selected/1')}, version, active)

    assert result == {'model': str(version)}
    assert 'tokenizer' not in result


def test_resolver_returns_a_fresh_dictionary_without_mutating_input(repository):
    version, active = repository
    config = {
        'model': str(active / 'selected/1'),
        'tokenizer': str(active / 'selected'),
        'revision': 'explicit-revision',
        'language_model_only': True,
        'engram_config': {'cpu_offload': True},
    }
    original = deepcopy(config)

    result = resolve(config, version, active)

    assert result is not config
    assert config == original
    assert result == {**original, 'model': str(version), 'tokenizer': str(version)}


def test_direct_physical_checkpoint_path_is_unchanged(repository):
    version, active = repository
    config = {'model': str(version), 'tokenizer': str(version)}

    result = resolve(config, version, active)

    assert result == config
    assert result is not config


def test_missing_explicit_external_model_is_never_replaced_by_the_checkout(repository, tmp_path):
    version, active = repository
    missing = tmp_path / 'weights' / 'selected' / '1'

    result = resolve({'model': str(missing)}, version, active)

    assert result == {'model': str(missing)}
    with pytest.raises(ValueError, match='model'):
        helper().validate_checkpoint(result)


def test_missing_explicit_external_tokenizer_is_not_replaced_by_the_model(repository, tmp_path, snapshot):
    version, active = repository
    missing = tmp_path / 'missing-explicit-tokenizer'
    config = {'model': str(snapshot), 'tokenizer': str(missing)}

    result = resolve(config, version, active)

    assert result == config
    with pytest.raises(ValueError, match='tokenizer'):
        helper().validate_checkpoint(result)


@pytest.mark.parametrize('field', ['model', 'tokenizer'])
@pytest.mark.parametrize('suffix', [
    'another-model/1',
    'selected/2',
    'selected/1/extra',
    'selected/../selected/1',
    'selected/1/..',
    '../outside/1',
])
def test_malformed_active_alias_is_rejected(repository, field, suffix):
    version, active = repository
    config = {'model': str(version), field: str(active) + '/' + suffix}

    with pytest.raises(ValueError):
        resolve(config, version, active)


@pytest.mark.parametrize('name', ['', '.', '..', 'selected/other', '../selected', None])
def test_repository_alias_requires_a_single_model_name(repository, name):
    version, active = repository

    with pytest.raises(ValueError):
        resolve({'model': str(active / 'selected/1')}, version, active, model_name=name)


@pytest.mark.parametrize('model_version', ['', 'two', '2', '../1'])
def test_repository_context_version_must_match_the_materialized_directory(repository, model_version):
    version, active = repository

    with pytest.raises(ValueError):
        resolve({'model': str(active / 'selected/1')}, version, active, model_version=model_version)


def test_alias_cannot_bind_an_unmaterialized_version(repository):
    version, active = repository
    version.rmdir()

    with pytest.raises(ValueError):
        resolve({'model': str(active / 'selected/1')}, version, active)


@pytest.mark.parametrize('symlink_component', ['version', 'checkout'])
def test_alias_cannot_escape_through_a_materialized_directory_symlink(repository, tmp_path, symlink_component):
    version, active = repository
    external = tmp_path / 'external' / '1'
    external.mkdir(parents=True)
    version.rmdir()
    if symlink_component == 'version':
        version.symlink_to(external, target_is_directory=True)
    else:
        version.parent.rmdir()
        version.parent.symlink_to(external.parent, target_is_directory=True)

    with pytest.raises(ValueError):
        resolve({'model': str(active / 'selected/1')}, version, active)


@pytest.mark.parametrize('variant', ['double-slash', 'dot', 'parent-traversal'])
def test_obfuscated_namespace_prefix_cannot_use_an_old_alias(repository, variant):
    version, active = repository
    parent = str(active.parent)
    if variant == 'double-slash':
        value = parent + '//models-active/selected/1'
    elif variant == 'dot':
        value = parent + '/./models-active/selected/1'
    else:
        value = parent + '/other/../models-active/selected/1'
    with pytest.raises(ValueError):
        resolve({'model': value}, version, active)


@pytest.mark.parametrize('field', ['model', 'tokenizer'])
@pytest.mark.parametrize('suffix', ['foreign/1', 'selected/2', 'selected/1'])
def test_double_leading_slash_is_not_an_external_snapshot(repository, field, suffix):
    version, active = repository
    config = {'model': str(version), field: '/' + str(active / suffix)}
    with pytest.raises(ValueError):
        resolve(config, version, active)
