import pytest

from porsuk.core.registry import (
    UnknownProviderError,
    build,
    register,
    registered,
)


@register("widget", "alpha")
class _Alpha:
    def __init__(self, size: int = 1, label: str = "a"):
        self.size = size
        self.label = label


@register("widget", "beta")
class _Beta:
    def __init__(self, size: int = 2):
        self.size = size


def test_build_constructs_registered_class_from_config():
    obj = build("widget", {"provider": "alpha", "size": 7, "label": "z"})
    assert isinstance(obj, _Alpha)
    assert obj.size == 7
    assert obj.label == "z"


def test_build_strips_provider_key_before_construction():
    obj = build("widget", {"provider": "beta", "size": 3})
    assert obj.size == 3


def test_unknown_provider_names_the_request_and_the_options():
    with pytest.raises(UnknownProviderError) as exc:
        build("widget", {"provider": "gamma"})
    message = str(exc.value)
    assert "gamma" in message
    assert "alpha" in message
    assert "beta" in message


def test_missing_provider_key_is_an_error():
    with pytest.raises(UnknownProviderError):
        build("widget", {"size": 1})


def test_registered_lists_names_for_a_kind():
    assert registered("widget") == ("alpha", "beta")


def test_register_returns_the_class_unchanged():
    assert _Alpha.__name__ == "_Alpha"
