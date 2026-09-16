"""Provider registry: the only place a provider name is branched on."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T", bound=type)

_REGISTRY: dict[tuple[str, str], type] = {}


class UnknownProviderError(KeyError):
    """Raised when a config names a provider that was never registered."""

    def __init__(self, kind: str, name: str | None) -> None:
        options = ", ".join(registered(kind)) or "<none registered>"
        super().__init__(f"unknown {kind} provider {name!r}; registered options: {options}")

    def __str__(self) -> str:
        return self.args[0]


def register(kind: str, name: str) -> Callable[[T], T]:
    def decorator(cls: T) -> T:
        _REGISTRY[(kind, name)] = cls
        return cls

    return decorator


def registered(kind: str) -> tuple[str, ...]:
    return tuple(sorted(n for k, n in _REGISTRY if k == kind))


def build(kind: str, cfg: dict[str, Any]) -> Any:
    name = cfg.get("provider")
    try:
        cls = _REGISTRY[(kind, name)]
    except KeyError:
        raise UnknownProviderError(kind, name) from None
    kwargs = {k: v for k, v in cfg.items() if k != "provider"}
    return cls(**kwargs)
