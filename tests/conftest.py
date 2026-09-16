"""Shared pytest fixtures for the test suite."""

from pathlib import Path

import pytest

from scripts.fixtures.generate import generate_fixtures

_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "generated"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Generate the fixture corpus once per test session.

    Generation runs unconditionally rather than only when the directory is
    missing. It is deterministic, offline and idempotent, so re-running is a
    no-op, while a conditional guard on "is the directory non-empty" would
    make a crashed partial run permanently sticky: the tree stays non-empty,
    never repairs itself, and fails later sessions with a message that points
    at the tests rather than at the stale fixtures.
    """
    generate_fixtures(_FIXTURE_ROOT)
    return _FIXTURE_ROOT
