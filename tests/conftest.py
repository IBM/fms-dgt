# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run tests that require live external services (e.g. LLM endpoints or credentials)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: mark test as requiring a live external service (skipped by default in CI)",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--live"):
        skip = pytest.mark.skip(reason="requires live external service — pass --live to run")
        for item in items:
            if item.get_closest_marker("live"):
                item.add_marker(skip)
