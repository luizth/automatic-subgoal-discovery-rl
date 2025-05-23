from confmodules import load_modules

load_modules()

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--log",
        action="store_true",
        default=False,
        help="Enable logging for the test run"
    )

@pytest.fixture(scope="session")
def log(request):
    return request.config.getoption("--log")
