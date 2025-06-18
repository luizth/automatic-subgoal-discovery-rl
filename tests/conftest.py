from confmodules import load_modules

load_modules()

import pytest
from env import OpenRoom, TwoRooms, FourRooms


def pytest_addoption(parser):
    parser.addoption(
        "--log",
        action="store_true",
        default=False,
        help="Enable logging for the test run"
    )
    parser.addoption(
        "--env",
        type=str,
        default="TwoRooms",
        choices=["OpenRoom", "TwoRooms", "FourRooms"],
        help="Specify the environment to use for tests (e.g., OpenRoom, TwoRooms, FourRooms)"
    )

@pytest.fixture(scope="session")
def log(request):
    return request.config.getoption("--log")

@pytest.fixture(scope="session")
def env(request):
    _env = request.config.getoption("--env")
    if _env == "OpenRoom":
        return OpenRoom()
    elif _env == "TwoRooms":
        return TwoRooms(
            size=(6,12),
            start_states=[24],
            goal_states=[68],
            hallway_height=2,
            negative_states_config="none",
            sparse_rewards=True,
            max_steps=None
        )
    elif _env == "FourRooms":
        return FourRooms()
    else:
        raise ValueError(f"Unknown environment: {_env}")
