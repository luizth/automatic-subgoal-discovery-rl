import numpy as np
from core import Policy, DeterministicPolicy
from plot import plot_deterministic_policy_in_tworooms


def get_random_deterministic_policy(env) -> DeterministicPolicy:
    return {s: env.action_space.sample() for s in range(env.observation_space.n)}


def get_random_policy(env) -> Policy:
    p = {s: np.zeros(env.action_space.n) for s in range(env.observation_space.n)}
    for s in p:
        p[s][env.action_space.sample()] = 1.0
    return p


def test_plot_deterministic_policy(env):
    policy = get_random_policy(env)
    plot_deterministic_policy_in_tworooms(env, policy, 63)
