from tqdm import tqdm
import numpy as np
import gymnasium as gym

from core import (
    Policy,
    Option,
    Subtask,
    RewardRespectingFeatureAttainment,
)
from utils import delta_function, UWT, one_hot, softmax

from env import TwoRooms


def actor_critic(
    task: Subtask,
    alpha=0.1,
    gamma=0.99,
    lmbda=0,
    alpha_=0.1,
    lmbda_=0,
    number_of_steps=50000,
    log=False,
):
    """Actor-Critic algorithm with linear function approximation"""

    def value_function(state, w):
        """Linear value function approximation: V(s) = w . state."""
        return np.dot(w, state)

    def policy(state, theta):
        """Softmax policy"""
        action_preferences = np.zeros(env.num_actions)
        for a in range(env.num_actions):
            features = env.state_action_to_features(state, a)
            action_preferences[a] = np.dot(theta, features)  # state-action value
        # Numerical stability
        max_value = np.max(action_preferences)
        action_preferences -= max_value
        # Softmax
        exp_values = np.exp(action_preferences)
        probs = exp_values / np.sum(exp_values)
        return probs

    # Env
    env: gym.Env = task.env

    # Initialize weight vector and theta for our linear V(s) and Policy (knowledge learned)
    w = np.zeros(env.num_states)  # Critic
    theta = np.zeros(env.num_states * env.num_actions)  # Actor

    # Behavior policy (for off-policy learning)
    behavior_policy_probs = np.ones(env.num_actions) / env.num_actions

    # Reset the environment
    state = env.reset()
    # start_state = state
    done = False
    stopping = False

    # Initialize eligibility traces to zero (episode-specific memory)
    e = np.zeros_like(w)
    e_ = np.zeros_like(theta)

    # Track number of steps
    for step in tqdm(range(number_of_steps)):

        # Check goal reached
        if done:
            # Reset the environment
            state = env.reset()
            # start_state = state
            done = False
            stopping = False

            # Reset eligibility traces
            e = np.zeros_like(w)
            e_ = np.zeros_like(theta)

        # Behavior policy action selection
        action = np.random.choice(env.num_actions, p=behavior_policy_probs)

        # Importance sampling ratio
        probs = policy(state, theta)
        rho = probs[action] / behavior_policy_probs[action]

        # Take step in the environment
        next_state, reward, done = env.step(action)  # next_state, reward, done, _, _ = env.step(action) # - TwoRooms env (legacy)

        # Extract features
        state_features = env.state_to_features(state)
        next_state_features = env.state_to_features(next_state)

        # GVFs
        # cumulant = task.c(state, next_state)  # cumulant for reward-respecting subtask is the reward
        stopping = task.B(next_state, w)
        stopping_value = task.z(next_state)

        V_s = value_function(state_features, w)
        V_s_next = value_function(next_state_features, w)

        # Critic - Compute TD error
        delta = delta_function(reward, stopping_value, V_s, V_s_next, int(stopping), gamma)
        alpha_delta = alpha * delta
        gamma_lambda = gamma * lmbda * (1 - int(stopping))
        w, e = UWT(w, e, state_features, alpha_delta, rho, gamma_lambda)

        # Actor - Compute policy gradient
        state_action_features = env.state_action_to_features(state, action)  # gradient is feature vector in linear case
        alpha_delta_ = alpha_ * delta
        gamma_lambda_ = gamma * lmbda_ * (1 - int(stopping))
        theta, e_ = UWT(theta, e_, state_action_features, alpha_delta_, rho, gamma_lambda_)

        # Move to next state
        state = next_state

    return w, theta


def to_policy(env: gym.Env, theta) -> Policy:
    """ Convert learned paremeters (theta) to softmax policy """
    policy: Policy = {}
    for s in range(env.observation_space.n):
        probs = softmax(env, s, theta)
        policy[s] = probs
    return policy


def to_deterministic_policy(env: gym.Env, theta) -> Policy:
    """ Convert learned paremeters (theta) to deterministic policy """
    policy: Policy = {}
    for s in range(env.observation_space.n):
        probs = softmax(env, s, theta)
        action = np.argmax(probs)
        policy[s] = one_hot(action, env.action_space.n)
    return policy


def get_option_to_reach_subgoal(env: gym.Env, subgoal_state: int) -> Option:

    # Create a subtask to reach the subgoal
    subgoal = RewardRespectingFeatureAttainment(env, feature_attainment=subgoal_state)
    w, theta, _, _ = actor_critic(subgoal, alpha=0.1, gamma=0.99, alpha_=0.1, number_of_steps=70000)
    policy: Policy = to_deterministic_policy(env, theta)

    # define initiation set as all states
    initiation_set = {s for s in range(env.observation_space.n)}

    # define termination condition
    def termination(state):
        return state == env.goal_state or state == subgoal_state

    option = Option(
        id=f"option_to_{subgoal_state}",
        initiation_set=initiation_set,
        policy=policy,
        termination=termination,
    )
    return option


if __name__ == "__main__":

    """
    This section of code reveals some bad interactions of subgoal and environment classes (see policy 1 and 2)
    This bad interactions may affect learning and are clearly visible in plotting the policy for both cases.
    """

    # Plot learned policies
    from plot import plot_deterministic_policy

    # Create the environment
    env = TwoRooms(start_state=24, goal_state=68, negative_states_config="none")

    # # Create a subtask to reach the hallway as subgoal
    subgoal1 = RewardRespectingFeatureAttainment(env, feature_attainment=env.hallway_state)
    w, theta = actor_critic(subgoal1, alpha=0.1, gamma=0.99, alpha_=0.1, number_of_steps=50000)
    policy1: Policy = to_deterministic_policy(env, theta)
    plot_deterministic_policy(env, policy1, subgoal1)

    # # Create a subtask to reach goal state as subgoal
    subgoal2 = RewardRespectingFeatureAttainment(env, feature_attainment=env.goal_state)
    w, theta = actor_critic(subgoal2, alpha=0.1, gamma=0.99, alpha_=0.1, number_of_steps=50000)
    policy2: Policy = to_deterministic_policy(env, theta)
    plot_deterministic_policy(env, policy2, subgoal2)

    # Create a subtask to reach other as subgoal
    subgoal3 = RewardRespectingFeatureAttainment(env, feature_attainment=63)
    w, theta = actor_critic(subgoal3, alpha=0.1, gamma=0.99, alpha_=0.1, number_of_steps=50000)
    policy3: Policy = to_deterministic_policy(env, theta)
    plot_deterministic_policy(env, policy3, subgoal3)
