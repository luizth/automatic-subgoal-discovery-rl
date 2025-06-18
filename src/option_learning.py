from tqdm import tqdm
import numpy as np
import gymnasium as gym

from core import (
    Policy,
    Option,
    Subtask,
    RewardRespectingFeatureAttainment,
)
from utils import (
    delta_function,
    UWT,
    one_hot,
    softmax
)
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
        action_preferences = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
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
    w = np.zeros(env.observation_space.n)  # Critic
    theta = np.zeros(env.observation_space.n * env.action_space.n)  # Actor

    # Behavior policy (for off-policy learning)
    behavior_policy_probs = np.ones(env.action_space.n) / env.action_space.n

    # Reset the environment
    state, info = env.reset()
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
            state, info = env.reset()
            # start_state = state
            done = False
            stopping = False

            # Reset eligibility traces
            e = np.zeros_like(w)
            e_ = np.zeros_like(theta)

        # Behavior policy action selection
        action = np.random.choice(env.action_space.n, p=behavior_policy_probs)

        # Importance sampling ratio
        probs = policy(state, theta)
        rho = probs[action] / behavior_policy_probs[action]

        # Take step in the environment
        next_state, reward, done, _, _ = env.step(action)  # next_state, reward, done, _, _ = env.step(action) # - TwoRooms env (legacy)

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
        # action = np.random.choice(np.flatnonzero(probs == probs.max()))  # In case of ties, choose randomly among the best actions
        action = np.argmax(probs)
        policy[s] = one_hot(action, env.action_space.n)
    return policy


def learn_option_to_reach_subgoal(env: gym.Env, subgoal_state: int) -> Option:

    # Create a subtask to reach the subgoal
    subgoal = RewardRespectingFeatureAttainment(env, feature_attainment=subgoal_state)
    w, theta = actor_critic(subgoal, alpha=0.1, gamma=0.99, alpha_=0.1, number_of_steps=70000)
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


class OnlineOptionLearning:
    def __init__(
            self,
            task: Subtask,
            alpha=0.1,
            gamma=0.99,
            lmbda=0,
            alpha_=0.1,
            lmbda_=0,
            number_of_steps=50000,
            log=False,
            ):
        """ Online option learning using actor-critic algorithm """
        self.task = task
        self.env: gym.Env = task.env
        self.alpha = alpha
        self.gamma = gamma
        self.lmbda = lmbda
        self.alpha_ = alpha_
        self.lmbda_ = lmbda_
        self.number_of_steps = number_of_steps
        self.log = log

        # Initialize weight vector and theta for our linear V(s) and Policy (knowledge learned)
        self.w = np.zeros(self.env.observation_space.n)  # Critic
        self.theta = np.zeros(self.env.observation_space.n * self.env.action_space.n)  # Actor

        # Reset eligibility traces
        self.e = np.zeros_like(self.w)
        self.e_ = np.zeros_like(self.theta)

    def option(self) -> Policy:
        """ Convert learned parameters (theta) to softmax policy """
        return to_deterministic_policy(self.env, self.theta)

    def learn_one_step(self, state, action, reward, next_state, done, behavior_policy_probs):
        """ Learn one step of the actor-critic algorithm """

        # Check goal reached
        if done:
            # Reset the environment
            # state, info = env.reset()
            # start_state = state
            # done = False
            stopping = False

            # Reset eligibility traces
            self.e = np.zeros_like(self.w)
            self.e_ = np.zeros_like(self.theta)

        # Behavior policy action selection
        # action = np.random.choice(self.env.action_space.n, p=behavior_policy_probs)

        # Importance sampling ratio
        probs = self.policy(state, self.theta)
        rho = probs[action] / behavior_policy_probs[action]

        # Take step in the environment
        # next_state, reward, done, _, _ = env.step(action)  # next_state, reward, done, _, _ = env.step(action) # - TwoRooms env (legacy)

        # Extract features
        state_features = self.env.state_to_features(state)
        next_state_features = self.env.state_to_features(next_state)

        # GVFs
        # cumulant = task.c(state, next_state)  # cumulant for reward-respecting subtask is the reward
        stopping = self.task.B(next_state, self.w)
        stopping_value = self.task.z(next_state)

        V_s = self.value_function(state_features, self.w)
        V_s_next = self.value_function(next_state_features, self.w)

        # Critic - Compute TD error
        delta = delta_function(reward, stopping_value, V_s, V_s_next, int(stopping), self.gamma)
        alpha_delta = self.alpha * delta
        gamma_lambda = self.gamma * self.lmbda * (1 - int(stopping))
        self.w, self.e = UWT(self.w, self.e, state_features, alpha_delta, rho, gamma_lambda)

        # Actor - Compute policy gradient
        state_action_features = self.env.state_action_to_features(state, action)  # gradient is feature vector in linear case
        alpha_delta_ = self.alpha_ * delta
        gamma_lambda_ = self.gamma * self.lmbda_ * (1 - int(stopping))
        self.theta, self.e_ = UWT(self.theta, self.e_, state_action_features, alpha_delta_, rho, gamma_lambda_)


    def value_function(self, state, w):
        """Linear value function approximation: V(s) = w . state."""
        return np.dot(w, state)


    def policy(self, state, theta):
        """Softmax policy"""
        action_preferences = np.zeros(self.env.action_space.n)
        for a in range(self.env.action_space.n):
            features = self.env.state_action_to_features(state, a)
            action_preferences[a] = np.dot(theta, features)  # state-action value
        # Numerical stability
        max_value = np.max(action_preferences)
        action_preferences -= max_value
        # Softmax
        exp_values = np.exp(action_preferences)
        probs = exp_values / np.sum(exp_values)
        return probs


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
    subgoal1 = RewardRespectingFeatureAttainment(env, feature_attainment=env.h)
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
