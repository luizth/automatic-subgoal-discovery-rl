import random
from typing import List

import numpy as np
import gymnasium as gym

from core import Option
from env import NavigationEnv, get_primitive_actions_as_options


class SMDPQLearning:

    def __init__(
            self,
            env: NavigationEnv,
            options: List[Option],
            learning_rate: float = 0.1,  # Learning rate
            discount_factor: float = 0.99,  # Discount factor
            exploration_rate: float = 1.0,  # Exploration rate
            min_exploration_rate: float = 0.1,
            exploration_decay: float = 0.99
            ):
        self.env = env
        self.initial_options = options
        self.options = options

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.initial_exploration_rate = exploration_rate
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay = exploration_decay

        if isinstance(env, NavigationEnv):
            self.q_table = np.zeros((env.num_states, len(options)))
        elif isinstance(env, gym.Env):
            self.q_table = np.zeros((env.observation_space.n, len(options)))

    def reset(self):
        """Reset the Q-table and exploration rate"""
        self.env.reset()
        self.options = self.initial_options
        if isinstance(self.env, NavigationEnv):
            self.q_table = np.zeros((self.env.num_states, len(self.options)))
        elif isinstance(self.env, gym.Env):
            self.q_table = np.zeros((self.env.observation_space.n, len(self.options)))
        self.exploration_rate = self.initial_exploration_rate

    def add_option(self, option: Option):
        """Add a new option to the Q-learning agent"""
        self.options.append(option)
        # Expand the Q-table to accommodate the new option
        self.q_table = np.hstack((self.q_table, np.zeros((self.env.num_states, 1))))

    def decay_exploration_rate(self):
        """Decay the exploration rate to gradually shift from exploration to exploitation"""
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

    def choose_option(self, state) -> Option:
        """Choose an option based on the exploration-exploitation trade-off"""
        if random.uniform(0, 1) < self.exploration_rate:
            # Exploration: choose a random option
            return np.random.choice(self.options)
        else:
            if all([e1 == e2 for e1, e2 in zip(self.q_table[state], self.q_table[state][1:])]):
                # If all Q-values are equal, choose a random option
                return np.random.choice(self.options)
            # Exploitation: choose the best option based on Q-values
            option_i = np.argmax(self.q_table[state])
            return self.options[option_i]

    def update_q_value(self, state, option_index, reward, next_state, done, option_k=1):
        """Update the Q-value for the given state and option index"""
        if done:
            td_target = reward
        else:
            best_next_option_index = np.argmax(self.q_table[next_state])
            td_target = reward + (self.discount_factor ** option_k) * self.q_table[next_state][best_next_option_index]
        td_error = td_target - self.q_table[state][option_index]
        self.q_table[state][option_index] += self.learning_rate * td_error

    def run_episode(self):
        """Run a single episode of the environment"""
        # Reset the environment
        initial_state, info = self.env.reset()
        state = initial_state

        # Statistics
        total_reward = 0
        steps = 0
        trajectory = [state]

        # Execute episode
        done = False
        while not done:

            # Choose an option based on the current state
            option = self.choose_option(state)
            option_i = self.options.index(option)

            # From executing option
            option_s = state
            option_r = 0
            option_k = 0
            option_done = False
            while not option_done and not done:
                # Choose an action using the option's policy
                action = option.choose_action(state)
                next_state, reward, done, trunc, info = self.env.step(action)

                # Increase option step
                option_k += 1

                # Update option reward model
                option_r += self.discount_factor ** (option_k-1) * reward

                # Check if the option terminates
                if option.terminate(next_state):
                    option_done = True

                # Step to next state
                state = next_state
                total_reward += reward

                steps += 1
                trajectory.append(state)

            # Update the Q-value for the option
            self.update_q_value(option_s, option_i, option_r, next_state, done, option_k)

        return initial_state, total_reward, steps, self.env.goal_reached(), trajectory

    def train(self, episodes=1000, log=False):
        """Train the agent for a specified number of episodes"""

        # Initialize statistics
        steps_to_goal = np.zeros(episodes)
        rewards = np.zeros(episodes)

        for episode in range(episodes):
            episode_reward = 0

            # Reset the env
            state, info = self.env.reset()

            # Execute episode
            done = False
            while not done:

                # Choose an option based on the current state
                option = self.choose_option(state)
                option_i = self.options.index(option)

                if option_i == 4:
                    # print(end=".")
                    pass

                option_s = state
                option_r = 0
                option_k = 0
                option_done = False
                while not option_done and not done:
                    # Choose an action using the option's policy
                    action = option.choose_action(state)
                    next_state, reward, done, trunc, info = self.env.step(action)

                    # Increase option step
                    option_k += 1

                    # Update option reward model
                    option_r += self.discount_factor ** (option_k-1) * reward

                    # Check if the option terminates
                    if option.terminate(next_state):
                        option_done = True

                    # Step to next state
                    state = next_state
                    episode_reward += reward

                # Update the Q-value for the option
                self.update_q_value(option_s, option_i, option_r, next_state, done, option_k)

            # Decay the exploration rate
            self.decay_exploration_rate()

            # Store statistics
            steps_to_goal[episode] = self.env.steps
            rewards[episode] = episode_reward

            if log:
                print(f"Episode {episode + 1}/{episodes}. Steps to end episode: {self.env.steps}. Start state Q values: {self.q_table[self.env.start_state]}")

        return steps_to_goal


if __name__ == "__main__":
    from env import TwoRooms

    # Create the TwoRooms environment
    env = TwoRooms(start_state=24, goal_state=68, negative_states_config="default", max_steps=1000)

    # Get the primitive actions as options
    primitive_options: List[Option] = get_primitive_actions_as_options(env)

    # Create the Q-learning agent
    agent = SMDPQLearning(
        env,
        primitive_options,
        learning_rate=0.1,
        discount_factor=0.99,
        exploration_rate=1.0,
        min_exploration_rate=0.1,
        exploration_rate_decay=0.99
    )
