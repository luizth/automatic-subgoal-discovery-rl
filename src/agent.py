import random
from abc import ABC, abstractmethod
from typing import List

import numpy as np
from tqdm import tqdm

from core import Option
from env import NavigationEnv
from experience_buffer import ExperienceBuffer


class AgentWithOptions(ABC):
    env: NavigationEnv
    eb: ExperienceBuffer
    options_size: int

    @abstractmethod
    def add_option(self, option: Option):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def run_episode(self):
        raise NotImplementedError()


class SMDPQLearning(AgentWithOptions):

    def __init__(
            self,
            env: NavigationEnv,
            options: List[Option],
            learning_rate: float = 0.1,  # Learning rate
            discount_factor: float = 0.99,  # Discount factor
            exploration_rate: float = 1.0,  # Exploration rate
            min_exploration_rate: float = 0.1,
            exploration_decay: float = 0.99,
            store_experience: bool = False,
            log: bool = False
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

        self.q_table = np.zeros((env.observation_space.n, len(options)))
        # self.prob_bonus = np.zeros(len(options))
        self.eb = None

        # Initialize the experience buffer if required
        if store_experience:
            self.eb = ExperienceBuffer()

        self.log = log

    @property
    def options_size(self):
        """Return the number of options available"""
        return len(self.options)

    def copy(self, copy_qtable=False):
        """Create a copy of the agent with the same environment and options"""
        new_agent = SMDPQLearning(
            env=self.env,
            options=self.options.copy(),
            learning_rate=self.learning_rate,
            discount_factor=self.discount_factor,
            exploration_rate=self.initial_exploration_rate,
            min_exploration_rate=self.min_exploration_rate,
            exploration_decay=self.exploration_decay,
            store_experience=(self.eb is not None),
            log=self.log
        )
        if copy_qtable:
            # Copy the Q-table if requested
            new_agent.q_table = self.q_table.copy()
        return new_agent

    def reset(self):
        """Reset the Q-table and exploration rate"""
        self.env.reset()
        self.options = self.initial_options
        self.q_table = np.zeros((self.env.observation_space.n, len(self.options)))
        self.exploration_rate = self.initial_exploration_rate

        # Clear the experience buffer if it exists
        if self.eb is not None:
            self.eb.clear()

    def add_option(self, option: Option, initial_q_value=None):
        """Add a new option to the Q-learning agent"""
        self.options.append(option)

        # Expand the Q-table to accommodate the new option
        if initial_q_value is not None:
            o_q_value = initial_q_value
        else:
            o_q_value = self.q_table.min()
        self.q_table = np.hstack( ( self.q_table, np.ones((self.env.observation_space.n, 1)) * o_q_value ) )

    def decay_exploration_rate(self):
        """Decay the exploration rate to gradually shift from exploration to exploitation"""
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

    def choose_option(self, state) -> Option:
        """Choose an option based on the exploration-exploitation trade-off"""
        if random.uniform(0, 1) < self.exploration_rate:
            # Exploration: choose a random option
            return np.random.choice(self.options)
        else:
            # Exploitation: choose the best option based on Q-values
            option_i = np.random.choice(np.flatnonzero(self.q_table[state] == self.q_table[state].max()))
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

    def run(self, number_of_steps=50000, episode_length=None):
        """Run the agent for a specified number of steps"""
        # Reset the environment
        initial_state, info = self.env.reset()
        state = initial_state

        # Statistics
        returns = np.zeros(number_of_steps)
        rewards = np.zeros(number_of_steps)
        total_rewards = np.zeros(number_of_steps)

        # Execute episode
        G = 0
        ep_steps = 0
        total_reward = 0
        done = False
        for step in tqdm(range(number_of_steps)):

            # Check if episode ended
            if done:
                # Reset the environment
                initial_state, info = self.env.reset()
                state = initial_state
                G = 0
                ep_steps = 0
                done = False

            # Choose an option based on the current state
            option = self.choose_option(state)  # policy over options
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

                # Update the cumulative return
                G += self.discount_factor ** (ep_steps) * reward
                ep_steps += 1
                total_reward += reward

                # Increase option step
                option_k += 1

                # Update option reward model
                option_r += self.discount_factor ** (option_k-1) * reward

                # Check if the option terminates
                if option.terminate(next_state):
                    option_done = True

                # Step to next state
                state = next_state

            # Statistics
            returns[step] = G
            rewards[step] = reward
            total_rewards[step] = total_reward

            # Store the experience in the buffer if it exists
            if self.eb is not None and next_state != self.env.goal_transition_state:
                self.eb.add((option_s, action, next_state))

            # Update the Q-value for the option
            self.update_q_value(option_s, option_i, option_r, next_state, done, option_k)

            if episode_length is not None and step >= episode_length:
                done = True

        return returns, rewards, total_rewards

    def run_episode(self, max_steps=None):
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
            option = self.choose_option(state)  # policy over options
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

            # Store the experience in the buffer if it exists
            if self.eb is not None and next_state != self.env.goal_transition_state:
                self.eb.add((option_s, action, next_state))

            # Update the Q-value for the option
            self.update_q_value(option_s, option_i, option_r, next_state, done, option_k)

            if max_steps is not None and steps >= max_steps:
                done = True

        return initial_state, total_reward, steps, self.env.goal_reached(), trajectory
