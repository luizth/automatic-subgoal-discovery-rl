import numpy as np
import gymnasium as gym

from core import Option


class TwoRooms(gym.Env):

    def __init__(self,
            size=(6, 12),
            start_state=None,
            goal_state=68,
            hallway_height=2,
            negative_states_config="default",
            max_steps=1000,
            sparse_rewards=True,
            ):

        """
        Two rooms environment with a hallway in between.

        The grid is represented as follows:

        0  1  2  3  4  5  -  6  7  8  9 10 11
        12 13 14 15 16 17 - 18 19 20 21 22 23
        24 25 26 27 28 29 - 30 31 32 33 34 35
        36 37 38 39 40 41 - 42 43 44 45 46 47
        48 49 50 51 52 53 - 54 55 56 57 58 59
        60 61 62 63 64 65 - 66 67 68 69 70 71

        G = 68, then

        0  1  2  3  4  5  -   6  7  8  9 10 11
        12 13 14 15 16 17 -  18 19 20 21 22 23
        24 25 26 27 28 29 68 30 31 32 33 34 35
        36 37 38 39 40 41 -  42 43 44 45 46 47
        48 49 50 51 52 53 -  54 55 56 57 58 59
        60 61 62 63 64 65 -  66 67  G 69 70 71

        """

        if size[0] % 2 != 0:
            raise ValueError("Width of the grid must be even")
        self.size = size
        self.rows = size[0]
        self.cols = size[1]
        self.room_size = self.cols // 2

        self.num_states = self.cols*self.rows  # non-terminal states
        self.num_actions = 4  # Up, Down, Left, Right

        self.goal_state = goal_state
        self.goal_transition_state = -1  # Reference to goal state
        self.hallway_height = hallway_height
        self.hallway_state = self.cols // 2 + self.cols * hallway_height  # Center

        self.negative_states_config = negative_states_config
        if negative_states_config == "none":
            self.negative_states = []
        elif negative_states_config == "default":
            self.negative_states = [1,2,3,4,13,14,15,16,25,26,27,28,37,38,39,40,49,50,51,52]  # [self.hallway_state - 1, self.hallway_state + 1]  # Adjacent to hallway
        elif negative_states_config == "left_square":
            self.negative_states = [13,14,15,16,25,26,27,28,37,38,39,40,49,50,51,52]
        elif negative_states_config == "right_square":
            self.negative_states = [19,20,21,22,31,32,33,34,43,44,45,46,55,56,57,58]
        elif negative_states_config == "two_squares":
            self.negative_states = [13,14,15,16,25,26,27,28,37,38,39,40,49,50,51,52] + [19,20,21,22,31,32,33,34,43,44,45,46,55,56,57,58]
        else:
            raise ValueError("Invalid negative states configuration")

        upper_wall = [j for j in range(self.cols)]
        lower_wall = [self.cols * (self.rows-1) + j for j in range(self.cols)]
        walls_to_left = [self.cols * i for i in range(self.rows)] + [self.cols * i + self.rows for i in range(self.rows)]
        walls_to_right = [self.cols * i - 1 for i in range(1, self.rows+1)] + [self.cols * i + self.rows - 1 for i in range(self.rows)]

        # State transition dynamics
        self.transitions = np.zeros((self.num_states, self.num_actions), dtype=int)
        for s in range(self.num_states):
            row, col = s // self.cols, s % self.cols
            # Up
            self.transitions[s, 0] = s if row == 0 or s in upper_wall else s - self.cols
            # Down
            self.transitions[s, 1] = s if row == self.rows-1 or s in lower_wall else s + self.cols
            # Left
            self.transitions[s, 2] = s if col == 0 or s in walls_to_left else s - 1
            # Right
            self.transitions[s, 3] = s if col == self.cols-1 or s in walls_to_right else s + 1

        # To goal transition dynamics
        above_goal = self.goal_state - self.cols
        if above_goal in self.transitions:
            self.transitions[above_goal, 1] = -1
        below_goal = self.goal_state + self.cols
        if below_goal in self.transitions:
            self.transitions[below_goal, 0] = -1
        left_goal = self.goal_state - 1
        if left_goal in self.transitions and left_goal not in walls_to_right:
            self.transitions[left_goal, 3] = -1
        right_goal = self.goal_state + 1
        if right_goal in self.transitions and right_goal not in walls_to_left:
            self.transitions[right_goal, 2] = -1

        # Hallway transition dynamics
        left_hallway = self.hallway_state - 1
        self.transitions[self.goal_state, 0] = self.goal_state
        self.transitions[self.goal_state, 1] = self.goal_state
        self.transitions[self.goal_state, 2] = left_hallway
        self.transitions[self.goal_state, 3] = self.hallway_state
        # adjacent to hallway
        self.transitions[left_hallway, 3] = self.goal_state
        self.transitions[self.hallway_state, 2] = self.goal_state

        # Env configuration
        self.start_state = start_state
        self.max_steps = max_steps
        self.sparse_rewards = sparse_rewards

        # Agent monitoring
        self.init_state()
        self.steps = 0
        self.trajectory = []

        # Gym environment configuration
        self.observation_space = gym.spaces.Discrete(self.num_states)
        self.action_space = gym.spaces.Discrete(self.num_actions)

    def step(self, action):
        """Take a step in the environment"""
        if action < 0 or action >= self.num_actions:
            raise ValueError("Invalid action")

        self.trajectory.append(self.state)

        # Get the next state based on the current state and action
        next_state = self.transitions[self.state, action]

        # Check if the next state is terminal
        done = self.is_terminal(next_state)

        # Get the reward for taking the action
        reward = self.get_reward(next_state)

        # Update the current state
        self.state = next_state
        self.steps += 1

        return next_state, reward, done, False, {}

    def reset(self):
        """Reset the environment to a random state"""
        self.init_state()
        self.trajectory = []
        self.steps = 0
        return self.state, {}

    def render(self):
        for s in range(env.num_states):
            to_print = '.'
            if s in env.negative_states:
                to_print = 'N'
            if s in self.trajectory:
                to_print = '*'
            if s == start_state:
                to_print = 'S'
            if s == env.goal_state:
                to_print = 'G'
            if s == self.state:
                to_print = 'X'

            print(to_print, end=' ')

            if (s+1) % env.cols == 0:
                print()

        if self.state == self.hallway_state:
            print("Hallway: ", self.state)

    @property
    def states(self):
        """Get the states of the environment"""
        return np.arange(self.num_states)

    def goal_reached(self):
        """Check if the goal state is reached"""
        return self.state == self.goal_transition_state

    def copy(self):
        """Create a copy of the environment"""
        env_copy = TwoRooms(
            size=self.size,
            start_state=self.start_state,
            goal_state=self.goal_state,
            hallway_height=self.hallway_height,
            negative_states_config=self.negative_states_config,
            max_steps=self.max_steps
        )
        return env_copy

    def get_action_distribution(self, state, action):
        """Get the action distribution for a given state and action"""
        if state == self.goal_transition_state:
            return np.zeros(self.num_actions)
        else:
            action_distribution = np.zeros(self.num_actions)
            action_distribution[action] = 1.0
            return action_distribution

    def get_reward(self, next_state):
        if next_state == self.goal_transition_state:
            return 1.0
        elif next_state in self.negative_states:
            return -1.0
        else:
            if self.sparse_rewards:
                return 0.0
            else:
                return -0.1

    def is_terminal(self, state):
        if self.max_steps is not None:
            return state == self.goal_transition_state or self.steps >= self.max_steps
        return state == self.goal_transition_state

    def state_to_features(self, state):
        if state == self.goal_transition_state:
            features = np.zeros(self.num_states)
        else:
            features = np.zeros(self.num_states)
            features[state] = 1.0
        return features

    def state_action_to_features(self, state, action):
        if state == self.goal_transition_state:
            features = np.zeros(self.num_states * self.num_actions)
        else:
            features = np.zeros(self.num_states * self.num_actions)
            features[state * self.num_actions + action] = 1.0
        return features

    def sample_state(self):
        """Sample a random state from the grid"""
        return np.random.randint(0, self.num_states)

    def sample_action(self):
        """Sample a random action"""
        return np.random.randint(0, self.num_actions)

    def init_state(self):
        """Initialize the environment to a random state"""
        if self.start_state and isinstance(self.start_state, list):
            self.state = np.random.choice(self.start_state)
        elif self.start_state and isinstance(self.start_state, int):
            self.state = self.start_state
        else:
            self.state = self.sample_state()
        return self.state


def get_primitive_actions_as_options(env: TwoRooms):
    options = []
    for a in range(env.num_actions):
        # Define the initiation set for the option
        initiation_set = {s for s in range(env.num_states)}

        # Define the policy for the option
        policy = {s: env.get_action_distribution(s, a) for s in initiation_set}

        # Define the termination condition for the option
        def termination(s):
            return True

        # Create the option
        options.append(
            Option(
                id=f"option_{a}",
                initiation_set=initiation_set,
                policy=policy,
                termination=termination
            )
        )
    return options


if __name__ == "__main__":
    env = TwoRooms(start_state=5, goal_state=6)

    start_state = env.reset()
    G = 0.
    done = False
    steps = 1
    while not done and steps <= 100:
        print()
        print("Step: ", steps)
        print("State: ", env.state)
        print("G: ", G)
        env.render()

        action = int(input("Action: "))
        next_state, reward, done, _, _ = env.step(action)
        G += reward
        steps += 1
