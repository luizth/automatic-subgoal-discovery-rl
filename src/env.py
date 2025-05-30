from abc import abstractmethod
import numpy as np
import gymnasium as gym

from core import Option

class RandomWalk(gym.Env):

    def __init__(self,
            n=4,
            start_state=0,
            goal_state=3,
            max_steps=None,
            sparse_rewards=False,
            negative_zone=False,
            ):

        """
        RandomWalk environment

        Four state: 0 - 1 - 2 - 3

        """

        # Store the environment parameters
        self.n = n
        self.start_state = start_state
        self.goal_state = goal_state
        self.max_steps = max_steps
        self.sparse_rewards = sparse_rewards
        self.negative_zone = negative_zone

        # Env configuration
        self.observation_space = gym.spaces.Discrete(n)
        self.action_space = gym.spaces.Discrete(2)  # Left, Right

        self.state = None
        self.init_state()
        self.steps = 0

        self.info = {
            "trajectory": [self.state],
            "goal_reached": False,
            "total_reward": 0
        }

        # State transition dynamics
        self.transitions = np.zeros((self.observation_space.n, self.action_space.n), dtype=int)
        for s in range(self.observation_space.n):
            # Left
            self.transitions[s, 0] = s if s == 0 else s - 1
            # Right
            self.transitions[s, 1] = s if s == n-1 else s + 1

        # Negative zone
        self.negative_reward_states = None

    def init_state(self):
        """Initialize the environment to a random state"""
        if self.start_state and isinstance(self.start_state, list):
            self.state = np.random.choice(self.start_state)
        elif self.start_state and isinstance(self.start_state, int):
            self.state = self.start_state
        else:
            self.state = np.random.randint(0, self.observation_space.n)
        return self.state

    def get_reward(self, next_state):
        """Get the reward for taking an action"""
        if next_state == self.goal_state:
            return 1.0
        elif self.negative_zone and next_state in self.negative_reward_states:
            return -1.0
        else:
            if self.sparse_rewards:
                return 0.0
            else:
                return -0.1

    def is_terminal(self, state):
        """Check if the state is terminal"""
        if self.max_steps is not None:
            return (state == self.goal_state) or self.steps >= self.max_steps
        return state == self.goal_state

    # gymnasium
    def reset(self, options: dict = {}):
        """Reset the environment to a random state"""
        if "state" in options:
            self.state = options["state"]
        else:
            self.init_state()
        self.steps = 0

        self.info = {
            "trajectory": [self.state],
            "goal_reached": False,
            "total_reward": 0
        }
        return self.state, self.info

    def step(self, action):
        """Take a step in the environment"""

        if action < 0 or action >= self.action_space.n:
            raise ValueError("Invalid action")

        # Get the next state based on the current state and action
        next_state = self.transitions[self.state, action]

        # Check if the next state is terminal
        done = self.is_terminal(next_state)

        # Get the reward for taking the action
        reward = self.get_reward(next_state)

        # Update the current state
        self.state = next_state

        self.info["trajectory"].append(self.state)
        self.info["total_reward"] += reward
        self.steps += 1

        return next_state, reward, done, False, self.info


# Abstract class for the rooms environment
class NavigationEnv(gym.Env):
    def __init__(self, *args, **kwds):
        self.observation_space = gym.spaces.Discrete(0)
        self.action_space = gym.spaces.Discrete(4)
        return super().__call__(*args, **kwds)

    @abstractmethod
    def get_action_distribution(self, state, action):
        raise NotImplementedError("get_action_distribution not implemented")

    @abstractmethod
    def goal_reached(self):
        raise NotImplementedError("goal_reached not implemented")


class TwoRooms(NavigationEnv):

    def __init__(self,
            size=(6, 12),
            start_state=None,
            goal_state=68,
            negative_states_config="default",
            max_steps=None,
            sparse_rewards=True,
            stochastic=False,
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

        # Store the environment parameters
        if size[1] % 2 != 0:
            raise ValueError("Width of the grid must be even")
        self.size = size
        self.rows = size[0]
        self.cols = size[1]
        self.room_size = self.cols // 2

        # Non-terminal state (hallway assumes goal state)
        num_states = self.rows * self.cols

        # Env configuration
        self.observation_space = gym.spaces.Discrete(num_states)
        self.action_space = gym.spaces.Discrete(4)  # Up, Down, Left, Right

        # State transition dynamics
        self.goal_state = goal_state
        self.goal_transition_state = -1  # Reference to goal state

        # Goal becomes a hallway
        self.h = goal_state
        self.hallway_height = 2

        # Configure negative areas
        self.negative_states_config = negative_states_config
        self.negative_states = self._configure_negative_states(negative_states_config)

        # Define room boundaries
        self._define_boundaries()

        # State transition dynamics
        self.transitions = self._build_transitions()

        # Env configuration
        self.start_state = start_state
        self.max_steps = max_steps
        self.sparse_rewards = sparse_rewards
        self.stochastic = stochastic

        # Agent monitoring
        self.init_state()
        self.steps = 0
        self.trajectory = []

    def _configure_negative_states(self, config):
        """Configure negative reward states based on configuration."""
        if config == "none":
            return {}
        elif config == "default":
            return {1,2,3,4,13,14,15,16,25,26,27,28,37,38,39,40,49,50,51,52}  # Adjacent to hallway
        elif config == "left_square":
            return {13,14,15,16,25,26,27,28,37,38,39,40,49,50,51,52}
        elif config == "right_square":
            return {19,20,21,22,31,32,33,34,43,44,45,46,55,56,57,58}
        elif config == "two_squares":
            return {13,14,15,16,25,26,27,28,37,38,39,40,49,50,51,52} + {19,20,21,22,31,32,33,34,43,44,45,46,55,56,57,58}
        else:
            raise ValueError("Invalid negative states configuration")

    def _define_boundaries(self):
        """Define the boundaries of the rooms and hallways."""
        self.upper_wall = [j for j in range(self.cols)]
        self.lower_wall = [self.cols * (self.rows-1) + j for j in range(self.cols)]
        self.walls_to_left = [self.cols * i for i in range(self.rows)] + [self.cols * i + self.rows for i in range(self.rows)]
        self.walls_to_right = [self.cols * i - 1 for i in range(1, self.rows+1)] + [self.cols * i + self.rows - 1 for i in range(self.rows)]

    def _build_transitions(self):
        """Build the state transition matrix."""
        # State transition dynamics
        transitions = np.zeros((self.observation_space.n, self.action_space.n), dtype=int)
        for s in range(self.observation_space.n):
            row, col = s // self.cols, s % self.cols
            # Up
            transitions[s, 0] = s if row == 0 or s in self.upper_wall else s - self.cols
            # Down
            transitions[s, 1] = s if row == self.rows-1 or s in self.lower_wall else s + self.cols
            # Left
            transitions[s, 2] = s if col == 0 or s in self.walls_to_left else s - 1
            # Right
            transitions[s, 3] = s if col == self.cols-1 or s in self.walls_to_right else s + 1

        # To goal transition dynamics
        above_goal = self.goal_state - self.cols
        below_goal = self.goal_state + self.cols
        left_goal = self.goal_state - 1
        right_goal = self.goal_state + 1
        if above_goal in transitions:
            transitions[above_goal, 1] = -1
        if below_goal in transitions:
            transitions[below_goal, 0] = -1
        if left_goal in transitions and left_goal not in self.walls_to_right:
            transitions[left_goal, 3] = -1
        if right_goal in transitions and right_goal not in self.walls_to_left:
            transitions[right_goal, 2] = -1

        # Hallway transition dynamics
        transitions[self.h, 0] = self.h
        transitions[self.h, 1] = self.h
        transitions[self.h, 2] = 29
        transitions[self.h, 3] = 30
        # adjacent to hallway
        transitions[29, 3] = self.h
        transitions[30, 2] = self.h

        return transitions

    def step(self, action):
        """Take a step in the environment"""
        if action < 0 or action >= self.action_space.n:
            raise ValueError("Invalid action")

        self.trajectory.append(self.state)

        # If stochastic, randomly choose the next action
        if self.stochastic:
            if np.random.rand() < 0.3:
                action = np.random.choice(self.action_space.n)

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

    def reset(self, options: dict = {}):
        """Reset the environment to a random state"""
        if "state" in options:
            self.state = options["state"]
        else:
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
            return np.zeros(self.action_space.n)
        else:
            action_distribution = np.zeros(self.action_space.n)
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
            features = np.zeros(self.observation_space.n)
        else:
            features = np.zeros(self.observation_space.n)
            features[state] = 1.0
        return features

    def state_action_to_features(self, state, action):
        if state == self.goal_transition_state:
            features = np.zeros(self.observation_space.n * self.action_space.n)
        else:
            features = np.zeros(self.observation_space.n * self.action_space.n)
            features[state * self.action_space.n + action] = 1.0
        return features

    def sample_state(self):
        """Sample a random state from the grid"""
        return np.random.randint(0, self.observation_space.n)

    def sample_action(self):
        """Sample a random action"""
        return np.random.randint(0, self.action_space.n)

    def init_state(self):
        """Initialize the environment to a random state"""
        if self.start_state and isinstance(self.start_state, list):
            self.state = np.random.choice(self.start_state)
        elif self.start_state and isinstance(self.start_state, int):
            self.state = self.start_state
        else:
            self.state = self.sample_state()
        return self.state


class FourRooms(NavigationEnv):
    """
    Four rooms environment with hallways connecting adjacent rooms.

    The grid is represented as follows:

    Room 1                   Room 2
    0   1  2  3  4 -  5  6  7  8  9
    10 11 12 13 14 - 15 16 17 18 19
    20 21 22 23 24 - 25 26 27 28 29
    30 31 32 33 34 - 35 36 37 38 39
    40 41 42 43 44 - 45 46 47 48 49
    -  -  -  -  -  - 50 51 52 53 54
    55 56 57 58 59 -  -  -  -  -  -
    60 61 62 63 64 - 65 66 67 68 69
    70 71 72 73 74 - 75 76 77 78 79
    80 81 82 83 84 - 85 86 87 88 89
    90 91 92 93 94 - 95 96 97 98 99
    Room 3                   Room 4

    when G = 75, then

    0   1  2  3  4   -  5  6   7  8  9
    10 11 12 13 14   - 15 16  17 18 19
    20 21 22 23 24 100 25 26  27 28 29
    30 31 32 33 34   - 35 36  37 38 39
    40 41 42 43 44   - 45 46  47 48 49
    - 75  -  -  -   - 50 51  52 53 54
    55 56 57 58 59   -  -  - 101  -  -
    60 61 62 63 64   - 65 66  67 68 69
    70 71 72 73 74   -  G 76  77 78 79
    80 81 82 83 84 102 85 86  87 88 89
    90 91 92 93 94   - 95 96  97 98 99
    """

    def __init__(self,
            size=(10, 10),
            start_state=None,
            goal_state=75,
            negative_states_config="default",
            max_steps=None,
            sparse_rewards=True,
            ):

        # Store the environment parameters
        if size[0] % 2 != 0 and size[1] % 2 != 0:
            raise ValueError("Width of the grid must be even")
        self.size = size
        self.rows = size[0]
        self.cols = size[1]

        # Non-terminal state + 3 hallways (one of them will be the goal)
        num_states = self.rows * self.cols + 3

        # Env configuration
        self.observation_space = gym.spaces.Discrete(num_states)
        self.action_space = gym.spaces.Discrete(4)  # Up, Down, Left, Right

        # State transition dynamics
        self.goal_state = goal_state
        self.goal_transition_state = -1  # Reference to goal state

        # Goal becomes a hallway
        self.h1 = goal_state
        self.h2 = 100
        self.h3 = 101
        self.h4 = 102

        # Configure negative areas
        self.negative_states_config = negative_states_config
        self.negative_states = self._configure_negative_states(negative_states_config)

        # Define room boundaries
        self._define_boundaries()

        # State transition dynamics
        self.transitions = self._build_transitions()

        # Env configuration
        self.start_state = start_state
        self.max_steps = max_steps
        self.sparse_rewards = sparse_rewards

        # Agent monitoring
        self.init_state()
        self.steps = 0
        self.trajectory = []

    def _get_room_for_state(self, state):
        """Determine which room a state belongs to."""
        if state in self.room_1:
            return 1
        elif state in self.room_2:
            return 2
        elif state in self.room_3:
            return 3
        elif state in self.room_4:
            return 4
        elif state in self.hallway_states:
            return "hallway"
        else:
            return "invalid"

    def _configure_negative_states(self, config):
        """Configure negative reward states based on configuration."""
        if config == "none":
            return set()
        elif config == "default":
            # Spread negative states in room 3
            return {51, 52, 53, 61, 62, 63, 71, 72, 73, 81, 82, 83}
        else:
            raise ValueError("Invalid negative states configuration")

    def _define_boundaries(self):
        """Define the boundaries of the rooms and hallways."""
        self._upper_wall = [j for j in range(self.cols)]
        room3_upper_wall = [self.cols * (self.rows // 2) + j for j in range(self.cols // 2)]
        room4_upper_wall = [self.cols * (self.rows // 2 + 1) + self.rows // 2 + j for j in range(self.cols // 2)]
        self._upper_wall += room3_upper_wall + room4_upper_wall

        self._lower_wall = [self.cols * (self.rows-1) + j for j in range(self.cols)]
        room1_lower_wall = [self.cols * (self.rows // 2 - 1) + j for j in range(self.cols // 2)]
        room2_lower_wall = [self.cols * (self.rows // 2) + self.rows // 2 + j for j in range(self.cols // 2)]
        self._lower_wall += room1_lower_wall + room2_lower_wall

        self._walls_to_left = [self.cols * i for i in range(self.rows)]
        room24_left_wall = [self.cols * i + self.rows // 2 for i in range(self.rows)]
        self._walls_to_left += room24_left_wall

        self._walls_to_right = [self.cols * i - 1 for i in range(1, self.rows+1)]
        room13_right_wall = [self.cols * i + self.rows // 2 - 1 for i in range(self.rows)]
        self._walls_to_right += room13_right_wall


    def _build_transitions(self):
        """Build the state transition matrix."""
        transitions = np.zeros((self.observation_space.n, self.action_space.n), dtype=int)
        for s in range(self.observation_space.n):

            # Keep the last 3 states as hallways
            if s >= self.observation_space.n - 3:
                transitions[s, 0] = s
                transitions[s, 1] = s
                transitions[s, 2] = s
                transitions[s, 3] = s
                continue

            row, col = s // self.cols, s % self.cols
            # Up
            transitions[s, 0] = s if row == 0 or s in self._upper_wall else s - self.cols
            # Down
            transitions[s, 1] = s if row == self.rows-1 or s in self._lower_wall else s + self.cols
            # Left
            transitions[s, 2] = s if col == 0 or s in self._walls_to_left else s - 1
            # Right
            transitions[s, 3] = s if col == self.cols-1 or s in self._walls_to_right else s + 1

        # To goal transition dynamics
        above_goal = self.goal_state - self.cols
        below_goal = self.goal_state + self.cols
        left_goal = self.goal_state - 1
        right_goal = self.goal_state + 1
        if above_goal in transitions and above_goal not in self._lower_wall:
            transitions[above_goal, 1] = -1
        if below_goal in transitions and below_goal not in self._upper_wall:
            transitions[below_goal, 0] = -1
        if left_goal in transitions and left_goal not in self._walls_to_right:
            transitions[left_goal, 3] = -1
        if right_goal in transitions and right_goal not in self._walls_to_left:
            transitions[right_goal, 2] = -1

        # Hallway transition dynamics
        transitions[self.h1, 0] = 41
        transitions[self.h1, 1] = 51
        transitions[self.h1, 2] = self.h1
        transitions[self.h1, 3] = self.h1
        transitions[41, 1] = self.h1
        transitions[51, 0] = self.h1

        transitions[self.h2, 0] = self.h2
        transitions[self.h2, 1] = self.h2
        transitions[self.h2, 2] = 24
        transitions[self.h2, 3] = 25
        transitions[24, 3] = self.h2
        transitions[25, 2] = self.h2

        transitions[self.h3, 0] = 57
        transitions[self.h3, 1] = 67
        transitions[self.h3, 2] = self.h3
        transitions[self.h3, 3] = self.h3
        transitions[57, 1] = self.h3
        transitions[67, 0] = self.h3

        transitions[self.h4, 0] = self.h4
        transitions[self.h4, 1] = self.h4
        transitions[self.h4, 2] = 84
        transitions[self.h4, 3] = 85
        transitions[84, 3] = self.h4
        transitions[85, 2] = self.h4

        return transitions

    def step(self, action):
        """Take a step in the environment."""
        if action < 0 or action >= self.action_space.n:
            raise ValueError(f"Invalid action: {action}")

        self.trajectory.append(self.state)

        # Get next state
        next_state = self.transitions[self.state, action]

        # Check if next state is terminal
        done = self.is_terminal(next_state)

        # Get reward
        reward = self.get_reward(next_state)

        # Update state
        self.state = next_state
        self.steps += 1

        return next_state, reward, done, False, {}

    def reset(self, options: dict = {}):
        """Reset environment to initial state."""
        if "state" in options:
            self.state = options["state"]
        else:
            self.init_state()
        self.trajectory = []
        self.steps = 0
        return self.state, {}

    def init_state(self):
        """Initialize the environment state."""
        if isinstance(self.start_state, list):
            self.state = np.random.choice(self.start_state)
        elif isinstance(self.start_state, int):
            self.state = self.start_state
        else:
            # Random start from navigable states
            self.state = np.random.choice(list(self.navigable_states))
        return self.state

    def get_reward(self, next_state):
        """Get reward for transitioning to next_state."""
        if next_state == self.goal_transition_state:
            return 1.0
        elif next_state in self.negative_states:
            return -1.0
        else:
            return 0.0 if self.sparse_rewards else -0.01

    def is_terminal(self, state):
        """Check if state is terminal."""
        if self.max_steps is not None and self.steps >= self.max_steps:
            return True
        return state == self.goal_transition_state

    def goal_reached(self):
        """Check if goal has been reached."""
        return self.state == self.goal_transition_state

    def state_to_features(self, state):
        """Convert state to feature vector (one-hot encoding)."""
        if state == self.goal_transition_state:
            features = np.zeros(self.observation_space.n)
        else:
            features = np.zeros(self.observation_space.n)
            if 0 <= state < self.observation_space.n:
                features[state] = 1.0
        return features

    def state_action_to_features(self, state, action):
        """Convert state-action pair to feature vector."""
        if state == self.goal_transition_state:
            features = np.zeros(self.observation_space.n * self.action_space.n)
        else:
            features = np.zeros(self.observation_space.n * self.action_space.n)
            if 0 <= state < self.observation_space.n and 0 <= action < self.action_space.n:
                features[state * self.action_space.n + action] = 1.0
        return features

    def get_action_distribution(self, state, action):
        """Get action distribution for given state and action."""
        if state == self.goal_transition_state:
            return np.zeros(self.action_space.n)
        else:
            action_dist = np.zeros(self.action_space.n)
            action_dist[action] = 1.0
            return action_dist

    def render(self):
        """Render the current state of the environment."""
        print("\nFourRooms Environment (6x6):")
        for row in range(self.rows):
            line = ""
            for col in range(self.cols):
                state = row * self.cols + col

                if state == self.state:
                    char = "  A"  # Agent
                elif state == self.goal_state:
                    char = "  G"  # Goal
                elif state in [self.h1, self.h2, self.h3, self.h4]:
                    char = "  H"  # Hallway
                elif state in self.negative_states:
                    char = "  N"  # Negative reward
                else:
                    str_state = str(state)
                    if len(str_state) == 1:
                        str_state = "  " + str_state
                    if len(str_state) == 2:
                        str_state = " " + str_state
                    char = str_state   # Empty navigable space

                line += char + " "
            print(line)

        print(f"Agent at state {self.state}, Steps: {self.steps}")
        print(f"Goal: {self.goal_state}")

    def get_room_states(self, room_num):
        """Get all states in a specific room."""
        rooms = {1: self.room_1, 2: self.room_2, 3: self.room_3, 4: self.room_4}
        return rooms.get(room_num, set())

    def get_hallway_connections(self):
        """Get all hallway connections as a dictionary."""
        return {
            "1-2": self.hallway_1_2,
            "1-3": self.hallway_1_3,
            "2-4": self.hallway_2_4,
            "3-4": self.hallway_3_4
        }


def get_primitive_actions_as_options(env: NavigationEnv):
    """Convert primitive actions to options for navigation environment."""
    options = []
    for a in range(env.action_space.n):
        # Define the initiation set for the option
        initiation_set = {s for s in range(env.observation_space.n)}  # env.navigable_states.copy()

        # Define the policy for the option
        policy = {s: env.get_action_distribution(s, a) for s in initiation_set}

        # Define the termination condition for the option
        def termination(s):
            return True

        # Create the option
        options.append(
            Option(
                id=f"action_{a}",
                initiation_set=initiation_set,
                policy=policy,
                termination=termination
            )
        )
    return options


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--env", type=str, default="tworooms", help="Environment to use")

    args = parser.parse_args()

    # Example usage of the TwoRooms environment
    if args.env == "tworooms":

        # Create the TwoRooms environment
        env = TwoRooms(
            start_state=5,
            goal_state=6,
            max_steps=100,
        )

        # Run a simple episode
        start_state = env.reset()
        G = 0.
        done = False
        steps = 1
        while not done:
            print()
            print("Step: ", steps)
            print("State: ", env.state)
            print("G: ", G)
            env.render()

            action = int(input("Action: "))
            next_state, reward, done, _, _ = env.step(action)
            G += reward
            steps += 1

    # Example usage of the FourRooms environment
    elif args.env == "fourrooms":

        # Test the FourRooms environment
        env = FourRooms(
            start_state=0,
            goal_state=35,
            negative_states_config="default"
        )

        print("=== FourRooms Environment Test ===")
        print(f"Grid size: {env.rows}x{env.cols} = {env.num_states} states")
        print(f"Room 1: {sorted(env.room_1)}")
        print(f"Room 2: {sorted(env.room_2)}")
        print(f"Room 3: {sorted(env.room_3)}")
        print(f"Room 4: {sorted(env.room_4)}")
        print(f"Hallway states: {sorted(env.hallway_states)}")
        print(f"Hallway connections: {env.get_hallway_connections()}")
        print(f"Negative states: {sorted(env.negative_states)}")
        print(f"Total navigable states: {len(env.navigable_states)}")

        # Test episode
        state, info = env.reset()
        env.render()

        print("\n=== Running test episode ===")
        done = False
        step_count = 0
        while not done and step_count < 30:
            action = env.sample_action()
            next_state, reward, done, truncated, info = env.step(action)
            print(f"Step {step_count+1}: Action {action}, State {env.state}, Reward {reward}")
            step_count += 1

            if step_count % 10 == 0:  # Show environment every 10 steps
                env.render()

        env.render()
        print(f"Episode finished: Goal reached = {env.goal_reached()}")
