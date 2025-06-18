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


class OpenRoom(NavigationEnv):

    def __init__(self,
        size=(10, 10),
        start_states=[0],
        goal_states=[99],
        negative_states_config="none",
        max_steps=None,
        sparse_rewards=True,
        stochastic=False,
        ):

        """
        OpenRoom environment with a single room.

        The grid is represented as follows:

        0  1  2  3  4  5  6  7  8  9
        10 11 12 13 14 15 16 17 18 19
        ...
        90 91 92 93 94 95 96 97 98 99

        S = start state, G = goal state

        """
        # Store the environment parameters
        self.size = size
        self.rows = size[0]
        self.cols = size[1]

        if not start_states or not isinstance(start_states, list):
            raise ValueError("start_states must be a list of integers, but is", type(start_states))
        if not goal_states or not isinstance(goal_states, list):
            raise ValueError("goal_states must be an integer or a list of integers, but is", type(goal_states))

        # Non-terminal state
        num_states = self.rows * self.cols

        # Env configuration
        self.observation_space = gym.spaces.Discrete(num_states)
        self.action_space = gym.spaces.Discrete(4)

        self.start_states = start_states
        self.goal_states = goal_states
        self.max_steps = max_steps
        self.sparse_rewards = sparse_rewards
        self.stochastic = stochastic
        self.init_state()
        self._init_state = self.state

        # State transition dynamics
        self.goal_transition_state = -1  # Reference to goal state

        # Configure negative areas
        self.negative_states_config = negative_states_config
        self.negative_states = self._configure_negative_states(negative_states_config)

        # Define room boundaries
        self._define_boundaries()

        # State transition dynamics
        self.transitions = self._build_transitions()

        # Agent monitoring
        self.steps = 0
        self.trajectory = []

    def _configure_negative_states(self, config):
        """Configure negative reward states based on configuration."""
        if config == "none":
            return set()
        else:
            raise ValueError("Invalid negative states configuration")

    def _define_boundaries(self):
        """Define the boundaries of the room."""
        self.upper_wall = [j for j in range(self.cols)]
        self.lower_wall = [self.cols * (self.rows-1) + j for j in range(self.cols)]
        self.walls_to_left = [self.cols * i for i in range(self.rows)]
        self.walls_to_right = [self.cols * i - 1 for i in range(1, self.rows+1)]

    def _build_transitions(self):
        """Build the state transition matrix."""
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
        if above_goal in transitions and above_goal not in self.upper_wall:
            transitions[above_goal, 1] = -1
        if below_goal in transitions and above_goal not in self.lower_wall:
            transitions[below_goal, 0] = -1
        if left_goal in transitions and left_goal not in self.walls_to_right:
            transitions[left_goal, 3] = -1
        if right_goal in transitions and right_goal not in self.walls_to_left:
            transitions[right_goal, 2] = -1

        return transitions

    def init_state(self):
        """Initialize the environment to a random state"""
        self.state = np.random.choice(self.start_states)
        self.goal_state = np.random.choice(self.goal_states)
        return self.state

    def get_reward(self, next_state):
        if next_state == self.goal_transition_state:
            return 10.0
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

    def get_action_distribution(self, state, action):
        """Get the action distribution for a given state and action"""
        if state == self.goal_transition_state:
            return np.zeros(self.action_space.n)
        else:
            action_distribution = np.zeros(self.action_space.n)
            action_distribution[action] = 1.0
            return action_distribution

    def goal_reached(self):
        """Check if the goal state is reached"""
        return self.state == self.goal_transition_state

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


class TwoRooms(NavigationEnv):

    def __init__(self,
        size=(6, 12),
        start_states=[24],
        goal_states=[68],
        hallway_height=2,
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

        Other example

        0   1   2   3   4   5   6   -   7   8   9   10  11  12  13
        14  15  16  17  18  19  20  -   21  22  23  24  25  26  27
        28  29  30  31  32  33  34  -   35  36  37  38  39  40  41
        42  43  44  45  46  47  48  -   49  50  51  52  53  54  55
        56  57  58  59  60  61  62  -   63  64  65  66  67  68  69
        70  71  72  73  74  75  76  -   77  78  79  80  81  82  83
        84  85  86  87  88  89  90  -   91  92  93  94  95  96  97
        98  99 100 101 102 103 104  -   105 106 107 108 109 110 111
        112 113 114 115 116 117 118 -   119 120 121 122 123 124 125
        126 127 128 129 130 131 132 -   133 134 135 136 137 138 139
        """
        # Store the environment parameters
        if size[1] % 2 != 0:
            raise ValueError("Width of the grid must be even")
        self.size = size
        self.rows = size[0]
        self.cols = size[1]
        self.room_size = self.cols // 2

        # Validate start and goal states
        if not start_states or not isinstance(start_states, list):
            raise ValueError("start_states must be a list of integers, but is", type(start_states))
        if not goal_states or not isinstance(goal_states, list):
            raise ValueError("goal_states must be an integer or a list of integers, but is", type(goal_states))

        # Non-terminal state (hallway assumes goal state)
        num_states = self.rows * self.cols

        # Env configuration
        self.observation_space = gym.spaces.Discrete(num_states)
        self.action_space = gym.spaces.Discrete(4)  # Up, Down, Left, Right

        self.start_states = start_states
        self.goal_states = goal_states
        self.max_steps = max_steps
        self.sparse_rewards = sparse_rewards
        self.stochastic = stochastic
        self.init_state()
        self._init_state = self.state

        # State transition dynamics
        self.goal_transition_state = -1  # Reference to goal state

        # Goal becomes a hallway
        self.h = self.goal_state
        self.hallway_height = hallway_height

        # Configure negative areas
        self.negative_states_config = negative_states_config
        self.negative_states = self._configure_negative_states(negative_states_config)

        # Define room boundaries
        self._define_boundaries()

        # State transition dynamics
        self.transitions = self._build_transitions()

        # Agent monitoring
        self.steps = 0
        self.trajectory = []

    @property
    def right_room(self):
        """Get the right room states."""
        room = []
        for i in range(self.rows):
            for j in range(self.room_size, self.cols):
                room.append(j + self.cols * i)
        if self.h in room:
            room.remove(self.h)
        return room

    def copy(self):
        """Create a copy of the environment"""
        new_env = TwoRooms(
            size=self.size,
            start_states=self.start_states,
            goal_states=self.goal_states,
            hallway_height=self.hallway_height,
            negative_states_config=self.negative_states_config,
            max_steps=self.max_steps,
            sparse_rewards=self.sparse_rewards,
            stochastic=self.stochastic
        )
        return new_env

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
        self.walls_to_left = [self.cols * i for i in range(self.rows)] + [self.room_size + self.cols * i for i in range(self.rows)]
        self.walls_to_right = [self.cols * i - 1 for i in range(1, self.rows+1)] + [self.room_size + self.cols * i - 1 for i in range(self.rows)]

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
        if above_goal in transitions and above_goal not in self.upper_wall:
            transitions[above_goal, 1] = -1
        if below_goal in transitions and below_goal not in self.lower_wall:
            transitions[below_goal, 0] = -1
        if left_goal in transitions and left_goal not in self.walls_to_right:
            transitions[left_goal, 3] = -1
        if right_goal in transitions and right_goal not in self.walls_to_left:
            transitions[right_goal, 2] = -1

        # Hallway transition dynamics
        left_of_hallway = self.room_size + (self.hallway_height * self.cols) - 1
        right_of_hallway = self.room_size + (self.hallway_height * self.cols)
        transitions[self.h, 0] = self.h
        transitions[self.h, 1] = self.h
        transitions[self.h, 2] = left_of_hallway
        transitions[self.h, 3] = right_of_hallway
        # adjacent to hallway
        transitions[left_of_hallway, 3] = self.h
        transitions[right_of_hallway, 2] = self.h

        return transitions

    def step(self, action):
        """Take a step in the environment"""
        if self.goal_reached():
            raise ValueError("Episode ended")

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
        for s in range(self.observation_space.n):
            to_print = '.'
            if s in self.negative_states:
                to_print = 'N'
            if s in self.trajectory:
                to_print = '*'
            if s == self._init_state:
                to_print = 'S'
            if s == self.goal_state:
                to_print = 'G'
            if s == self.state:
                to_print = 'X'

            print(to_print, end=' ')

            if (s+1) % self.cols == 0:
                print()

        if self.state == self.h:
            print("Hallway: ", self.state)

    @property
    def states(self):
        """Get the states of the environment"""
        return np.arange(self.num_states)

    def goal_reached(self):
        """Check if the goal state is reached"""
        return self.state == self.goal_transition_state

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
            return 10.0
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

    def init_state(self):
        """Initialize the environment to a random state"""
        self.state = np.random.choice(self.start_states)
        self.goal_state = np.random.choice(self.goal_states)
        return self.state


class FourRooms(NavigationEnv):
    """
    Four rooms environment with hallways connecting adjacent rooms.

    The grid is represented as follows:

        Room1               H2  Room2
        0   1   2   3   4   -   5   6   7   8   9
        10  11  12  13  14  -   15  16  17  18  19
        20  21  22  23  24  -   25  26  27  28  29
        30  31  32  33  34  -   35  36  37  38  39
        40  41  42  43  44  -   45  46  47  48  49
    H1  -   -   -   -    -  -   50  51  52  53  54
        55  56  57  58  59  -    -   -   -   -   -  H3
        60  61  62  63  64  -   65  66  67  68  69
        70  71  72  73  74  -   75  76  77  78  79
        80  81  82  83  84  -   85  86  87  88  89
        90  91  92  93  94  -   95  96  97  98  99
        Room3               H4  Room4

    when G = 75,
     and hallways = [1, 2, 2. 2],
        because (
            from left to right in horizontal stripes,
            and top to bottom in vertical stripes),
        the grid will look like this
    then

        Room1               H2  Room2
        0   1   2   3   4   -   5   6   7   8   9
        10  11  12  13  14  -   15  16  17  18  19
        20  21  22  23  24  100 25  26  27  28  29
        30  31  32  33  34  -   35  36  37  38  39
        40  41  42  43  44  -   45  46  47  48  49
    H1  -   75  -   -    -  -   50  51  52  53  54
        55  56  57  58  59  -    -   -  101 -   -   H3
        60  61  62  63  64  -   65  66  67  68  69
        70  71  72  73  74  -   75  76  77  78  79
        80  81  82  83  84  102 85  86  87  88  89
        90  91  92  93  94  -   95  96  97  98  99
        Room3               H4  Room4
    """

    def __init__(self,
            size=(10, 10),
            start_states=[20],
            goal_states=[75],
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
        self.rooms_size = [
            (self.rows // 2, self.cols // 2, ),
            (self.rows // 2 + 1, self.cols // 2, ),
            (self.rows // 2, self.cols // 2, ),
            (self.rows // 2 - 1, self.cols // 2, )
        ]
        self.hallways_pos = [1, 2, 2, 2]  # Hallways in each stripe

        if not start_states or not isinstance(start_states, list):
            raise ValueError("start_states must be a list of integers, but is", type(start_states))
        if not goal_states or not isinstance(goal_states, list):
            raise ValueError("goal_states must be an integer or a list of integers, but is", type(goal_states))

        # Non-terminal state + 3 hallways (one of them will be the goal)
        num_states = self.rows * self.cols + 3

        # Env configuration
        self.observation_space = gym.spaces.Discrete(num_states)
        self.action_space = gym.spaces.Discrete(4)  # Up, Down, Left, Right

        self.start_states = start_states
        self.goal_states = goal_states
        self.max_steps = max_steps
        self.sparse_rewards = sparse_rewards
        self.init_state()
        self._init_state = self.state

        # State transition dynamics
        self.goal_transition_state = -1  # Reference to goal state

        # Goal becomes a hallway
        self.h1 = self.goal_state
        self.h2 = num_states - 3  # Hallway 2
        self.h3 = num_states - 2  # Hallway 3
        self.h4 = num_states - 1  # Hallway 4

        # Configure negative areas
        self.negative_states_config = negative_states_config
        self.negative_states = self._configure_negative_states(negative_states_config)

        # Define room boundaries
        self._define_boundaries()

        # State transition dynamics
        self.transitions = self._build_transitions()

        # Agent monitoring
        self.steps = 0
        self.trajectory = []

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

    def copy(self):
        """Create a copy of the environment"""
        new_env = FourRooms(
            size=self.size,
            start_states=self.start_states,
            goal_states=self.goal_states,
            negative_states_config=self.negative_states_config,
            max_steps=self.max_steps,
            sparse_rewards=self.sparse_rewards
        )
        return new_env

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
        self.state = np.random.choice(self.start_states)
        self.goal_state = np.random.choice(self.goal_states)
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
