from dataclasses import dataclass
from typing import Optional

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Rectangle
import matplotlib.colors as mcolors

from core import Policy
from env import TwoRooms


@dataclass
class State:
    num: int
    row: int
    col: int
    env: TwoRooms
    subgoal: int
    type: str = None
    color: str = None
    action: int = None

    def __post_init__(self):
        if self.num in self.env.negative_states:
            self.type = 'negative'
        elif self.num == self.env.start_state:
            self.type = 'start'
        elif self.num == -1:
            self.type = 'goal'
        elif self.subgoal and self.num == self.subgoal.feature_attainment:  # self.env.goal_state:
            self.type = 'hallway'
        elif self.subgoal is None and self.num == self.env.goal_state:
            self.type = 'hallway'
        else:
            self.type = 'state'

        if self.type == 'state':
            self.color = 'black'
        elif self.type == 'negative':
            self.color = 'lightgray'
        elif self.type == 'start':
            self.color = 'green'
        elif self.type == 'goal':
            self.color = 'red'
        elif self.type == 'hallway':
            self.color = 'red'
        else:
            raise ValueError(f"Unknown state type: {self.type}")


def states(env, subgoal):
    return [
        State(env.goal_state, env.hallway_height+1, 7, env, subgoal),
        State(0,  1,  1, env, subgoal),
        State(1,  1,  2, env, subgoal),
        State(2,  1,  3, env, subgoal),
        State(3,  1,  4, env, subgoal),
        State(4,  1,  5, env, subgoal),
        State(5,  1,  6, env, subgoal),
        State(6,  1,  8, env, subgoal),
        State(7,  1,  9, env, subgoal),
        State(8,  1, 10, env, subgoal),
        State(9,  1, 11, env, subgoal),
        State(10, 1, 12, env, subgoal),
        State(11, 1, 13, env, subgoal),

        State(12, 2,  1, env, subgoal),
        State(13, 2,  2, env, subgoal),
        State(14, 2,  3, env, subgoal),
        State(15, 2,  4, env, subgoal),
        State(16, 2,  5, env, subgoal),
        State(17, 2,  6, env, subgoal),
        State(18, 2,  8, env, subgoal),
        State(19, 2,  9, env, subgoal),
        State(20, 2, 10, env, subgoal),
        State(21, 2, 11, env, subgoal),
        State(22, 2, 12, env, subgoal),
        State(23, 2, 13, env, subgoal),

        State(24, 3,  1, env, subgoal),
        State(25, 3,  2, env, subgoal),
        State(26, 3,  3, env, subgoal),
        State(27, 3,  4, env, subgoal),
        State(28, 3,  5, env, subgoal),
        State(29, 3,  6, env, subgoal),
        State(30, 3,  8, env, subgoal),
        State(31, 3,  9, env, subgoal),
        State(32, 3, 10, env, subgoal),
        State(33, 3, 11, env, subgoal),
        State(34, 3, 12, env, subgoal),
        State(35, 3, 13, env, subgoal),

        State(36, 4,  1, env, subgoal),
        State(37, 4,  2, env, subgoal),
        State(38, 4,  3, env, subgoal),
        State(39, 4,  4, env, subgoal),
        State(40, 4,  5, env, subgoal),
        State(41, 4,  6, env, subgoal),
        State(42, 4,  8, env, subgoal),
        State(43, 4,  9, env, subgoal),
        State(44, 4, 10, env, subgoal),
        State(45, 4, 11, env, subgoal),
        State(46, 4, 12, env, subgoal),
        State(47, 4, 13, env, subgoal),

        State(48, 5,  1, env, subgoal),
        State(49, 5,  2, env, subgoal),
        State(50, 5,  3, env, subgoal),
        State(51, 5,  4, env, subgoal),
        State(52, 5,  5, env, subgoal),
        State(53, 5,  6, env, subgoal),
        State(54, 5,  8, env, subgoal),
        State(55, 5,  9, env, subgoal),
        State(56, 5, 10, env, subgoal),
        State(57, 5, 11, env, subgoal),
        State(58, 5, 12, env, subgoal),
        State(59, 5, 13, env, subgoal),

        State(60, 6,  1, env, subgoal),
        State(61, 6,  2, env, subgoal),
        State(62, 6,  3, env, subgoal),
        State(63, 6,  4, env, subgoal),
        State(64, 6,  5, env, subgoal),
        State(65, 6,  6, env, subgoal),
        State(66, 6,  8, env, subgoal),
        State(67, 6,  9, env, subgoal),
        State(69, 6, 11, env, subgoal),
        State(70, 6, 12, env, subgoal),
        State(71, 6, 13, env, subgoal),

        State(68, 6, 10, env, subgoal),
        # State(-1, 6, 10, env, ),
    ]


def get_state(row, col, env, subgoal):
    """Get the state object for a given row and column from the States series."""
    for state in states(env, subgoal):
        if state.row == row and state.col == col:
            return state
    raise ValueError(f"No state found at row {row}, col {col}")


def plot_deterministic_policy(env: TwoRooms, policy: Policy, subgoal=None):
    # Define grid dimensions
    rows, cols = 8, 15
    vertical_strip = 7
    hallway_pos = (3, 7)
    goal_pos = (6, 10)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors for different cell types
    colors = {
        'wall': 'white',
        'state': 'black',
        'negative_reward': 'gray',
        'positive_reward': 'red',
        'special_state_S': 'green',
        'special_state_h': 'brown',
        'special_state_G': 'red'
    }

    # Create the grid layout
    grid = np.zeros((rows, cols), dtype=int)

    # Mark walls (1) - the outer border and the middle vertical strip
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 8  # Outer walls
    grid[:, vertical_strip] = 8  # Middle vertical wall

    grid[1:6, 2:6] = -1  # Negative area

    grid[hallway_pos] = 1  # Hallway
    grid[goal_pos] = 1  # Goal position

    # Draw the grid cells
    for i in range(rows):
        for j in range(cols):

            state: Optional[State] = None
            try:
                state = get_state(i, j, env, subgoal)
                state.type
                color = state.color
            except ValueError:
                color = 'white'  # wall

            rect = Rectangle((j, rows-i-1), 1, 1,
                             facecolor=color,
                             edgecolor='gray',
                             linewidth=1)
            ax.add_patch(rect)

            # Add special state labels
            if grid[i, j] == 4:
                ax.text(j+0.5, rows-i-1+0.5, 'S',
                        ha='center', va='center', color='white', fontsize=12)
            elif grid[i, j] == 5:
                ax.text(j+0.5, rows-i-1+0.5, 'h',
                        ha='center', va='center', color='white', fontsize=12)
            elif grid[i, j] == 6:
                ax.text(j+0.5, rows-i-1+0.5, 'G',
                        ha='center', va='center', color='white', fontsize=12)

            if state:
                state.action = np.argmax(policy[state.num])

                # Draw action arrows
                if state.action == 0:
                    ax.arrow(j+0.5, rows-i-1+0.5, 0, 0.3,
                             head_width=0.1, head_length=0.1, fc='white', ec='white')
                elif state.action == 1:
                    ax.arrow(j+0.5, rows-i-1+0.5, 0, -0.3,
                             head_width=0.1, head_length=0.1, fc='white', ec='white')
                elif state.action == 2:
                    ax.arrow(j+0.5, rows-i-1+0.5, -0.3, 0,
                             head_width=0.1, head_length=0.1, fc='white', ec='white')
                elif state.action == 3:
                    ax.arrow(j+0.5, rows-i-1+0.5, 0.3, 0,
                             head_width=0.1, head_length=0.1, fc='white', ec='white')
                else:
                    raise ValueError("Invalid action value")

    # Set the plot limits
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Set title
    ax.set_title('Grid World Environment with Policy', fontsize=14)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # Create env
    env = TwoRooms(start_state=24, goal_state=20, negative_states_config="none")

    # Create sample policy
    policy: Policy = {s: np.random.randint(0, 4) for s in range(env.observation_space.n)}

    # Plot the policy
    plot_deterministic_policy(env, policy)


from env import TwoRooms, FourRooms

def state_to_coords(env, state, dim):
    """
    Convert a state number to (x, y) coordinates.

    Parameters:
    state_num (int): The state number
    has_wall (bool): Whether there's a wall

    Returns:
    tuple: (row, col) coordinates
    """
    # Convert state number to coordinates
    row = state // dim
    col = state % dim

    if isinstance(env, TwoRooms):
        if state in [
            6,  7,  8,  9, 10, 11,
            18, 19, 20, 21, 22, 23,
            30, 31, 32, 33, 34, 35,
            42, 43, 44, 45, 46, 47,
            54, 55, 56, 57, 58, 59,
            66, 67,  68, 69, 70, 71
        ]:
            col += 1
    elif isinstance(env, FourRooms):
        if state in [
            5,  6,  7,  8,  9,
            15, 16, 17, 18, 19,
            25, 26, 27, 28, 29,
            35, 36, 37, 38, 39,
            45, 46, 47, 48, 49,
            55, 56, 57, 58, 59,
            65, 66, 67, 68, 69,
            75, 76, 77, 78, 79,
            85, 86, 87, 88, 89,
            95, 96, 97, 98, 99
            ]:
            col += 1
        if state in [
            50, 51, 52, 53, 54,
            60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
            70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
            80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
            90, 91, 92, 93, 94, 95, 96, 97, 98, 99
        ]:
            row += 1

    return row, col


def plot_heatmap_from_state_scores(env, state_scores: dict):
    """
    Plot a heatmap of state scores (different metrics) for a given environment.
    """

    # Get the size of the environment
    size = env.size
    wall_col = size[0] // 2
    rows, cols = size

    if isinstance(env, TwoRooms):
        dim = 12
        cols += 1
    elif isinstance(env, FourRooms):
        dim = 10
        cols += 1
        rows += 1

    # Initialize the data grid
    data = np.zeros((rows, cols))

    # Walls
    data[:, wall_col] = -1.
    if isinstance(env, FourRooms):
        data[5, :5] = -1.
        data[6, 6:] = -1.

    # Hallways
    data[2, wall_col] = 0.0
    if isinstance(env, FourRooms):
        data[5, 1] = 0.0
        data[6, 8] = 0.0
        data[9, wall_col] = 0.0

    for state, score in state_scores.items():

        if isinstance(env, TwoRooms) and state == 68:
            row, col = 2, 6
        elif isinstance(env, FourRooms) and state == 75:
            row, col = 5, 1
        elif isinstance(env, FourRooms) and state == 100:
            row, col = 2, 5
        elif isinstance(env, FourRooms) and state == 101:
            row, col = 6, 8
        elif isinstance(env, FourRooms) and state == 102:
            row, col = 9, 5
        else:
            row, col = state_to_coords(env, state, dim)
        data[row, col] = score

    sns.heatmap(data)
