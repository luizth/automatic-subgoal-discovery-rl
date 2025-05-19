from dataclasses import dataclass
from typing import Optional

import numpy as np
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
