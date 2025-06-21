import itertools

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from core import Policy
from env import NavigationEnv, TwoRooms, FourRooms


# Helper functions
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
        if state in env.right_room:
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


def coord_to_state(row, col, dim):
    """
    Convert (x, y) coordinates to a state number.

    Parameters:
    row (int): The row index
    col (int): The column index
    dim (int): The dimension of the grid

    Returns:
    int: The state number
    """
    return row * dim + col


def plot_deterministic_policy_in_tworooms(
        env: NavigationEnv,
        policy: Policy,
        g,
        plot_policy=True,
        plot_start_and_goal=True,
        save=False,
        filename="tworooms_policy.pdf"):
    """
    Plot a deterministic policy in the given environment.

    Parameters:
    env (NavigationEnv): The environment to plot the policy in.
    policy (Policy): The deterministic policy to plot.
    """
    # Define grid dimensions
    rows, cols = env.rows + 2, env.cols + 3
    vertical_strip = env.room_size + 1

    hallway_pos = (env.hallway_height + 1, vertical_strip)
    goal_pos = state_to_coords(env, env.goal_state, env.cols)  # (6, 10)

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

    grid[hallway_pos] = 1  # Hallway
    grid[goal_pos] = 1  # Goal position

    # Draw the grid cells
    for i in range(rows):
        for j in range(cols):

            color = 'white'  # wall

            if i == rows - 1 or j == cols - 1:
                pass
            else:
                _i = i - 1
                _j = j - 1 if j < vertical_strip else j - 2  # Adjust for the wall column
                s = coord_to_state(_i, _j, env.cols)

                # walls
                if i == 0 or i == rows - 1 or j == 0 or j == cols - 1 or (j == vertical_strip and i != hallway_pos[0]):
                    color = 'white'  # wall
                elif s == env.goal_state or s == g and plot_start_and_goal:
                    color = 'red'
                elif s == env._init_state and plot_start_and_goal:
                    color = 'green'
                else:
                    color = 'black'  # default state color

            rect = Rectangle(
                (j, rows-i-1), 1, 1,
                facecolor=color,
                edgecolor='gray',
                linewidth=1)
            ax.add_patch(rect)

            # Add special state labels
            if grid[i, j] == 4:
                ax.text(j+0.5, rows-i-1+0.5, 'S', ha='center', va='center', color='white', fontsize=12)
            elif grid[i, j] == 5:
                ax.text(j+0.5, rows-i-1+0.5, 'h', ha='center', va='center', color='white', fontsize=12)
            elif grid[i, j] == 6:
                ax.text(j+0.5, rows-i-1+0.5, 'G', ha='center', va='center', color='white', fontsize=12)

            # Draw action arrows if the cell is a state and has a policy
            if plot_policy and color == 'black' and s in policy:
                a = np.argmax(policy[s])

                # Draw action arrows
                if a == 0:
                    ax.arrow(j+0.5, rows-i-1+0.5, 0, 0.3, head_width=0.1, head_length=0.1, fc='white', ec='white')
                elif a == 1:
                    ax.arrow(j+0.5, rows-i-1+0.5, 0, -0.3, head_width=0.1, head_length=0.1, fc='white', ec='white')
                elif a == 2:
                    ax.arrow(j+0.5, rows-i-1+0.5, -0.3, 0, head_width=0.1, head_length=0.1, fc='white', ec='white')
                elif a == 3:
                    ax.arrow(j+0.5, rows-i-1+0.5, 0.3, 0, head_width=0.1, head_length=0.1, fc='white', ec='white')
                else:
                    raise ValueError("Invalid action value")

    # Set the plot limits
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Set title
    # ax.set_title('Grid World Environment with Policy', fontsize=14)

    plt.tight_layout()
    plt.show()
    plt.rc('legend', fontsize=22)

    if save:
        fig.savefig(f"../imgs/{filename}", format="pdf")
        print(f"Policy plot saved as {filename}")

    return fig, ax


def plot_heatmap_from_state_scores(env, state_scores: dict, save: bool, filename: str, show_scores=False):
    """
    Plot a heatmap of state scores (different metrics) for a given environment.
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get the size of the environment
    size = env.size
    wall_col = size[0] // 2
    rows, cols = size

    if isinstance(env, TwoRooms):
        dim = size[1]
        cols += 1
    elif isinstance(env, FourRooms):
        dim = size[1]
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
            if state == env.h:
                row, col = env.hallway_height, env.room_size
            else:
                row, col = state_to_coords(env, state, dim)
        elif isinstance(env, FourRooms) and state == 99:
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

    plt.figure(figsize=(12, 8))
    sns.heatmap(data, annot=show_scores, fmt=".2f", linewidth=.5, cmap="crest", ax=ax)  # .set_title(title)
    if save:
        plt.savefig(f"../imgs/{filename}", format="pdf")

    return fig, ax
