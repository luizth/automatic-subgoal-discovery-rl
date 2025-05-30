from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt

from core import (
    Policy,
    Option,
    Subtask,
    RewardRespectingFeatureAttainment,
)
from plot import (
    plot_deterministic_policy,
    plot_heatmap_from_state_scores
)
from agent import AgentWithOptions, SMDPQLearning
from env import (
    NavigationEnv,
    TwoRooms,
    FourRooms,
    get_primitive_actions_as_options
)
from trajectory_sampler import sample, reset
from subgoal_discovery import (
    diverse_density,
    relative_novelty
)
from empowerment import empowerment_subgoal_discovery
from option_learning import (
    actor_critic,
    to_deterministic_policy
)


parser = ArgumentParser(description="Run experiments for evaluation.")

# Experiment argumentents
parser.add_argument("--use_config_file", default=False, action="store_true", help="Use a configuration file for the experiment settings.")
parser.add_argument("--config_file", type=str, default="config.yaml", help="Path to the configuration file.")
parser.add_argument("--num_episodes", type=int, default=50, help="Number of episodes to run in the experiment.")
parser.add_argument("--log", default=False, action="store_true", help="Log the results of the experiment.")
# parser.add_argument("--log_file", type=str, default="./logs/experiment_log.txt", help="File to log the results of the experiment.")

# Environment arguments
parser.add_argument("--env", type=str, default="TwoRooms", help="Environment to use for the experiment.")
parser.add_argument("--start_state", type=int, default=None, help="Starting state for the environment." )
parser.add_argument("--goal_state", type=int, default=None, help="Goal state for the environment.")
parser.add_argument("--negative_states_config", type=str, default="none", help="Configuration for negative states in the environment.")
parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of steps per episode.")
parser.add_argument("--sparse_rewards", default=False, action="store_true", help="Use sparse rewards in the environment.")
parser.add_argument("--stochastic", default=False, action="store_true", help="Use stochastic transitions in the environment.")

# Agent arguments
parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate for the agent.")
parser.add_argument("--discount_factor", type=float, default=0.99, help="Discount factor for the agent.")
parser.add_argument("--exploration_rate", type=float, default=0.1, help="Initial exploration rate for the agent.")
parser.add_argument("--min_exploration_rate", type=float, default=0.1, help="Minimum exploration rate for the agent.")
parser.add_argument("--exploration_rate_decay", type=float, default=1.0, help="Decay rate for the exploration rate.")

# Subgoal discovery arguments
parser.add_argument("--discovery_method", type=str, default="none", help="Method for subgoal discovery (e.g., 'dd', 'rn', 'emp').")
parser.add_argument("--n", type=int, default=20, help="Episode window to learn a new option.")
parser.add_argument("--k", type=int, default=3, help="Select top k subgoals.")
parser.add_argument("--n_step", type=int, default=5, help="Number of steps for empowerment subgoal discovery.")


# Experiment function to run the agent in the environment
def run_experiment(env: NavigationEnv, agent: AgentWithOptions, num_episodes, discovery_method, n, k, n_step, log=False):
    """ Run the experiment for a specified number of episodes. """

    # Reset
    env, agent = reset(env, agent)

    # Sample n episodes
    pre_trajectories, pre_results, pre_steps = sample(env, agent, n)

    assert agent.eb.size > 0

    # Discover subgoals using the specified method
    if discovery_method == "none":
        subgoals = None
    elif discovery_method == "emp":
        subgoals = empowerment_subgoal_discovery(agent.eb, k=k, n_step=n_step)
    elif discovery_method == "dd":
        subgoals = diverse_density(env, pre_trajectories, k=k)
    elif discovery_method == "rn":
        subgoals = relative_novelty(env, pre_trajectories, k=k)
    else:
        raise ValueError(f"Unknown discovery method: {discovery_method}")

    # Create a subtask with the discovered subgoal
    if subgoals is not None:
        subgoal = subgoals[0]

        # Define the subtask to reach the subgoal
        task = RewardRespectingFeatureAttainment(env, feature_attainment=subgoal)

        # Learn the option that solves the subtask
        w, theta = actor_critic(task, alpha=0.1, gamma=0.99, alpha_=0.1, number_of_steps=50000)
        policy: Policy = to_deterministic_policy(env, theta)

        def termination_fn(state):
            """ Termination condition for the option """
            return state == subgoal or state == env.goal_transition_state

        option = Option(
            id=f"o{agent.options_size}_to_subgoal_{subgoal}",
            initiation_set=[s for s in range(env.observation_space.n)],  # Full observation space
            policy=policy,
            termination=termination_fn
        )

        # Add the option to the agent
        agent.add_option(option)

    # Run the agent for the remaining episodes
    trajectories, results, steps = sample(env, agent, num_episodes - n)

    return pre_trajectories + trajectories, pre_results + results, pre_steps + steps


if __name__ == "__main__":

    # Parse command line arguments
    args = parser.parse_args()

    use_config_file = args.use_config_file
    config_file = args.config_file
    num_episodes = args.num_episodes

    # If using a config file, load the configuration
    if use_config_file:
        import yaml
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
            args.__dict__.update(config)

    # Parse command line arguments for the environment
    _env = args.env
    start_state = args.start_state
    goal_state = args.goal_state
    negative_states_config = args.negative_states_config
    max_steps = args.max_steps
    sparse_rewards = args.sparse_rewards
    stochastic = args.stochastic

    # Create the environment based on the specified type
    if _env == "TwoRooms":
        env = TwoRooms(
            start_state=start_state,
            goal_state=goal_state,
            negative_states_config=negative_states_config,
            max_steps=max_steps,
            sparse_rewards=sparse_rewards,
            stochastic=stochastic
        )
    elif _env == "FourRooms":
        env = FourRooms(
            start_state=start_state,
            goal_state=goal_state,
            negative_states_config=negative_states_config,
            max_steps=max_steps,
            sparse_rewards=sparse_rewards
        )
    else:
        raise ValueError(f"Unknown environment: {_env}")

    # Get the primitive actions as options
    primitive_options = get_primitive_actions_as_options(env)

    # Parse command line arguments for the agent
    learning_rate = args.learning_rate
    discount_factor = args.discount_factor
    exploration_rate = args.exploration_rate
    min_exploration_rate = args.min_exploration_rate
    exploration_rate_decay = args.exploration_rate_decay

    # Create the agent
    agent = SMDPQLearning(
        env,
        primitive_options,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        exploration_rate=exploration_rate,
        min_exploration_rate=min_exploration_rate,
        exploration_decay=exploration_rate_decay,
        store_experience=True
    )

    # Parse command line arguments for subgoal discovery
    discovery_method = args.discovery_method
    n = args.n
    k = args.k
    n_step = args.n_step

    # Store results
    exp_trajectories = []
    exp_results = []
    exp_steps = []

    # Recreate agent
    agent = SMDPQLearning(
        env,
        primitive_options,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        exploration_rate=exploration_rate,
        min_exploration_rate=min_exploration_rate,
        exploration_decay=exploration_rate_decay,
        store_experience=True
    )

    # Run the experiment with arguments
    trajectories, results, steps = run_experiment(
        env,
        agent,
        num_episodes,
        discovery_method,
        n,
        k,
        n_step,
        log=args.log
    )
    exp_trajectories.append(trajectories)
    exp_results.append(results)
    exp_steps.append(steps)

    # Run the experiment for vanilla agent (without subgoal discovery)
    trajectories, results, steps = run_experiment(
        env,
        agent,
        num_episodes,
        "none",
        n,
        k,
        n_step,
        log=args.log
    )
    exp_trajectories.append(trajectories)
    exp_results.append(results)
    exp_steps.append(steps)


    # Plot the results
    import matplotlib.pyplot as plt
    # Create the figure and primary axis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # fig.set_facecolor('black')
    # ax1.set_facecolor('black')

    # Primary axis (left y-axis) for subgoal discovery (red)
    ax1.set_xlabel('Episodes', fontsize=12, color='black')
    ax1.set_ylabel('Steps', color='coral', fontsize=12)
    ax1.plot(range(num_episodes), exp_steps[0], color='coral', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='coral')
    ax1.set_xlim(0, num_episodes)
    # ax1.set_ylim(0, 2.2)
    ax1.grid(False)

    # Primary axis (left y-axis) for vanilla (blue)
    ax1.set_xlabel('Episodes', fontsize=12, color='black')
    ax1.set_ylabel('Steps', color='skyblue', fontsize=12)
    ax1.plot(range(num_episodes), exp_steps[1], color='skyblue', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax1.set_xlim(0, num_episodes)
    # ax1.set_ylim(0, 2.2)
    ax1.grid(False)

    # Add title at the top (uncomment if needed)
    # plt.title('Reinforcement Learning Training Progress', color='black', fontsize=14)

    # Style the plot
    for spine in ax1.spines.values():
        spine.set_color('black')

    ax1.tick_params(colors='black')
    ax1.xaxis.label.set_color('black')

    plt.tight_layout()
    plt.show()
