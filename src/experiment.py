from argparse import ArgumentParser
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from core import (
    Policy,
    Option,
    RewardRespectingFeatureAttainment,
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
parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
parser.add_argument("--id", type=str, default="default", help="ID for the experiment.")
parser.add_argument("--use_config_file", default=False, action="store_true", help="Use a configuration file for the experiment settings.")
parser.add_argument("--config_file", type=str, default="config.yaml", help="Path to the configuration file.")
parser.add_argument("--rep", type=int, default=1, help="Number of repetitions for the experiment.")
parser.add_argument("--num_episodes", type=int, default=50, help="Number of episodes to run in the experiment.")
parser.add_argument("--log", default=False, action="store_true", help="Log the results of the experiment.")
parser.add_argument("--save", default=False, action="store_true", help="Save the results of the experiment.")
parser.add_argument("--plot", default=False, action="store_true", help="Plot the results of the experiment.")

# Environment arguments
parser.add_argument("--env", type=str, default="TwoRooms", help="Environment to use for the experiment.")
parser.add_argument("--start_states", type=int, nargs='+', default=None, help="Starting state for the environment." )
parser.add_argument("--goal_states", type=int, nargs='+', default=None, help="Goal state for the environment.")
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


@dataclass
class ExperimentConfig:
    """ Configuration for the experiment. """
    id: str
    rep: int
    num_episodes: int
    env: str
    start_states: list[int]
    goal_states: list[int]
    negative_states_config: str
    max_steps: int
    sparse_rewards: bool
    stochastic: bool
    learning_rate: float
    discount_factor: float
    exploration_rate: float
    min_exploration_rate: float
    exploration_rate_decay: float
    discovery_method: str
    n: int
    k: int
    n_step: int
    save: bool = False
    plot: bool = False
    log: bool = False


# Experiment function to run the agent in the environment
def run_experiment(env: NavigationEnv, agent: AgentWithOptions, num_episodes, discovery_method, n, k, n_step, log=False):
    """ Run the experiment for a specified number of episodes. """

    # Reset
    env, agent = reset(env, agent)

    # Sample n episodes
    pre_trajectories, pre_results, pre_steps, pre_rewards, _ = sample(env, agent, n)

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
    subgoal = None
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
    trajectories, results, steps, rewards, _ = sample(env, agent, num_episodes - n)

    # Combine the pre-sampled trajectories with the new ones
    exp_trajectories = pre_trajectories + trajectories
    exp_results = pre_results + results
    exp_steps = pre_steps + steps
    exp_rewards = pre_rewards + rewards

    return exp_trajectories, exp_results, exp_steps, exp_rewards, subgoal


def get_env_and_agent(
        _env: str,
        start_states: int,
        goal_states: int,
        negative_states_config: str,
        max_steps: int,
        sparse_rewards: bool,
        stochastic: bool,
        learning_rate: float,
        discount_factor: float,
        exploration_rate: float,
        min_exploration_rate: float,
        exploration_rate_decay: float
    ):

    # Create the environment based on the specified type
    if _env == "TwoRooms":
        env = TwoRooms(
            size=(10,14),
            start_states=start_states,
            goal_states=goal_states,
            hallway_height=4,
            negative_states_config=negative_states_config,
            max_steps=max_steps,
            sparse_rewards=sparse_rewards,
            stochastic=stochastic
        )
    elif _env == "FourRooms":
        env = FourRooms(
            start_states=start_states,
            goal_states=goal_states,
            negative_states_config=negative_states_config,
            max_steps=max_steps,
            sparse_rewards=sparse_rewards
        )
    else:
        raise ValueError(f"Unknown environment: {_env}")

    # Get the primitive actions as options
    primitive_options = get_primitive_actions_as_options(env)

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

    return env, agent


if __name__ == "__main__":

    # Parse command line arguments
    args = parser.parse_args()

    # If using a config file, load the configuration
    configs = []
    if args.use_config_file:
        print(f"Using configuration file: {args.config_file}")
        import json
        with open(args.config_file, 'r') as f:
            configs = [ExperimentConfig(**cfg) for cfg in json.load(f)]
    else:
        del args.use_config_file
        del args.config_file
        configs = [ExperimentConfig(**args.__dict__)]

    seed = args.seed if hasattr(args, 'seed') else None
    if seed is not None:
        np.random.seed(seed)

    for cfg in configs:
        # Store results
        exp_trajectories = []
        exp_results = []
        exp_steps = []
        exp_rewards = []
        exp_subgoals = []

        print(f"\nStarting {cfg.id} with {cfg.rep} reps for  discovery method: {cfg.discovery_method}...")

        # Run the experiment with arguments
        for i in range(cfg.rep):
            print(f"Running experiment {i+1}/{cfg.rep}...")

            # Get the environment and agent
            env, agent = get_env_and_agent(
                cfg.env,
                cfg.start_states,
                cfg.goal_states,
                cfg.negative_states_config,
                cfg.max_steps,
                cfg.sparse_rewards,
                cfg.stochastic,
                cfg.learning_rate,
                cfg.discount_factor,
                cfg.exploration_rate,
                cfg.min_exploration_rate,
                cfg.exploration_rate_decay
            )

            try:
                trajectories, results, steps, rewards, discovered_subgoal = run_experiment(
                    env,
                    agent,
                    cfg.num_episodes,
                    cfg.discovery_method,
                    cfg.n,
                    cfg.k,
                    cfg.n_step,
                    log=cfg.log
                )

                del env
                del agent
            except Exception as e:
                print(f"Error during experiment run {i+1}: {e}. Retrying")
                i -= 1
                continue

            exp_trajectories.append(trajectories)
            exp_results.append(results)
            exp_steps.append(steps)
            exp_rewards.append(rewards)
            exp_subgoals.append(discovered_subgoal)

        # Save results to file
        if cfg.save:
            import json
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super(NpEncoder, self).default(obj)

            import os
            # Ensure the results directory exists
            os.makedirs("./src/results", exist_ok=True)
            os.makedirs(f"./src/results/{cfg.env}", exist_ok=True)
            os.makedirs(f"./src/results/{cfg.env}/{cfg.id}", exist_ok=True)
            # Save the results to a JSON file
            filename = f"./src/results/{cfg.env}/{cfg.id}/D{cfg.discovery_method}_Rp{cfg.rep}_n{cfg.n}_nstep{cfg.n_step}_N{cfg.negative_states_config}_SR{'t' if cfg.sparse_rewards else 'f'}_ST{'t' if cfg.stochastic else 'f'}.json"
            with open(filename, "w") as f:
                json.dump({
                    "trajectories": exp_trajectories,
                    "steps": exp_steps,
                    "subgoals": exp_subgoals,
                }, f, cls=NpEncoder)

        if cfg.plot:
            # Plot the results
            import matplotlib.pyplot as plt
            # Create the figure and primary axis
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Primary axis (left y-axis) for subgoal discovery (red)
            ax1.set_xlabel('Episodes', fontsize=12, color='black')
            ax1.set_ylabel('Steps', color='coral', fontsize=12)
            ax1.plot(range(cfg.num_episodes), steps, color='coral', linewidth=2)
            ax1.tick_params(axis='y', labelcolor='coral')
            ax1.set_xlim(0, cfg.num_episodes)
            ax1.grid(False)

            # Add title at the top (uncomment if needed)
            plt.title('Reinforcement Learning Training Progress', color='black', fontsize=14)

            # Style the plot
            for spine in ax1.spines.values():
                spine.set_color('black')

            ax1.tick_params(colors='black')

            # Style the plot
            for spine in ax1.spines.values():
                spine.set_color('black')

            ax1.tick_params(colors='black')
            ax1.xaxis.label.set_color('black')

            plt.tight_layout()
            plt.show()
