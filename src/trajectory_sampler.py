from tqdm import tqdm
import gymnasium as gym

from agent import AgentWithOptions


def reset(env: gym.Env, agent):
    """Reset the environment and agent state"""
    newenv = env.copy()
    newagent = agent.copy()
    newagent.env = newenv
    del env
    del agent
    return newenv, newagent


def sample(env: gym.Env, agent: AgentWithOptions, number_of_samples: int = 50):
    """ Sample trajectories from the environment using the agent """

    # Set agent environment
    agent.env = env

    # Loop
    trajectories = []
    results = []
    steps = []
    total_rewards = []
    for t in tqdm(range(number_of_samples)):
        initial_state, total_reward, step, goal_reached, trajectory = agent.run_episode()
        trajectories.append(trajectory)
        results.append(goal_reached)
        steps.append(step)
        total_rewards.append(total_reward)

    return trajectories, results, steps, total_rewards


if __name__ == "__main__":
    from argparse import ArgumentParser
    from typing import List
    from core import Option
    from agent import SMDPQLearning
    from env import TwoRooms, FourRooms, get_primitive_actions_as_options

    # Parse arguments
    parser = ArgumentParser(description="Sample trajectories application.")
    parser.add_argument("--env", type=str, default="TwoRooms", help="Environment to use.")
    parser.add_argument("--number_of_samples", type=int, default=50, help="Number of samples to collect.")

    args = parser.parse_args()

    # Set the environment based on the argument
    if args.env == "TwoRooms":
        env = TwoRooms(
            start_state=24,
            goal_state=68,
            negative_states_config="default",
            max_steps=None,
            sparse_rewards=False
        )
    elif args.env == "FourRooms":
        env = FourRooms(
            start_state=6,
            goal_state=34,
            negative_states_config="default",
            max_steps=None,
            sparse_rewards=False
        )
    else:
        raise ValueError(f"Unknown environment: {args.env}")

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
        exploration_decay=0.99
    )

    trajectories, results, steps, _ = sample(env, agent, 1)
    print(f"Trajectory start: {trajectories[0][:3]} end: {trajectories[0][-3:]}  result: {results[0]} steps {steps[0]}")
