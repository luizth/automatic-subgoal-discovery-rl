from tqdm import tqdm
import gymnasium as gym


def sample(env: gym.Env, agent, number_of_samples: int = 50):

    # Reset
    agent.env = env
    agent.reset()

    # Loop
    trajectories = []
    results = []
    steps = []
    for t in tqdm(range(number_of_samples)):
        initial_state, total_reward, step, goal_reached, trajectory = agent.run_episode()
        trajectories.append(trajectory)
        results.append(goal_reached)
        steps.append(step)

    return trajectories, results, steps


if __name__ == "__main__":
    from typing import List
    from core import Option
    from agent import SMDPQLearning
    from env import TwoRooms, get_primitive_actions_as_options

    # Create the TwoRooms environment
    env = TwoRooms(
        start_state=24,
        goal_state=68,
        negative_states_config="default",
        max_steps=None,
        sparse_rewards=False
    )

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

    trajectories, results, steps = sample(env, agent, 1)
    print(f"Trajectory start: {trajectories[0][:3]} end: {trajectories[0][-3:]}  result: {results[0]} steps {steps[0]}")
