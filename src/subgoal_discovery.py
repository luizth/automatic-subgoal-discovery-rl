from typing import List, Dict, Tuple, Set
import numpy as np
from collections import defaultdict

from core import Trajectory, Subgoal


def diverse_density(
    trajectories: List[Trajectory],
    results: List[bool],
    static_filter: Set[int] = None,
    threshold: float = 0.0001
) -> List[Tuple[Subgoal, float]]:
    """
    Discover subgoals using diverse density.

    Args:
        trajectories: List of trajectories (state sequences)
        results: List of booleans indicating if each trajectory was successful
        static_filter: Set of states to exclude from consideration as subgoals
        threshold: Minimum diverse density value to consider a state as a subgoal

    Returns:
        List of (state, density) tuples sorted by density in descending order
    """
    if static_filter is None:
        static_filter = set()

    # Create positive and negative bags
    positive_bags = [trajectory for trajectory, result in zip(trajectories, results) if result]
    negative_bags = [trajectory for trajectory, result in zip(trajectories, results) if not result]

    # If no positive bags, return empty list
    if not positive_bags:
        return []

    # Get all unique states
    all_states = set()
    for trajectory in trajectories:
        all_states.update(trajectory)
    all_states = all_states - static_filter  # Remove filtered states

    # Calculate diverse density for each state
    dd_values = {}
    for state in all_states:
        if state == -1:  # Skip terminal state
            continue
        dd_values[state] = calculate_dd(state, positive_bags, negative_bags)

    # Sort states by diverse density
    if threshold > 0:
        sorted_dd = sorted([(state, dd) for state, dd in dd_values.items() if dd > threshold],
                           key=lambda x: x[1], reverse=True)
    else:
        sorted_dd = sorted(dd_values.items(), key=lambda x: x[1], reverse=True)

    return sorted_dd


def calculate_dd(state: Subgoal, positive_bags: List[Trajectory], negative_bags: List[Trajectory]) -> float:
    """
    Calculate the diverse density value for a given state using the noisy-or model
    as described in the McGovern paper.

    Args:
        state: The state to calculate diverse density for
        positive_bags: List of successful trajectories
        negative_bags: List of unsuccessful trajectories

    Returns:
        The diverse density value
    """
    # Convert trajectories to first-visit trajectories as mentioned in the paper
    first_visit_positive = []
    for bag in positive_bags:
        first_visit = []
        seen = set()
        for s in bag:
            if s not in seen:
                first_visit.append(s)
                seen.add(s)
        first_visit_positive.append(first_visit)

    first_visit_negative = []
    for bag in negative_bags:
        first_visit = []
        seen = set()
        for s in bag:
            if s not in seen:
                first_visit.append(s)
                seen.add(s)
        first_visit_negative.append(first_visit)

    # Calculate probability for each positive bag
    pos_prob = 1.0
    for bag in first_visit_positive:
        # Using the noisy-or model from the paper
        bag_prob = 0.0
        for s in bag:
            if s == state:
                # In our binary case (state is either in concept or not),
                # the instance probability is 1.0 if the state matches
                instance_prob = 1.0
                bag_prob = 1.0 - (1.0 - bag_prob) * (1.0 - instance_prob)
        pos_prob *= bag_prob

    # Calculate probability for each negative bag
    neg_prob = 1.0
    for bag in first_visit_negative:
        # For negative bags, no instance should be in the concept
        bag_prob = 1.0
        for s in bag:
            if s == state:
                # If state appears in negative bag, reduce probability
                instance_prob = 1.0
                bag_prob *= (1.0 - instance_prob)
        neg_prob *= bag_prob

    # Diverse density is the product of the two probabilities
    return pos_prob * neg_prob


def running_average_dd(
    trajectories: List[Trajectory],
    results: List[bool],
    lambda_value: float = 0.9,
    static_filter: Set[int] = None,
    threshold: float = 0.5
) -> Dict[Subgoal, float]:
    """
    Maintain a running average of diverse density values to identify persistent peaks.

    Args:
        trajectories: List of trajectories
        results: List of results (True for success, False for failure)
        lambda_value: Decay parameter for running average
        static_filter: Set of states to exclude from consideration
        threshold: Minimum running average value to consider a state as a subgoal

    Returns:
        Dictionary mapping states to their running average diverse density values
    """
    if static_filter is None:
        static_filter = set()

    # Initialize running averages
    running_avg = defaultdict(float)

    # Process trajectories one by one
    for i in range(len(trajectories)):
        # For each trajectory, calculate the DD values
        current_trajectories = trajectories[:i+1]
        current_results = results[:i+1]

        dd_values = diverse_density(current_trajectories, current_results, static_filter)

        # Update running averages for states found
        for state, _ in dd_values:
            running_avg[state] = lambda_value * (running_avg[state] + 1)

    # Filter by threshold
    filtered_avg = {state: avg for state, avg in running_avg.items() if avg > threshold}

    return filtered_avg


def create_option_for_subgoal_dd(subgoal: Subgoal, trajectories: List[Trajectory], n: int = 5):
    """
    Create an option with an input set for a given subgoal by examining trajectories.

    Args:
        subgoal: The subgoal state
        trajectories: List of trajectories
        n: Number of steps before subgoal to include in the input set

    Returns:
        Dictionary with input_set for the option
    """
    input_set = set()

    # Examine each trajectory for occurrences of the subgoal
    for trajectory in trajectories:
        if subgoal in trajectory:
            # Find the index of the subgoal in the trajectory
            subgoal_idx = trajectory.index(subgoal)

            # Add states from n steps before the subgoal to the input set
            start_idx = max(0, subgoal_idx - n)
            for i in range(start_idx, subgoal_idx):
                input_set.add(trajectory[i])

    return {"input_set": input_set}


def calculate_novelty(visit_counts, state):
    """
    Calculate novelty of a state as 1/sqrt(visit_count).

    Args:
        visit_counts: Dictionary mapping states to visit counts
        state: The state to calculate novelty for

    Returns:
        Novelty value for the state
    """
    count = visit_counts.get(state, 0)
    if count == 0:
        return 1.0  # Maximum novelty for unvisited states
    return 1.0 / np.sqrt(count)

def calculate_set_novelty(visit_counts, states):
    """
    Calculate novelty of a set of states as 1/sqrt(average_count).

    Args:
        visit_counts: Dictionary mapping states to visit counts
        states: Set of states to calculate novelty for

    Returns:
        Novelty value for the set of states
    """
    if not states:
        return 0.0

    total_count = sum(visit_counts.get(s, 0) for s in states)
    avg_count = total_count / len(states)

    if avg_count == 0:
        return 1.0
    return 1.0 / np.sqrt(avg_count)

def calculate_relative_novelty(visit_counts, trajectory, state_index, novelty_lag):
    """
    Calculate relative novelty of a state in a trajectory.

    Args:
        visit_counts: Dictionary mapping states to visit counts
        trajectory: The sequence of states
        state_index: Index of the state in the trajectory
        novelty_lag: Number of steps to look forward and backward

    Returns:
        Relative novelty score for the state
    """
    if state_index < novelty_lag or state_index >= len(trajectory) - novelty_lag:
        return 0.0  # Not enough context to calculate relative novelty

    # Get preceding and following states
    preceding_states = trajectory[state_index - novelty_lag:state_index]
    # Include current state in following states as per paper
    following_states = trajectory[state_index:state_index + novelty_lag + 1]

    # Calculate novelty of preceding and following states
    preceding_novelty = calculate_set_novelty(visit_counts, preceding_states)
    following_novelty = calculate_set_novelty(visit_counts, following_states)

    # Avoid division by zero
    if preceding_novelty == 0:
        return 0.0

    # Calculate relative novelty
    return following_novelty / preceding_novelty

def is_subgoal(state, rn_scores, p, q, priors_ratio, costs_ratio, threshold, return_proportion=False):
    """
    Decide if a state is a subgoal based on its relative novelty scores.

    Args:
        state: The state to evaluate
        rn_scores: Dictionary mapping states to lists of relative novelty scores
        p: P(x=1|Target) - probability of high RN score for target states
        q: P(x=1|Non-target) - probability of high RN score for non-target states
        priors_ratio: P(Non-target)/P(Target) - ratio of prior probabilities
        costs_ratio: λ_fa/λ_miss - ratio of costs for false alarms vs misses
        threshold: Relative novelty threshold

    Returns:
        True if the state is classified as a subgoal, False otherwise
    """
    if state not in rn_scores or not rn_scores[state]:
        return False

    # Count how many scores are above threshold
    scores = rn_scores[state]
    n = len(scores)
    n1 = sum(1 for score in scores if score > threshold)

    if n == 0:
        return False

    # Implement decision rule from equation 3 in the paper
    ln_term1 = np.log((1-q)/(1-p))
    ln_term2 = np.log(p*(1-q)/(q*(1-p)))
    ln_term3 = np.log(costs_ratio * priors_ratio)

    proportion = n1 / n
    decision_threshold = ln_term1 / ln_term2 + (ln_term3 / (n * ln_term2))

    if return_proportion:
        return [proportion, decision_threshold]

    return proportion > decision_threshold

def relative_novelty(
    trajectories: List[Trajectory],
    novelty_lag: int = 7,
    option_lag: int = 10,
    p: float = 0.0712,  # P(x=1|Target) from paper
    q: float = 0.0056,  # P(x=1|Non-target) from paper
    rn_threshold: float = 2.0,  # Relative novelty threshold from paper
    costs_ratio: float = 100.0,  # λ_fa/λ_miss from paper
    priors_ratio: float = 100.0,  # P(N)/P(T) from paper
    static_filter: Set[int] = None,
    return_scores: bool = False
) -> List[Subgoal]:
    """
    Discover subgoals using the relative novelty algorithm.

    Args:
        trajectories: List of trajectories (state sequences)
        novelty_lag: Number of steps to consider for relative novelty calculation
        option_lag: Number of steps to include in option's initiation set
        p: Probability of high relative novelty score for target states
        q: Probability of high relative novelty score for non-target states
        rn_threshold: Threshold for considering a relative novelty score as high
        costs_ratio: Ratio of costs for false alarms vs misses
        priors_ratio: Ratio of prior probabilities for non-targets vs targets
        static_filter: Set of states to exclude from consideration as subgoals

    Returns:
        List of subgoal states
    """
    if static_filter is None:
        static_filter = set()

    # Initialize visit counts and relative novelty scores
    visit_counts = {}
    rn_scores = defaultdict(list)

    # Process each trajectory
    for trajectory in trajectories:
        # Reset visit counts at the beginning of each trajectory as per paper
        visit_counts = {}

        # Count state visits
        for state in trajectory:

            # Skip terminal state
            if state == -1:
                continue

            if state in visit_counts:
                visit_counts[state] += 1
            else:
                visit_counts[state] = 1

        # Calculate relative novelty for each state in the trajectory
        for i in range(len(trajectory)):
            state = trajectory[i]

            # Skip terminal state
            if state == -1:
                continue

            # Skip states in static filter
            if state in static_filter:
                continue

            # Only calculate for states with enough context
            if i >= novelty_lag and i < len(trajectory) - novelty_lag:
                rn = calculate_relative_novelty(visit_counts, trajectory, i, novelty_lag)
                rn_scores[state].append(rn)

    if return_scores:
        # Return the relative novelty scores for all states
        rn_proportion_scores = {s: 0.0 for s, _ in rn_scores.items()}
        for state, scores in rn_scores.items():
            rn_proportion_scores[state] = is_subgoal(state, rn_scores, p, q, priors_ratio, costs_ratio, rn_threshold, return_proportion=True)
        return rn_proportion_scores

    # Identify subgoals using the decision rule
    subgoals = []
    for state, scores in rn_scores.items():
        if is_subgoal(state, rn_scores, p, q, priors_ratio, costs_ratio, rn_threshold):
            subgoals.append(state)

    return subgoals


def create_option_for_subgoal_rn(
    subgoal: Subgoal,
    trajectories: List[Trajectory],
    option_lag: int = 10
):
    """
    Create an option with an input set for a given subgoal by examining trajectories.
    This is specifically for options created using the relative novelty algorithm.

    Args:
        subgoal: The subgoal state
        trajectories: List of trajectories
        option_lag: Number of steps before subgoal to include in the input set

    Returns:
        Dictionary with input_set for the option
    """
    input_set = set()

    # Examine each trajectory for occurrences of the subgoal
    for trajectory in trajectories:
        if subgoal in trajectory:
            # Find all occurrences of the subgoal in the trajectory
            for i in range(len(trajectory)):
                if trajectory[i] == subgoal:
                    # Add states from option_lag steps before the subgoal to the input set
                    start_idx = max(0, i - option_lag)
                    for j in range(start_idx, i):
                        input_set.add(trajectory[j])

    return {"input_set": input_set}


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Subgoal Discovery in Two-Room Gridworld")
    parser.add_argument(
        "--method",
        type=str,
        choices=["dd", "rn"],
        default="dd",
        help="Method for subgoal discovery: 'dd' for diverse density, 'rn' for relative novelty"
    )

    args = parser.parse_args()
    method = args.method

    # Test for method
    if method == "dd":
        """
        Test case for diverse density subgoal discovery in a two-room gridworld.

        The gridworld is a 6x12 grid with two rooms connected by a hallway:
        - The hallway is at state 29
        - The goal is at state 68
        - The start is at state 24
        - Negative reward states are in a region in the first room

        We expect the diverse density algorithm to identify the hallway as a subgoal,
        because it appears in successful trajectories but not in unsuccessful ones.
        """
        from env import TwoRooms
        from agent import SMDPQLearning
        from trajectory_sampler import sample

        # Create the environment
        env = TwoRooms(
            start_state=24,
            goal_state=68,
            negative_states_config="default",
            max_steps=1000
        )

        # Set up the agent with primitive options (actions)
        from env import get_primitive_actions_as_options
        primitive_options = get_primitive_actions_as_options(env)

        agent = SMDPQLearning(
            env,
            primitive_options,
            learning_rate=0.1,
            discount_factor=0.99,
            exploration_rate=1.0,
            min_exploration_rate=0.1,
            exploration_decay=0.99
        )

        # Sample trajectories
        print("Sampling trajectories...")
        n_samples = 50
        trajectories, results, steps = sample(env, agent, n_samples)

        # Report on the samples
        success_rate = sum(results) / len(results)
        avg_steps = sum(steps) / len(steps)
        print(f"Success rate: {success_rate:.2f}, Average steps: {avg_steps:.2f}")

        # Define static filter to exclude start and goal states
        static_filter = {env.start_state, env.goal_state}

        # Calculate diverse density
        print("\nCalculating diverse density...")
        dd_values = diverse_density(trajectories, results, static_filter=static_filter)

        # Print the top 5 subgoal candidates
        print("\nTop subgoal candidates (state, diverse density):")
        for i, (state, dd) in enumerate(dd_values[:5], 1):
            print(f"{i}. State {state}: {dd:.4f}")

        # Calculate running average
        print("\nCalculating running average diverse density...")
        running_avg = running_average_dd(
            trajectories,
            results,
            lambda_value=0.9,
            static_filter=static_filter,
            threshold=0.5
        )

        # Print the top persistent subgoals
        print("\nPersistent subgoals (state, running average):")
        sorted_avg = sorted(running_avg.items(), key=lambda x: x[1], reverse=True)
        for i, (state, avg) in enumerate(sorted_avg[:5], 1):
            print(f"{i}. State {state}: {avg:.4f}")

        # Check if the hallway was identified as a subgoal
        hallway_state = env.hallway_state
        print(f"\nHallway state: {hallway_state}")

        if hallway_state in running_avg:
            print(f"Hallway state was identified as a subgoal with average: {running_avg[hallway_state]:.4f}")

            # Create option for hallway subgoal
            option_params = create_option_for_subgoal_dd(hallway_state, trajectories)
            print(f"\nCreated option with input set of size: {len(option_params['input_set'])}")
            print(f"Sample of input set: {list(option_params['input_set'])[:5]}...")
        else:
            print("Hallway state was not identified as a subgoal.")

        # Visualize where the discovered subgoals are in the environment
        print("\nEnvironment layout with discovered subgoals:")

        # Show a grid representation
        for row in range(6):
            line = ""
            for col in range(12):
                state = row * 12 + col
                if state == env.start_state:
                    line += "S "
                elif state == env.goal_state:
                    line += "G "
                elif state == env.hallway_state:
                    line += "H "
                elif state in running_avg:
                    line += "* "  # Subgoal
                elif state in env.negative_states:
                    line += "N "  # Negative reward state
                else:
                    line += ". "
            print(line)


    elif method == "rn":
        # Test case for relative novelty subgoal discovery
        from env import TwoRooms
        from agent import SMDPQLearning
        from trajectory_sampler import sample

        # Create the environment
        env = TwoRooms(
            start_state=24,
            goal_state=68,
            negative_states_config="default",
            max_steps=1000
        )

        # Set up the agent with primitive options (actions)
        from env import get_primitive_actions_as_options
        primitive_options = get_primitive_actions_as_options(env)

        agent = SMDPQLearning(
            env,
            primitive_options,
            learning_rate=0.1,
            discount_factor=0.99,
            exploration_rate=1.0,
            min_exploration_rate=0.1,
            exploration_decay=0.99
        )

        # Sample trajectories
        print("Sampling trajectories...")
        n_samples = 50
        trajectories, results, steps = sample(env, agent, n_samples)

        # Report on the samples
        success_rate = sum(results) / len(results)
        avg_steps = sum(steps) / len(steps)
        print(f"Success rate: {success_rate:.2f}, Average steps: {avg_steps:.2f}")

        # Define static filter to exclude start and goal states
        static_filter = {env.start_state, env.goal_state}

        # Run the relative novelty algorithm
        print("\nRunning relative novelty algorithm...")
        subgoals = relative_novelty(
            trajectories,
            novelty_lag=7,
            static_filter=static_filter,
            return_scores=False
        )

        # Print the discovered subgoals
        print(f"\nDiscovered {len(subgoals)} subgoals:")
        for i, subgoal in enumerate(subgoals, 1):
            print(f"{i}. State {subgoal}")

        # Check if the hallway was identified as a subgoal
        hallway_state = env.hallway_state
        print(f"\nHallway state: {hallway_state}")
        if hallway_state in subgoals:
            print("Hallway state was identified as a subgoal!")

            # Create option for hallway subgoal
            option_params = create_option_for_subgoal_rn(hallway_state, trajectories)
            print(f"\nCreated option with input set of size: {len(option_params['input_set'])}")
            print(f"Sample of input set: {list(option_params['input_set'])[:5]}...")
        else:
            print("Hallway state was not identified as a subgoal.")

        # Visualize where the discovered subgoals are in the environment
        print("\nEnvironment layout with discovered subgoals:")

        # Show a grid representation
        for row in range(6):
            line = ""
            for col in range(12):
                state = row * 12 + col
                if state == env.start_state:
                    line += "S "
                elif state == env.goal_state:
                    line += "G "
                elif state == env.hallway_state:
                    line += "H "
                elif state in subgoals:
                    line += "* "  # Subgoal
                elif state in env.negative_states:
                    line += "N "  # Negative reward state
                else:
                    line += ". "
            print(line)
