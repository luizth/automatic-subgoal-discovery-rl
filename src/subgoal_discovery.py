""" A collection of subgoal discovery algorithms (currently not working). """

from typing import List, Dict, Tuple, Set
import numpy as np
from collections import defaultdict

from core import Trajectory, State


# Terminal state in the environment
TERMINAL_STATE = -1


def diverse_density(
    trajectories: List[Trajectory],
    results: List[bool],
    static_filter: Set[int] = None,
    threshold: float = 0.0001
) -> List[State] | List[Tuple[State, float]]:
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

        # Skip terminal state
        if state == TERMINAL_STATE:
            continue

        dd_values[state] = calculate_dd(state, positive_bags, negative_bags)

    # Sort states by diverse density
    if threshold > 0:
        sorted_dd = sorted([(state, dd) for state, dd in dd_values.items() if dd > threshold],
                           key=lambda x: x[1], reverse=True)
    else:
        sorted_dd = sorted(dd_values.items(), key=lambda x: x[1], reverse=True)

    return sorted_dd


def calculate_dd(state: State, positive_bags: List[Trajectory], negative_bags: List[Trajectory]) -> float:
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
) -> Dict[State, float]:
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
