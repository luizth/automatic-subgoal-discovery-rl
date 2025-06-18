import itertools
import numpy as np
from scipy.stats import entropy

from env import NavigationEnv
from experience_buffer import ExperienceBuffer


def __transition_matrix(env: NavigationEnv):
    """
    Computes probabilistic transition matrix model T[s',a,s] for the environment: p(a,s'|s)

    This method currently makes a few assumptions:
    1. We have full knowledge of the environment transition dynamics
    2. Environment transitions are deterministic

    TODO: In future work this should use a agent's model of the environment transition dynamics
    """
    n_a = env.action_space.n
    n_s = env.observation_space.n

    actions = list(range(n_a))
    states = list(range(n_s))

    # Transition matrix: the probability in landing in s' given action is taken in state s
    T = np.zeros( (n_s, n_a, n_s) )
    for s in states:
        for a in actions:
            env.reset(options={"state": s})
            next_s, _,_,_,_ = env.step(a)
            if next_s == -1:
                # If the next state is -1, is terminal state
                continue

            # Assumes deterministic env
            T[s, a, next_s] = 1.
    return T


def calculate_empowerment_exact(T: dict, state: int, time_horizon: int, num_actions: int) -> float:
        """
        Calculate exact empowerment by enumerating all possible action sequences.
        Only feasible for small time horizons and action spaces.

        Args:
            state: State to calculate empowerment for

        Returns:
            Exact empowerment value for the state
        """
        # Generate all possible action sequences
        all_action_sequences = list(itertools.product(range(num_actions), repeat=time_horizon))

        # Calculate reachable state distribution for each action sequence
        all_final_states = []

        for action_sequence in all_action_sequences:
            current_state = state
            for action in action_sequence:
                if T[current_state, action, :].sum() == 0:
                    break  # If no transition exists, break
                current_state = np.argmax(T[current_state, action, :])
            all_final_states.append(current_state)

        # Calculate entropy of the reachable state distribution
        unique_states, counts = np.unique(all_final_states, return_counts=True)
        probabilities = counts / len(all_final_states)

        if len(probabilities) <= 1:
            return 0.0

        # Calculate empowerment as log2 of the number of unique reachable states for grid environments
        return np.log2(len(unique_states))  # - entropy(probabilities, base=2)


def empowerment_subgoal_discovery(eb: ExperienceBuffer, k=3, n_step=5, n_samples=1000):
    """ Discover subgoals based on empowerment scores from the experience buffer. """

    # Get experience buffer
    eb = eb.copy()
    print("Collected experience in buffer", eb.size)

    # Build transition matrix
    states, actions, next_states = eb.transition_matrix()
    assert states.shape == actions.shape == next_states.shape

    unique_states = np.unique(states)
    unique_actions = np.unique(actions)
    unique_next_states = np.unique(next_states)

    T = np.zeros((unique_next_states.shape[0], unique_actions.shape[0], unique_states.shape[0]), dtype=np.float64)
    for s, a, ns in zip(states, actions, next_states):
        try:
            T[s, a, ns] = 1.
        except IndexError:
            continue

    # Check the shape of the transition matrix
    print("Build transition matrix T of shape", T.shape)

    # Compute empowerment score for each state
    emp_scores = {}
    for s in range(unique_states.shape[0]):
        emp_scores[s] = calculate_empowerment_exact(T, n_step, s, n_samples=n_samples)
    emp_scores

    # Select highest empowerment scores
    empowerment_scores = sorted(emp_scores.items(), key=lambda state_emp: state_emp[1], reverse=True)
    selected = [state_emp[0] for state_emp in empowerment_scores[:k]]
    print("Selected %d states with highest empowerment scores" % k, selected)

    return selected


if __name__ == "__main__":
    from env import RandomWalk

    env = RandomWalk(n=4, start_state=0, goal_state=3)

    T = __transition_matrix(env)

    assert list( T[0, 0, :] ) == [1., 0., 0., 0.]
    assert list( T[0, 1, :] ) == [0., 1., 0., 0.]
    assert list( T[1, 0, :] ) == [1., 0., 0., 0.]
    assert list( T[1, 1, :] ) == [0., 0., 1., 0.]
    assert list( T[2, 0, :] ) == [0., 1., 0., 0.]
    assert list( T[2, 1, :] ) == [0., 0., 0., 1.]
    assert list( T[3, 0, :] ) == [0., 0., 1., 0.]
    assert list( T[3, 1, :] ) == [0., 0., 0., 1.]

    print( calculate_empowerment_exact(T, 0, 5, 2) )
