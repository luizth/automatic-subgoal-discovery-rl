import itertools
from functools import reduce
import numpy as np

from env import NavigationEnv
from experience_buffer import ExperienceBuffer


def compute_empowerment(T, n_step, state, n_samples=1000):
    """
    Compute the empowerment of a state in a grid world
    T : numpy array, shape (n_states, n_actions, n_states)
        Transition matrix describing the probabilistic dynamics of a markov decision process
        (without rewards). Taking action a in state s, T describes a probability distribution
        over the resulting state as T[:,a,s]. In other words, T[s',a,s] is the probability of
        landing in state s' after taking action a in state s. The indices may seem "backwards"
        because this allows for convenient matrix multiplication.
    n_step : int
        Determines the "time horizon" of the empowerment computation. The computed empowerment is
        the influence the agent has on the future over an n_step time horizon.
    n_samples : int
        Number of samples for approximating the empowerment in the deterministic case.
    state : int
        State for which to compute the empowerment.
    """
    n_states, n_actions, _  = T.shape

    # Sample if too many action sequences
    if n_actions**n_step < 5000:
        nstep_samples = np.array( list( itertools.product( range(n_actions), repeat=n_step ) ) )
    else:
        nstep_samples = np.random.randint(0, n_actions, [n_samples, n_step])

    # Fold over each nstep actions, get unique end states
    def tmap(s, a):
        return np.argmax(T[:,a,s])

    seen = set()
    for i in range(len(nstep_samples)):
        aseq = nstep_samples[i,:]
        seen.add(reduce(tmap, [state,*aseq]))

    # Empowerment = log # of reachable states
    return np.log2(len(seen))


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

            # Assumes deterministic env
            T[next_s, a, s] = 1.

    return T


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
            T[ns, a, s] = 1.
        except IndexError:
            print(f"IndexError for state {s}, action {a}, next_state {ns}.")
            continue

    # Check the shape of the transition matrix
    print("Build transition matrix T of shape", T.shape)

    # Compute empowerment score for each state
    emp_scores = {}
    for s in range(unique_states.shape[0]):
        emp_scores[s] = compute_empowerment(T, n_step, s, n_samples=n_samples)
    emp_scores

    # Select highest empowerment scores
    empowerment_scores = sorted(emp_scores.items(), key=lambda state_emp: state_emp[1], reverse=True)
    selected = [state_emp[0] for state_emp in empowerment_scores[:k]]
    print("Selected %d states with highest empowerment scores" % k, selected)

    return selected

    # Subgoal tie break strategy
    # subgoal = np.random.choice(selected)
    # print("Subgoal randomly selected", subgoal)


if __name__ == "__main__":
    from env import RandomWalk

    env = RandomWalk(n=4, start_state=0, goal_state=3)

    T = __transition_matrix(env)

    assert list( T[:, 0, 0] ) == [1., 0., 0., 0.]
    assert list( T[:, 1, 0] ) == [0., 1., 0., 0.]
    assert list( T[:, 0, 1] ) == [1., 0., 0., 0.]
    assert list( T[:, 1, 1] ) == [0., 0., 1., 0.]
    assert list( T[:, 0, 2] ) == [0., 1., 0., 0.]
    assert list( T[:, 1, 2] ) == [0., 0., 0., 1.]
    assert list( T[:, 0, 3] ) == [0., 0., 1., 0.]
    assert list( T[:, 1, 3] ) == [0., 0., 0., 1.]

    emp_s0 = compute_empowerment(T, 5, 0)
    emp_s1 = compute_empowerment(T, 5, 1)
    emp_s2 = compute_empowerment(T, 5, 2)
    emp_s3 = compute_empowerment(T, 5, 3)

    print(emp_s0, emp_s1, emp_s2, emp_s3)


    from env import TwoRooms

    env = TwoRooms(start_state=24, goal_state=68, negative_states_config="none", max_steps=None)

    T = __transition_matrix(env)

    print(T.shape, T)

    emp_s = np.zeros(env.observation_space.n)
    for s in range(env.observation_space.n):
        emp_s[s] = compute_empowerment(T, 5, s)

    for s, emp in enumerate(emp_s):
        print(s, emp)
