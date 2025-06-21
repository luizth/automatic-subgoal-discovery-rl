from typing import Callable, Dict, List, Set
from abc import ABC, abstractmethod

import numpy as np

from torch.distributions import Categorical

from utils import to_tensor


State = int
Action = int
Dist = List[float]  # prob dist that sums to 1.
Policy = Dict[State, Dist]  # policy is a mapping from state to action distribution
DeterministicPolicy = Dict[State, Action]  # deterministic policy is a mapping from state to action
Trajectory = List[State]


class Subtask(ABC):
    """Subtasks are defined by GVFs"""
    env = None
    def __init__(self, env):
        pass
    @abstractmethod
    def c(self, last_state: int, state: int):
        """Cumulant function"""
        raise NotImplementedError
    @abstractmethod
    def B(self, state: int, w: np.ndarray):
        """Stopping function"""
        raise NotImplementedError
    @abstractmethod
    def z(self, state: int, w: np.ndarray):
        """Stopping value function"""
        raise NotImplementedError

class RewardRespectingFeatureAttainment(Subtask):
    def __init__(self, env, feature_attainment: int):
        self.env = env
        self.w = np.zeros(env.observation_space.n)  # weights for the features (main task)
        self.bonus_weight = 1.
        self.feature_attainment = feature_attainment

    def c(self, last_state, state):
        return self.env.get_reward(last_state)

    def B(self, state, w_option):
        """
            The option only stops if the stopping value (which is estimated main-task value + the bonus weight)
            is greater than or equal the estimated subtask value. That is, the option does not stop if the
            estimated subtask value of continuing is better than the stopping value.
        """
        state_features = self.env.state_to_features(state)
        return self.z(state) >= np.dot(w_option, state_features) or self.env.is_terminal(state)

    def z(self, state, local_w=None):
        """Stopping value"""
        state_features = self.env.state_to_features(state)
        feature_intensity = state_features[self.feature_attainment]
        if local_w is not None:
            feature_w = local_w[self.feature_attainment]
            return np.dot(local_w, state_features) + (self.bonus_weight - feature_w) * feature_intensity  # stopping bonus
        else:
            feature_w = self.w[self.feature_attainment]
            return np.dot(self.w, state_features) + (self.bonus_weight - feature_w) * feature_intensity  # stopping bonus


class Option:

    def __init__(self,
            id: str,
            initiation_set: Set[State],
            policy: Policy,  # Dict[State, Dist],
            termination: Callable[[State], bool]
        ):

        self.id = id
        self.initiation_set = initiation_set
        self.policy = policy
        self.termination = termination

    def choose_action(self, state):
        action_dist = to_tensor(self.policy[state])
        action_dist = Categorical(action_dist)
        action = action_dist.sample()
        return action.item()

    def terminate(self, state):
        if state not in self.initiation_set:
            return True
        return self.termination(state)

    def save_policy(self, path: str):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)

    @staticmethod
    def load(path: str):
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)

    def __repr__(self):
        return f"""
ID: {self.id}
Initiation Set: {self.initiation_set}
Policy: {self.policy}
Termination: {self.termination}
        """
