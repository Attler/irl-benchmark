
"""Module for maximum entropy inverse reinforcement learning."""

from typing import Callable, Dict, List

import gym
import numpy as np
from irl_benchmark.rl.model.model_wrapper import BaseWorldModelWrapper
from irl_benchmark.utils.wrapper import unwrap_env

# Type Alias
Trajectory = Dict[str, list]
Trajectories = List[Trajectories]

class BaseCIRLAlgorithm():

    def __init__(self, env: gym.Env):
        self.env = env

    def cirl_trajectories(self, expected_features: np.ndarray): -> Trajectories
        raise NotImplimentedError()


class ExaustiveSearchCIRL(BaseIRLAlgorithm):
    """
    Exhausitvely search all trajectories, find the best one by the CIRL
    objective formula.
    """

    def __init__(self, env: gym.Env):
        """See :class:`irl_benchmark.irl.algorithms.base_algorithm.BaseIRLAlgorithm`."""
        assert is_unwrappable_to(env, BaseWorldModelWrapper)
        super(ExaustiveSearchCIRL, self).__init__(env)

        self.model_wrapper = unwrap_env(env, BaseWorldModelWrapper)

        self.no_states = self.model_wrapper.n_states()
        self.no_actions = env.action_space.n
        self.transitions = self.model_wrapper.get_transition_array()


    def cirl_trajectories(self, expected_features):
        trajs = self.generate_trajs()
        best_traj = max(trajs, key=self.evaluate_traj)


    def generate_trajs(self) -> Trajectories:
        self.max_length = 10
        state = env.reset()
        for action in range(self.no_actions):
            pass

    def evaluate_traj(self, traj : Trajectory) -> float:
        raise NotImplimentedError()
