
"""Module for maximum entropy inverse reinforcement learning."""

from typing import Callable, Dict, List

import gym
import numpy as np
from irl_benchmark.rl.model.model_wrapper import BaseWorldModelWrapper
from irl_benchmark.utils.wrapper import unwrap_env
from irl_benchmark.cirl import cirl_wrapper
from math import inf
import tqdm

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

        self.env = env

    def cirl_trajectory(self, expected_features):

        trajs = _cartesian_product( *[np.arange(env.action_space.n) for _ in range(10)])

        cirl_env = cirl_wrapper(self.env, expected_features)
        r = lambda t: self.cirl_reward(cirl_env, t)

        return max(trajs, key=r)

    def cirl_reward(cirl_env : gym.Env, traj : np.ndarray) -> int:
        cirl_env.reset()
        for a in traj:
            _ , reward, terminated, _ = cirl_env.step()
            if terminated:
                return reward

        # Trajectory didn't terminate :(
        return - inf

    def _cartesian_product(*arrays):
        """
        From https://stackoverflow.com/a/11146645
        """
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[...,i] = a
        return arr.reshape(-1, la)
