
"""Module for maximum entropy inverse reinforcement learning."""

from typing import Callable, Dict, List

import gym
import numpy as np
from irl_benchmark.rl.model.model_wrapper import BaseWorldModelWrapper
from irl_benchmark.utils.wrapper import unwrap_env, is_unwrappable_to
from irl_benchmark.cirl import cirl_wrapper
from math import inf
from tqdm import tqdm

# Type Alias
Trajectory = Dict[str, list]
Trajectories = List[Trajectory]

class BaseCIRLAlgorithm():

    def __init__(self, env: gym.Env):
        self.env = env

    def cirl_trajectories(self, expected_features: np.ndarray) -> Trajectories:
        raise NotImplimentedError()


class ExhaustiveSearchCIRL(BaseCIRLAlgorithm):
    """
    Exhausitvely search all trajectories, find the best one by the CIRL
    objective formula.
    """

    def __init__(self, env: gym.Env):
        """See :class:`irl_benchmark.irl.algorithms.base_algorithm.BaseIRLAlgorithm`."""
        super(ExhaustiveSearchCIRL, self).__init__(env)
        self.env = env

    def cirl_trajectory(self, expected_features, limit=None):
        """ Finds the most pedalogical trajectory, as defined by equation (1)
        in the CIRL paper.  This trajectory maximizes reward while being close
        to the expected feature values for the expert policy.
        """

        actions = np.arange(self.env.action_space.n)
        trajs = self._cartesian_product(*[actions for _ in range(10)])

        if limit is not None:
            trajs = trajs[:limit]

        rewards = np.zeros(len(trajs))
        for i, t in enumerate(tqdm(trajs)):
            rewards[i] = self.cirl_reward(cirl_env, t)

        print(np.argmax(rewards))

        return trajs[np.argmax(rewards)]

    def cirl_reward(self, cirl_env : gym.Env, traj : np.ndarray) -> float:
        cirl_env.reset()

        reward: float
        for a in traj:
            _ , reward, terminated, _ = cirl_env.step(a)
            if terminated:
                return reward

        return reward - cirl_env.feature_dist()

    def _cartesian_product(self,*arrays):
        """
        From https://stackoverflow.com/a/11146645
        """
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[...,i] = a
        return arr.reshape(-1, la)
