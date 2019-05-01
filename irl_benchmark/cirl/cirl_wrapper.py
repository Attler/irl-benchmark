import gym
import numpy as np
from gym.envs.registration import register

import irl_benchmark
from irl_benchmark.irl.feature import feature_wrapper
from irl_benchmark.utils.wrapper import is_unwrappable_to


class CIRLRewards(gym.Wrapper):
    """Wrapper that changes the rewards recieved to match the CIRL algorithm,
    e.g. the reward is penalized by it's l2 distance from the expected feature
    distribution of the expert policy.

    The enviroment should return a feature vector in the info dictionary.
     """


    def __init__(self, env: gym.Env, expert_features):
        """

        Parameters
        ----------
        env: gym.Env
            The gym environment to be wrapped.

        expert_features: numpy array of size (d)
            where d is the feature size of the env
            The feature distribution we are trying to match
        """
        super(CIRLRewards, self).__init__(env)
        self.expert_features = expert_features
        self.d = env.feature_dimensionality[0]

        assert is_unwrappable_to(env, feature_wrapper)
        assert len(self.expert_features) == d

        self.feature_trajectory = np.empty((0,d))

    def step(self):
        next_state, reward, terminated, info = self.env.step(action)
        f = info["features"]
        self.feature_trajectory.append(f)

        if terminated:
            reward -= self.feature_dist()

        return next_state, reward, terminated, info


    def feature_dist(self):
        """ Returns the l2 distance of the average features for the current
        trajectory and the expert features.
        """

        trajectory_average_features = self.feature_trajectory.mean(axis=0)
        dist_vec = trajectory_average_features - self.expert_features
        return np.linalg.norm(dist_vec, ord=2)
