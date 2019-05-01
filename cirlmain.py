import gym
import numpy as np
from gym.envs.registration import register


from irl_benchmark.envs import envs_feature_based
import irl_benchmark.cirl.cirl

from irl_benchmark.envs import make_wrapped_env
from irl_benchmark.irl.collect import collect_trajs
from irl_benchmark.irl.feature import feature_wrapper
from irl_benchmark.rl.algorithms.value_iteration import ValueIteration
from irl_benchmark.utils.wrapper import unwrap_env
from irl_benchmark.rl.model.discrete_env import DiscreteEnvModelWrapper

# Run this script to generate all expert data.

# FROZEN LAKE:
env = feature_wrapper.make('FrozenLake-v0')
env = DiscreteEnvModelWrapper(env)

def rl_alg_factory(env):
    return ValueIteration(env, {'gamma': 0.9})


expert_agent = rl_alg_factory(env)
expert_agent.train(None)
expert_trajs = collect_trajs(
    env, expert_agent, 1000, None, 'data/frozen/expert/', verbose=True)

# a list of lists of feature vectors
features = [t['features'] for t in expert_trajs]
features = np.concatenate(features)
print(features.shape)
avg_features = features.mean(axis=0)


register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

wrapper = feature_wrapper.FrozenLakeFeatureWrapper
env = wrapper(gym.make('FrozenLakeNotSlippery-v0'))


cirl_test = irl_benchmark.cirl.cirl.ExhaustiveSearchCIRL(env)

print(cirl_test.cirl_trajectory(avg_features, limit=10000))
