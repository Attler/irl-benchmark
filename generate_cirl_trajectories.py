import numpy as np

from irl_benchmark.envs import make_wrapped_env
from irl_benchmark.irl.collect import collect_trajs
from irl_benchmark.irl.feature import feature_wrapper
from irl_benchmark.rl.algorithms.value_iteration import ValueIteration
from irl_benchmark.rl.model.maze_world import MazeModelWrapper
from irl_benchmark.utils.wrapper import unwrap_env

# Run this script to generate all expert data.

# FROZEN LAKE:
env = feature_wrapper.make('FrozenLake-v0')

expected_feature_counts = feature_counts(expert_trajs)

# ensure that generated trajectories have the same format as expert trajs
cirl_trajectories = search_trajectories(env, expected_feature_counts)

# Then save them somewhere.
