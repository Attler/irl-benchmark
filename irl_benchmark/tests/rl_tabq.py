import gym
import numpy as np
import unittest

from irl_benchmark.rl.algorithms.tabular_q import TabularQ


class TabQTestCase(unittest.TestCase):

    def test_frozen_finds_good_solution(self, duration=2):
        env = gym.make('FrozenLake-v0')
        agent = TabularQ(env)
        agent.train(duration)
        N = 100
        episode_rewards = []
        for episode in range(N):
            episode_reward = 0
            state = env.reset()
            done = False
            while not done:
                state, reward, done, _ = env.step(agent.pick_action(state))
                episode_reward += reward
            episode_rewards.append(episode_reward)

        if duration < 5:
            return

        assert np.mean(episode_rewards) > 0.4
        assert np.max(episode_rewards) == 1.0