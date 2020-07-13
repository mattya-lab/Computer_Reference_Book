# -*- coding: utf-8 -*-

import gym

name_map = ['CartPole-v1', 'MountainCar-v0', 'Pendulum-v0']
example = 2
env = gym.make(name_map[example])
env.reset()

for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    env.step(action)

env.close()


