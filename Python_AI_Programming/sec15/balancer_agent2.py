# -*- coding: utf-8 -*-

import gym
import numpy as np
import random
import matplotlib.pyplot as plt

class Agent:
    def __init__(self):
        self.Q = np.zeros((5**4, 2))
        self.last_s = None
        self.last_a = None
        
    def quantize5(self, x, a, b):
        return 0 if x < -a else 1 if x < -b else \
               2 if x <= b else 3 if x <= a else 4

    def quantize(self, obs):
        pos = self.quantize5(obs[0], 1.2, 0.2)
        vel = self.quantize5(obs[1], 1.5, 0.2)
        ang = self.quantize5(obs[2], 0.25, 0.02)
        acc = self.quantize5(obs[2], 1.0, 0.2)
        
        return pos + vel*5 + ang*25 + acc*125
    
    def action(self, obs, episode, reward):
        s = self.quantize(obs)
        if random.random() > 0.5 * (1 / (episode + 1)):
            a = np.argmax(self.Q[s,:])
        else:
            a = random.randint(0, 1)
        
        if self.last_s is not None:
            q = self.Q[self.last_s, self.last_a]
            self.Q[self.last_s, self.last_a] = \
              q + 0.2*(reward + 0.99*np.max(self.Q[s,:]) - q)
        self.last_s = s
        self.last_a = a
        return a

agent = Agent()

env = gym.make('CartPole-v1')
steps = []
# Start iterating
for episode in range(100):
    # Reset the environment
    observation = env.reset()
    
    reward = 0
    # Iterate 100 times
    for step in range(200):
        # Render the environment
        env.render()
        action = agent.action(observation, episode, reward)
        
        # Extract the observation, reward, status and
        # Other info based on the action taken
        observation, reward, done, info = env.step(action)
        if done: 
            agent.action(observation, episode, -200)
            break
    
    print('Episode {} finished after {} timesteps'.format(episode+1, step+1))
    steps.append(step+1)
    
env.close()   

plt.figure()
plt.plot(steps)
plt.xlabel('Episode')
plt.ylabel('Step')
plt.show()