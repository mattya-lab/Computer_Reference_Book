# -*- coding: utf-8 -*-
import gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')
steps = []

# Start iterating
for episode in range(100):
    # Reset the environment
    observation = env.reset()
    # Iterate 100 times
    for step in range(100):
        # Render the environment
        env.render()
        
        _, _, th, _ = observation
        
        if th < 0:
            action = 0
        else:
            action = 1
        
        # Extract the observation, reward, status and
        # Other info based on the action taken
        observation, reward, done, info = env.step(action)
        if done: 
            break
    
    print('Episode {} finished after {} timesteps'.format(episode+1, step+1))
    steps.append(step+1)
    
env.close()   

plt.figure()
plt.plot(steps)
plt.xlabel('Episode')
plt.ylabel('Step')
plt.show()

