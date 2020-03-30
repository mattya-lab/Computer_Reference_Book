# -*- coding: utf-8 -*-
import gym

env = gym.make('CartPole-v1')

# Start iterating
for episode in range(20):
    # Reset the environment
    observation = env.reset()
    # Iterate 100 times
    for step in range(100):
        # Render the environment
        env.render()
        
        # Print the current observation
        print(observation)
        
        # Take action
        action = env.action_space.sample()
        
        # Extract the observation, reward, status and
        # Other info based on the action taken
        observation, reward, done, info = env.step(action)
        if done: break
    
    print('Episode finished after {} timesteps'.format(step+1))
env.close()    


