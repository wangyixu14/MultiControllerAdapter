import gym.spaces
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from env import ContinuousCartPoleEnv
# %matplotlib inline

import os
import sys

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
	sys.path.append(module_path)

from Agent import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("Device: ", device)

random_seed = int(sys.argv[2])
print(random_seed)

# env = gym.make('MountainCarContinuous-v0')
env = ContinuousCartPoleEnv()
env.seed(random_seed)

# size of each action
action_size = env.action_space.shape[0]
print('Size of each action:', action_size)

# examine the state space 
state_size = env.observation_space.shape[0]
print('Size of state:', state_size)

action_low = env.action_space.low
print('Action low:', action_low)

action_high = env.action_space.high
print('Action high: ', action_high)

from itertools import count
import time

agent = Agent(state_size=state_size, action_size=action_size, random_seed=random_seed)
pathTxt = './log.txt'

def save_model(ep, seed):
	print("Model Save...")
	torch.save(agent.actor_local.state_dict(), 'actor'+ str(ep) + '_' + str(seed)+'.pth')
	torch.save(agent.critic_local.state_dict(), 'critic'+ str(ep) + '_' + str(seed) +'.pth')

def ddpg(n_episodes=50000, max_t=200, print_every=1, save_every=50):
	scores_deque = deque(maxlen=100)
	scores = []
	
	for i_episode in range(1, n_episodes+1):
		state = env.reset()
		agent.reset()
		score = 0
		timestep = time.time()
		for t in range(max_t):
			action = agent.act(state)
			next_state, reward, done, _ = env.step(action)
			agent.step(state, action, reward, next_state, done, t)
			score += reward
			state = next_state            
			if done:
				break 
				
		scores_deque.append(score)
		scores.append(score)
		score_average = np.mean(scores_deque)
		
		if i_episode % save_every == 0 and np.mean(scores_deque) >= 150:
			save_model(i_episode, random_seed)
		
		if i_episode % print_every == 0:
			print('\rEpisode {}, Average Score: {:.2f}, Current Score:{:.2f}, Max: {:.2f}, Min: {:.2f}, Time: {:.2f}, Momery:{:.1f}'\
				  .format(i_episode, score_average,  scores[-1], np.max(scores), np.min(scores), time.time() - timestep, len(agent.memory)), end="\n")
					
		# if i_episode >= 20 and np.mean(scores_deque) >= 150:            
		# 	save_model(random_seed)

			# file = open(pathTxt, 'a')
			# file.write(str(i_episode)+'\n')
			# file.close()
			# print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, score_average))            
			# break            
			
			
	return scores

# scores = ddpg()
# assert False

agent.actor_local.load_state_dict(torch.load('actor4850_1.pth'))
# agent.critic_local.load_state_dict(torch.load('critic1.pth'))

state_list = np.load('init_state.npy')
fuel_list = []
for ep in range(500):
	total_reward = 0
	fuel = 0
	# state = state_list[ep]
	# state = env.reset(state=state, set_state=True)
	state = env.reset()
	for t in range(200):
		action = agent.act(state, add_noise=False)
		print(action, type(action))
		assert False
		fuel += abs(action)
		state, reward, done, _ = env.step(action)
		total_reward += reward
		if done:
			break
	print(t, total_reward)
	if t == 199:
		fuel_list.append(fuel) 
# np.save('init_state.npy', np.array(state_list))
print(len(fuel_list)/500, np.mean(fuel_list))
env.close()