import gym.spaces
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from interval import Interval
from env import Osillator

env = Osillator()

import os
import sys

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
	sys.path.append(module_path)

from Agent import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random_seed = int(sys.argv[2])

from itertools import count
import time

agent = Agent(state_size=2, action_size=1, random_seed=random_seed, fc1_units=64, fc2_units=64)
pathTxt = './log.txt'

def save_model(i_episode):
	print("Model Save...")
	if i_episode >= 2400:
		torch.save(agent.actor_local.state_dict(), './actors/actor_'+str(i_episode)+ '.pth')

def ddpg(n_episodes=10000, max_t=200, print_every=1, save_every=200):
	scores_deque = deque(maxlen=100)
	scores = []
	
	for i_episode in range(1, n_episodes+1):
		state = env.reset()
		agent.reset()
		score = 0
		timestep = time.time()
		for t in range(max_t):
			action = agent.act(state)[0]
			next_state, reward, done = env.step(action, smoothness=1)
			agent.step(state, action, reward, next_state, done, t)
			score += reward
			state = next_state            
			if done:
				break 
				
		scores_deque.append(score)
		scores.append(score)
		score_average = np.mean(scores_deque)
		
		if i_episode % save_every == 0:
			save_model(i_episode)
		
		if i_episode % print_every == 0:
			print('\rEpisode {}, Average Score: {:.2f}, Current Score:{:.2f}, Max: {:.2f}, Min: {:.2f}, Epsilon: {:.2f}, Momery:{:.1f}'\
				  .format(i_episode, score_average,  scores[-1], np.max(scores), np.min(scores), agent.epsilon, len(agent.memory)), end="\n")     
					
	return scores

# scores = ddpg()
# assert False

def model_test(agent, filename, renew, state_list=[]):
	agent.actor_local.load_state_dict(torch.load(filename))
	EP_NUM = 500
	safe = []
	unsafe = []
	fuel_list = []
	for ep in range(EP_NUM):
		total_reward = 0
		fuel = 0
		if renew:
			state = env.reset()
			state_list.append(state)
		else: 
			assert len(state_list) == EP_NUM
			state = env.reset(state_list[ep][0], state_list[ep][1])
		for t in range(201):
			action = agent.act(state, add_noise=False)[0]
			fuel += 20 * abs(action)
			next_state, reward, done = env.step(action)
			total_reward += reward
			state = next_state
			if done:
				break
		if t >= 95:
			fuel_list.append(fuel)
			safe.append(state_list[ep])
		else:
			unsafe.append(state_list[ep]) 
			print(ep, state_list[ep])
	safe = np.array(safe)
	unsafe = np.array(unsafe)
	plt.scatter(safe[:, 0], safe[:, 1], c='green')
	plt.scatter(unsafe[:, 0], unsafe[:, 1], c='red', marker='*')
	plt.savefig(filename+'.png')	
	return state_list, fuel_list 

state_list, fuel_list = model_test(agent, './actors/actor_2800.pth', renew=True, state_list=[])
print(len(fuel_list), np.mean(fuel_list))

