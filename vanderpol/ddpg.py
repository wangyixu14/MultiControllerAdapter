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

agent = Agent(state_size=2, action_size=1, random_seed=random_seed)
pathTxt = './log.txt'

def save_model(i_episode):
	print("Model Save...")
	if i_episode > 2400:
		torch.save(agent.actor_local.state_dict(), './models/actor_'+str(i_episode)+ '.pth')

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
			next_state, reward, done = env.step(action)
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

scores = ddpg()
assert False

def model_test(agent, filename, state_list, renew, fig):
	agent.actor_local.load_state_dict(torch.load(filename))
	dis = []
	velocity = []
	accleration = []
	fuel_list = []
	bp0 = []
	bp1=[]
	bp2=[]
	if renew:
		state_list = []
	for ep in range(1):
		total_reward = 0
		fuel = 0
		if renew:
			state = env.reset(-1.2, -1.5)
			state_list.append(state)
		else: 
			state = env.reset(state_list[ep][0], state_list[ep][1])
		# print(state)
		for t in range(201):
			action = agent.act(state, add_noise=False)[0]
			action_partition = BP2800partition(state)
			action_single_d3 = BP2800single_d3(state)
			action_single_d7 = BP2800single_d7(state)
			fuel += 20 * abs(action)
			next_state, reward, done = env.step(action)
			if ep == 0:
				dis.append(state[0])
				velocity.append(state[1])
				accleration.append(action*20)
				bp0.append(action_partition*20)
				bp1.append(action_single_d3*20)
				bp2.append(action_single_d7*20)
				# print(t, state, next_state, action_bp*20, abs(action-action_bp), done)
				# assert False
			total_reward += reward
			state = next_state
			if done:
				break
		if t >= 190:
			fuel_list.append(fuel) 
		print(ep, fuel)
	# fig = plt.figure(fig)
	# ax1 = fig.add_subplot(131)
	# ax2 = fig.add_subplot(132)
	# ax3 = fig.add_subplot(133)

	# ax1.title.set_text('x0')
	# ax2.title.set_text('x1')
	# ax3.title.set_text('u')

	# plt.subplot(1, 3, 1)
	# plt.plot(dis)
	# plt.subplot(1, 3, 2)
	# plt.plot(velocity, color='y')
	# plt.subplot(1, 3, 3)
	# plt.plot(accleration, color='g')
	# plt.savefig(filename+'_0.05.png')
	
	# plt.figure(2)
	plt.plot(accleration, label='NN')
	plt.plot(bp0, label='partition')
	plt.plot(bp1, label='single,d=3')
	plt.plot(bp2, label='single,d=7')
	plt.legend()
	plt.savefig('comparison.png')
	# np.save('./plot/rl1_ori.npy', np.array(accleration))
	# np.save('./plot/rl1_bp.npy', np.array(bp))
	# plt.plot(dis, velocity)
	# plt.savefig('test.png')
	return state_list, fuel_list

