# this file is to train and test NN controller
# Invariant of Bernstein polynomial approximation is also shown here whose computation is referred to
# files in ./mat folder, where value-based method and polySOS are used
import gym.spaces
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from interval import Interval
from env import Osillator
import scipy.io as io
from scipy.interpolate import interp2d
from Model import Individualtanh

env = Osillator()

import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
	sys.path.append(module_path)
from Agent import Agent

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:
		os.makedirs(path)

def save_model(i_episode, score_average):
	print("Model Save...")
	if score_average > 300:
		torch.save(agent.actor_local.state_dict(), './actors/actor_'+str(i_episode)+ '.pth')

# train controller candidate for the env system
def ddpg(n_episodes=10000, max_t=200, print_every=1, save_every=200):
	mkdir('./actors')
	scores_deque = deque(maxlen=100)
	scores = []
	
	for i_episode in range(1, n_episodes+1):
		state = env.reset()
		agent.reset()
		score = 0
		timestep = time.time()
		for t in range(max_t):
			action = agent.act(state)[0]
			next_state, reward, done = env.step(action, smoothness=0.3)
			agent.step(state, action, reward, next_state, done, t)
			score += reward
			state = next_state            
			if done:
				break 
				
		scores_deque.append(score)
		scores.append(score)
		score_average = np.mean(scores_deque)
		
		if i_episode % save_every == 0:
			save_model(i_episode, score_average)
		
		if i_episode % print_every == 0:
			print('\rEpisode {}, Average Score: {:.2f}, Current Score:{:.2f}, Max: {:.2f}, Min: {:.2f}, Epsilon: {:.2f}, Momery:{:.1f}'\
				  .format(i_episode, score_average,  scores[-1], np.max(scores), np.min(scores), agent.epsilon, len(agent.memory)), end="\n")     
					
	return scores

# random intial state test for safe, unsafe region or 
# test the controlled trajectory for individual controller
def test(modelname, renew, state_list=[], EP_NUM=500, random_initial_test=True):
	model = Individualtanh(state_size=2, action_size=1, seed=0, fc1_units=25)
	model.load_state_dict(torch.load(modelname))
	safe = []
	unsafe = []
	fuel_list = []
	control_input = []
	if not random_initial_test:
		assert EP_NUM == 1
	for ep in range(EP_NUM):
		total_reward = 0
		fuel = 0
		if renew:
			while True:
				state = env.reset()
				if where_inv_valuebased(state):
					break
			# state = env.reset()
			state_list.append(state)
		else: 
			assert len(state_list) == EP_NUM
			state = env.reset(state_list[ep][0], state_list[ep][1])
		
		for t in range(100):
			u = model(torch.from_numpy(state).float()).detach().numpy()[0]
			fuel += 20 * abs(np.clip(u, -1, 1))
			next_state, reward, done = env.step(u)
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
	# if random_initial_test:
	# 	plt.scatter(safe[:, 0], safe[:, 1], c='green')
	# 	if unsafe.shape[0] > 0:
	# 		plt.scatter(unsafe[:, 0], unsafe[:, 1], c='red', marker='*')
	# 	plt.savefig(filename+'.png')
	control_input = np.array(control_input)
	# plt.plot(control_input[:, 0], label='nn')
	# plt.plot(control_input[:, 1], label='BP')
	# plt.legend()
	# plt.savefig('nn2BP.jpg')
	return state_list, fuel_list


def where_inv_valuebased(state):
	x = state[0] / 3
	y = state[1] / 3
	inv = -0.357383768019-0.0977143650762*x-0.0267949152538*y+0.662983677257*x**2-0.361192480951*x**2*y+0.410252670351*y**2+0.141324130177*y**3+0.349205035506*x**3-0.871585127725*x*y**3+0.384718470545*x**2*y**3+2.4959091816*x**3*y**3+0.439444426395*x*y**2-3.90897619843*x**2*y**2-1.56816467439*x**3*y**2+0.221712997844*x*y-0.316901787701*x**3*y+0.533815517192*x**4+1.43864453527*y**4-0.388073012871*x**5+1.05577057015*x**4*y-0.623034893834*x*y**4-0.110474858834*y**5-0.246895122972*x**6-0.466767461534*x**5*y+7.0796985825*x**4*y**2+4.19562725704*x**2*y**4+0.693863980004*x*y**5-1.0655262006*y**6+0.135886379069*x**7-0.689005901172*x**6*y+1.09589765935*x**5*y**2-0.720675129093*x**4*y**3+1.30156887551*x**3*y**4-0.135248531981*x**2*y**5+0.264868804654*x*y**6-0.00463431140825*y**7-0.16707732275*x**8+0.59845931213*x**7*y-3.03208198348*x**6*y**2-1.24626552468*x**5*y**3-8.77296493305*x**4*y**4-1.88879419528*x**3*y**5-0.634824553648*x**2*y**6+0.00199333871765*x*y**7-0.000108666568477*y**8
	return inv <= -0.01

if __name__ == '__main__':
	# print(where_inv_valuebased([1.9792317,  0.72447223]))
	# assert False
	
	# agent = Agent(state_size=2, action_size=1, random_seed=0, fc1_units=50, fc2_units=None, individual=True)
	# scores = ddpg()
	# assert False

	state_list, fuel_list = test('./robust_distill_0915.pth', renew=True, state_list=[], EP_NUM=1500)
	print(np.mean(fuel_list), len(fuel_list))


	

