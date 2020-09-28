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

# train controller for the env system
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
#test the controlled trajectory for individual controller and Bernstein polynomial approximation
def test(agent, filename, renew, state_list=[], EP_NUM=500, random_initial_test=True, BP=False):
	agent.actor_local.load_state_dict(torch.load(filename))
	safe = []
	unsafe = []
	fuel_list = []
	trajectory = []
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
		
		if ep == 0:
			trajectory.append(state)
		for t in range(100):
			action = agent.act(state, add_noise=False)[0]
			BP_action = individual_Bernstein_polynomial(state)
			if ep == 0:
				print(state, action, BP_action, abs(action-BP_action))
			fuel += 20 * abs(np.clip(action, -1, 1))
			if BP:
				next_state, reward, done = env.step(BP_action)
			else:
				next_state, reward, done = env.step(action)
			total_reward += reward
			state = next_state
			if ep == 0:
				trajectory.append(state)
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
	if random_initial_test:
		plt.scatter(safe[:, 0], safe[:, 1], c='green')
		if unsafe.shape[0] > 0:
			plt.scatter(unsafe[:, 0], unsafe[:, 1], c='red', marker='*')
		plt.savefig(filename+'.png')
	return state_list, fuel_list, np.array(trajectory)

# please refer to ReachNN code for Bernstein Polynomial approximation
# ./models/Indi_exp1.pth
def individual_Bernstein_polynomial(state):
	x0 = state[0]
	x1 = state[1]
	if x0 in Interval(-2, -1): #0.08
		y = 0.0435725822050459*(0.5 - 0.25*x1)*(-1.0*x0 - 1.0)**5*(0.5*x1 + 1)**4 + 0.00525112569711154*(0.5 - 0.25*x1)*(-1.0*x0 - 1.0)**4*(1.0*x0 + 2.0)*(0.5*x1 + 1)**4 + 0.970120610633042*(0.5 - 0.25*x1)*(-1.0*x0 - 1.0)**3*(0.5*x0 + 1)**2*(0.5*x1 + 1)**4 + 3.77518875708639*(0.5 - 0.25*x1)*(-1.0*x0 - 1.0)**2*(0.5*x0 + 1)**3*(0.5*x1 + 1)**4 + 5.40752779754075*(0.5 - 0.25*x1)*(-1.0*x0 - 1.0)*(0.5*x0 + 1)**4*(0.5*x1 + 1)**4 + 2.71396863613472*(0.5 - 0.25*x1)*(0.5*x0 + 1)**5*(0.5*x1 + 1)**4 + 0.031143533099941*(1 - 0.5*x1)**5*(-1.0*x0 - 1.0)**5 + 0.155168051068054*(1 - 0.5*x1)**5*(-1.0*x0 - 1.0)**4*(1.0*x0 + 2.0) + 1.23243979207846*(1 - 0.5*x1)**5*(-1.0*x0 - 1.0)**3*(0.5*x0 + 1)**2 + 2.42900977400397*(1 - 0.5*x1)**5*(-1.0*x0 - 1.0)**2*(0.5*x0 + 1)**3 + 2.35755584564433*(1 - 0.5*x1)**5*(-1.0*x0 - 1.0)*(0.5*x0 + 1)**4 + 0.898694593651958*(1 - 0.5*x1)**5*(0.5*x0 + 1)**5 + 0.309816961904818*(1 - 0.5*x1)**4*(-1.0*x0 - 1.0)**5*(0.25*x1 + 0.5) + 1.53530630733455*(1 - 0.5*x1)**4*(-1.0*x0 - 1.0)**4*(1.0*x0 + 2.0)*(0.25*x1 + 0.5) + 12.0610162133811*(1 - 0.5*x1)**4*(-1.0*x0 - 1.0)**3*(0.5*x0 + 1)**2*(0.25*x1 + 0.5) + 23.2443672600848*(1 - 0.5*x1)**4*(-1.0*x0 - 1.0)**2*(0.5*x0 + 1)**3*(0.25*x1 + 0.5) + 21.5509424419587*(1 - 0.5*x1)**4*(-1.0*x0 - 1.0)*(0.5*x0 + 1)**4*(0.25*x1 + 0.5) + 7.37839877958033*(1 - 0.5*x1)**4*(0.5*x0 + 1)**5*(0.25*x1 + 0.5) + 0.305612616960474*(1 - 0.5*x1)**3*(-1.0*x0 - 1.0)**5*(0.5*x1 + 1)**2 + 1.49563517639774*(1 - 0.5*x1)**3*(-1.0*x0 - 1.0)**4*(1.0*x0 + 2.0)*(0.5*x1 + 1)**2 + 11.4456212467953*(1 - 0.5*x1)**3*(-1.0*x0 - 1.0)**3*(0.5*x0 + 1)**2*(0.5*x1 + 1)**2 + 20.9274560343607*(1 - 0.5*x1)**3*(-1.0*x0 - 1.0)**2*(0.5*x0 + 1)**3*(0.5*x1 + 1)**2 + 17.4243589203542*(1 - 0.5*x1)**3*(-1.0*x0 - 1.0)*(0.5*x0 + 1)**4*(0.5*x1 + 1)**2 + 5.84262876682205*(1 - 0.5*x1)**3*(0.5*x0 + 1)**5*(0.5*x1 + 1)**2 + 0.280561913946506*(1 - 0.5*x1)**2*(-1.0*x0 - 1.0)**5*(0.5*x1 + 1)**3 + 1.25510169385051*(1 - 0.5*x1)**2*(-1.0*x0 - 1.0)**4*(1.0*x0 + 2.0)*(0.5*x1 + 1)**3 + 7.9746568158603*(1 - 0.5*x1)**2*(-1.0*x0 - 1.0)**3*(0.5*x0 + 1)**2*(0.5*x1 + 1)**3 + 13.5049836146893*(1 - 0.5*x1)**2*(-1.0*x0 - 1.0)**2*(0.5*x0 + 1)**3*(0.5*x1 + 1)**3 + 13.1763479301358*(1 - 0.5*x1)**2*(-1.0*x0 - 1.0)*(0.5*x0 + 1)**4*(0.5*x1 + 1)**3 + 5.13660667743809*(1 - 0.5*x1)**2*(0.5*x0 + 1)**5*(0.5*x1 + 1)**3 - 0.00877183218731313*(-1.0*x0 - 1.0)**5*(0.5*x1 + 1)**5 - 0.05014500015286*(-1.0*x0 - 1.0)**4*(1.0*x0 + 2.0)*(0.5*x1 + 1)**5 - 0.305753800119799*(-1.0*x0 - 1.0)**3*(0.5*x0 + 1)**2*(0.5*x1 + 1)**5 - 0.237696136194338*(-1.0*x0 - 1.0)**2*(0.5*x0 + 1)**3*(0.5*x1 + 1)**5 - 0.570204676058344*(-1.0*x0 - 1.0)*(0.5*x0 + 1)**4*(0.5*x1 + 1)**5 - 0.661244434670083*(0.5*x0 + 1)**5*(0.5*x1 + 1)**5
	elif x0 in Interval(-1, 0): 
		y = -0.0848115198792099*x0**5*(0.5 - 0.25*x1)*(0.5*x1 + 1)**4 - 0.0280842060516237*x0**5*(1 - 0.5*x1)**5 - 0.230574961861885*x0**5*(1 - 0.5*x1)**4*(0.25*x1 + 0.5) - 0.182582148963189*x0**5*(1 - 0.5*x1)**3*(0.5*x1 + 1)**2 - 0.16051895866994*x0**5*(1 - 0.5*x1)**2*(0.5*x1 + 1)**3 + 0.0206638885834401*x0**5*(0.5*x1 + 1)**5 - 0.04184199392751*x0**4*(0.5 - 0.25*x1)*(1.0*x0 + 1.0)*(0.5*x1 + 1)**4 + 0.136173953805368*x0**4*(1 - 0.5*x1)**5*(1.0*x0 + 1.0) + 0.863156804790327*x0**4*(1 - 0.5*x1)**4*(1.0*x0 + 1.0)*(0.25*x1 + 0.5) + 0.933804828674357*x0**4*(1 - 0.5*x1)**3*(1.0*x0 + 1.0)*(0.5*x1 + 1)**2 + 0.816778259477465*x0**4*(1 - 0.5*x1)**2*(1.0*x0 + 1.0)*(0.5*x1 + 1)**3 - 0.136747129322713*x0**4*(1.0*x0 + 1.0)*(0.5*x1 + 1)**5 + 1.01298503582607*x0**3*(0.5 - 0.25*x1)*(1.0*x0 + 1.0)**2*(0.5*x1 + 1)**4 - 0.2706949205799*x0**3*(1 - 0.5*x1)**5*(1.0*x0 + 1.0)**2 - 1.37723821501169*x0**3*(1 - 0.5*x1)**4*(1.0*x0 + 1.0)**2*(0.25*x1 + 0.5) - 2.03475438340932*x0**3*(1 - 0.5*x1)**3*(1.0*x0 + 1.0)**2*(0.5*x1 + 1)**2 - 1.70313140658258*x0**3*(1 - 0.5*x1)**2*(1.0*x0 + 1.0)**2*(0.5*x1 + 1)**3 + 0.298439496408126*x0**3*(1.0*x0 + 1.0)**2*(0.5*x1 + 1)**5 - 1.95252571298771*x0**2*(0.5 - 0.25*x1)*(1.0*x0 + 1.0)**3*(0.5*x1 + 1)**4 + 0.262688486581088*x0**2*(1 - 0.5*x1)**5*(1.0*x0 + 1.0)**3 + 1.98831546856502*x0**2*(1 - 0.5*x1)**4*(1.0*x0 + 1.0)**3*(0.25*x1 + 0.5) + 2.26547508511396*x0**2*(1 - 0.5*x1)**3*(1.0*x0 + 1.0)**3*(0.5*x1 + 1)**2 + 0.959727001648322*x0**2*(1 - 0.5*x1)**2*(1.0*x0 + 1.0)**3*(0.5*x1 + 1)**3 - 0.306521772809081*x0**2*(1.0*x0 + 1.0)**3*(0.5*x1 + 1)**5 + 1.2796609751422*x0*(0.5 - 0.25*x1)*(1.0*x0 + 1.0)**4*(0.5*x1 + 1)**4 - 0.133416154240592*x0*(1 - 0.5*x1)**5*(1.0*x0 + 1.0)**4 - 1.22464492926908*x0*(1 - 0.5*x1)**4*(1.0*x0 + 1.0)**4*(0.25*x1 + 0.5) - 0.924107257193429*x0*(1 - 0.5*x1)**3*(1.0*x0 + 1.0)**4*(0.5*x1 + 1)**2 - 0.0366567809946536*x0*(1 - 0.5*x1)**2*(1.0*x0 + 1.0)**4*(0.5*x1 + 1)**3 + 0.154950951365019*x0*(1.0*x0 + 1.0)**4*(0.5*x1 + 1)**5 - 0.286797657706196*(0.5 - 0.25*x1)*(1.0*x0 + 1.0)**5*(0.5*x1 + 1)**4 + 0.0289209061892386*(1 - 0.5*x1)**5*(1.0*x0 + 1.0)**5 + 0.258328797854661*(1 - 0.5*x1)**4*(1.0*x0 + 1.0)**5*(0.25*x1 + 0.5) + 0.113355051736745*(1 - 0.5*x1)**3*(1.0*x0 + 1.0)**5*(0.5*x1 + 1)**2 - 0.0168031083896429*(1 - 0.5*x1)**2*(1.0*x0 + 1.0)**5*(0.5*x1 + 1)**3 - 0.0311374359692583*(1.0*x0 + 1.0)**5*(0.5*x1 + 1)**5
	elif x0 in Interval(0, 1.5):
		y = -0.221576753742516*x0**3*(0.5 - 0.25*x1)*(0.5*x1 + 1)**2 + 0.0273370990887415*x0**3*(1 - 0.5*x1)**3 - 0.0103006373022517*x0**3*(1 - 0.5*x1)**2*(0.25*x1 + 0.5) - 0.0370368618050895*x0**3*(0.5*x1 + 1)**3 - 0.983472078896807*x0**2*(0.5 - 0.25*x1)*(1 - 0.666666666666667*x0)*(0.5*x1 + 1)**2 + 0.147911306407804*x0**2*(1 - 0.666666666666667*x0)*(1 - 0.5*x1)**3 + 0.146114530996197*x0**2*(1 - 0.666666666666667*x0)*(1 - 0.5*x1)**2*(0.25*x1 + 0.5) - 0.166657706328419*x0**2*(1 - 0.666666666666667*x0)*(0.5*x1 + 1)**3 - 1.34087740256057*x0*(0.5 - 0.25*x1)*(1 - 0.666666666666667*x0)**2*(0.5*x1 + 1)**2 + 0.24134252011*x0*(1 - 0.666666666666667*x0)**2*(1 - 0.5*x1)**3 + 0.0245134543834145*x0*(1 - 0.666666666666667*x0)**2*(1 - 0.5*x1)**2*(0.25*x1 + 0.5) - 0.249889216905718*x0*(1 - 0.666666666666667*x0)**2*(0.5*x1 + 1)**3 - 0.276145641532087*(0.5 - 0.25*x1)*(1 - 0.666666666666667*x0)**3*(0.5*x1 + 1)**2 + 0.115683624756955*(1 - 0.666666666666667*x0)**3*(1 - 0.5*x1)**3 + 0.426568730373086*(1 - 0.666666666666667*x0)**3*(1 - 0.5*x1)**2*(0.25*x1 + 0.5) - 0.124549743877033*(1 - 0.666666666666667*x0)**3*(0.5*x1 + 1)**3
	elif x0 in Interval(1.5, 2):
		y = -319.935817838493*(0.5 - 0.25*x1)*(1 - 0.5*x0)**5*(0.5*x1 + 1)**4 - 399.946957257211*(0.5 - 0.25*x1)*(1 - 0.5*x0)**4*(2.0*x0 - 3.0)*(0.5*x1 + 1)**4 - 1799.84218978684*(0.5 - 0.25*x1)*(1 - 0.5*x0)**3*(0.666666666666667*x0 - 1)**2*(0.5*x1 + 1)**4 - 1349.92174908893*(0.5 - 0.25*x1)*(1 - 0.5*x0)**2*(0.666666666666667*x0 - 1)**3*(0.5*x1 + 1)**4 - 126.557612017576*(0.5 - 0.25*x1)*(4.0 - 2.0*x0)*(0.666666666666667*x0 - 1)**4*(0.5*x1 + 1)**4 - 75.9352716888999*(0.5 - 0.25*x1)*(0.666666666666667*x0 - 1)**5*(0.5*x1 + 1)**4 + 23.6192536126727*(1 - 0.5*x0)**5*(1 - 0.5*x1)**5 + 11.7033615728635*(1 - 0.5*x0)**5*(1 - 0.5*x1)**4*(0.25*x1 + 0.5) - 107.394323437658*(1 - 0.5*x0)**5*(1 - 0.5*x1)**3*(0.5*x1 + 1)**2 - 315.434644447754*(1 - 0.5*x0)**5*(1 - 0.5*x1)**2*(0.5*x1 + 1)**3 - 31.9998485995974*(1 - 0.5*x0)**5*(0.5*x1 + 1)**5 + 28.0544239822634*(1 - 0.5*x0)**4*(1 - 0.5*x1)**5*(2.0*x0 - 3.0) - 5.01589388389946*(1 - 0.5*x0)**4*(1 - 0.5*x1)**4*(2.0*x0 - 3.0)*(0.25*x1 + 0.5) - 176.623122799601*(1 - 0.5*x0)**4*(1 - 0.5*x1)**3*(2.0*x0 - 3.0)*(0.5*x1 + 1)**2 - 395.656673749074*(1 - 0.5*x0)**4*(1 - 0.5*x1)**2*(2.0*x0 - 3.0)*(0.5*x1 + 1)**3 - 39.9998646034803*(1 - 0.5*x0)**4*(2.0*x0 - 3.0)*(0.5*x1 + 1)**5 + 118.8849731783*(1 - 0.5*x0)**3*(1 - 0.5*x1)**5*(0.666666666666667*x0 - 1)**2 - 110.865682887353*(1 - 0.5*x0)**3*(1 - 0.5*x1)**4*(0.666666666666667*x0 - 1)**2*(0.25*x1 + 0.5) - 973.470419210769*(1 - 0.5*x0)**3*(1 - 0.5*x1)**3*(0.666666666666667*x0 - 1)**2*(0.5*x1 + 1)**2 - 1785.13054391813*(1 - 0.5*x0)**3*(1 - 0.5*x1)**2*(0.666666666666667*x0 - 1)**2*(0.5*x1 + 1)**3 - 179.999564096466*(1 - 0.5*x0)**3*(0.666666666666667*x0 - 1)**2*(0.5*x1 + 1)**5 + 83.0589039999059*(1 - 0.5*x0)**2*(1 - 0.5*x1)**5*(0.666666666666667*x0 - 1)**3 - 148.970644569484*(1 - 0.5*x0)**2*(1 - 0.5*x1)**4*(0.666666666666667*x0 - 1)**3*(0.25*x1 + 0.5) - 848.162042751341*(1 - 0.5*x0)**2*(1 - 0.5*x1)**3*(0.666666666666667*x0 - 1)**3*(0.5*x1 + 1)**2 - 1341.51834081319*(1 - 0.5*x0)**2*(1 - 0.5*x1)**2*(0.666666666666667*x0 - 1)**3*(0.5*x1 + 1)**3 - 134.999783623425*(1 - 0.5*x0)**2*(0.666666666666667*x0 - 1)**3*(0.5*x1 + 1)**5 + 6.63717037659708*(1 - 0.5*x1)**5*(4.0 - 2.0*x0)*(0.666666666666667*x0 - 1)**4 + 2.85136700166291*(1 - 0.5*x1)**5*(0.666666666666667*x0 - 1)**5 - 19.8175367927388*(1 - 0.5*x1)**4*(4.0 - 2.0*x0)*(0.666666666666667*x0 - 1)**4*(0.25*x1 + 0.5) - 15.2992697998482*(1 - 0.5*x1)**4*(0.666666666666667*x0 - 1)**5*(0.25*x1 + 0.5) - 90.0526181076045*(1 - 0.5*x1)**3*(4.0 - 2.0*x0)*(0.666666666666667*x0 - 1)**4*(0.5*x1 + 1)**2 - 59.1304382190112*(1 - 0.5*x1)**3*(0.666666666666667*x0 - 1)**5*(0.5*x1 + 1)**2 - 125.965508731429*(1 - 0.5*x1)**2*(4.0 - 2.0*x0)*(0.666666666666667*x0 - 1)**4*(0.5*x1 + 1)**3 - 75.6740213246269*(1 - 0.5*x1)**2*(0.666666666666667*x0 - 1)**5*(0.5*x1 + 1)**3 - 12.6562365690179*(4.0 - 2.0*x0)*(0.666666666666667*x0 - 1)**4*(0.5*x1 + 1)**5 - 7.59374466437586*(0.666666666666667*x0 - 1)**5*(0.5*x1 + 1)**5	
	else:
		raise ValueError('Undefined Partition')
	return y

def where_inv_valuebased(state):
	invariant = io.loadmat('./inv.mat')['V']
	x_loc = state[0]
	y_loc = state[1]
	x1 = np.linspace(-2.4, 2.4, 240)
	y1 = np.linspace(-2.4, 2.4, 240)
	inv1 = interp2d(x1, y1, invariant, kind='linear')(x_loc, y_loc)
	return abs(inv1)<1e-8

def where_inv_polySOS(state):
	pass

if __name__ == '__main__':
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	random_seed = int(sys.argv[2])
	from itertools import count
	import time
	# print(where_inv_valuebased([-1.89283535, -1.86153018]))
	# assert False
	
	# for trained multuple actors	
	# agent = Agent(state_size=2, action_size=1, random_seed=random_seed, fc1_units=25, fc2_units=None, individual=False)
	# for individual distilled controller
	agent = Agent(state_size=2, action_size=1, random_seed=random_seed, fc1_units=50, fc2_units=None, individual=True)

	# scores = ddpg()
	# assert False

	#random intial state test to generate the scatter plot of safe and unsafe region
	state_list, fuel_list, trajectory = test(agent, './robust_distill_l2_new0824.pth', renew=True, state_list=[], EP_NUM=500)
	print(np.mean(fuel_list), len(fuel_list))
	# print('')
	# bpstate_list, bpfuel_list, bptrajectory = test(agent, './robust_distill_l2tanh.pth', renew=False, state_list=state_list, EP_NUM=1, BP=True)
	# print(len(fuel_list), len(bpfuel_list), np.mean(fuel_list), np.mean(bpfuel_list))
	# plt.plot(trajectory[:, 0], trajectory[:, 1], label='NN')
	# plt.plot(bptrajectory[:, 0], bptrajectory[:, 1], label='BP')
	# plt.legend()
	# plt.savefig('BPvsNN.png')
	# np.save('initial_state_500_poly10_err0.05.npy', np.array(state_list))

	# To compare the individual controller and Bernstein polynomial approximation controlled trajectory
	# state_list, _, indi_trajectory = test(agent, './models/Indi_exp1.pth', renew=True, state_list=[], 
	# 	EP_NUM=1, random_initial_test=False)
	# state_list, _, BP_trajectory = test(agent, './models/Indi_exp1.pth', renew=False, state_list=state_list, 
	# 	EP_NUM=1, random_initial_test=False, BP=True)
	# plt.plot(indi_trajectory[:, 0], indi_trajectory[:, 1], label='individual')
	# plt.plot(BP_trajectory[:, 0], BP_trajectory[:, 1], label='BP')
	# plt.legend()
	# plt.savefig('Control_Indi_BP.png')

	

