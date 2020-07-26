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

def save_model(i_episode):
	print("Model Save...")
	if i_episode >= 2400:
		torch.save(agent.actor_local.state_dict(), './actors/actor_'+str(i_episode)+ '.pth')

# train controller for the env system
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
			state = env.reset()
			state_list.append(state)
		else: 
			assert len(state_list) == EP_NUM
			state = env.reset(state_list[ep][0], state_list[ep][1])
		print(state)
		trajectory.append(state)
		for t in range(201):
			action = agent.act(state, add_noise=False)[0]
			BP_action = individual_Bernstein_polynomial(state)

			fuel += 20 * abs(action)
			if BP:
				next_state, reward, done = env.step(BP_action)
			else:
				next_state, reward, done = env.step(action)
			total_reward += reward
			state = next_state
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
		plt.scatter(unsafe[:, 0], unsafe[:, 1], c='red', marker='*')
		plt.savefig(filename+'.png')
	return state_list, fuel_list, np.array(trajectory)

def individual_Bernstein_polynomial(state):
	x0 = state[0]
	x1 = state[1]
	
	if x0 <= 0 and x1 <= 0:#[5, 5] err 0.0567
		y = 0.000928948617280185*x0**5*x1**5 - 0.00916399869757268*x0**5*x1**4*(0.5*x1 + 1.0) + 0.0359492755565422*x0**5*x1**3*(0.5*x1 + 1.0)**2 - 0.0697812831650494*x0**5*x1**2*(0.5*x1 + 1.0)**3 + 0.0663546609086653*x0**5*x1*(0.5*x1 + 1.0)**4 - 0.0242989823616826*x0**5*(0.5*x1 + 1.0)**5 - 0.00901303508196066*x0**4*x1**5*(0.5*x0 + 1.0) + 0.0888179680344847*x0**4*x1**4*(0.5*x0 + 1.0)*(0.5*x1 + 1.0) - 0.345683835873506*x0**4*x1**3*(0.5*x0 + 1.0)*(0.5*x1 + 1.0)**2 + 0.662967804739889*x0**4*x1**2*(0.5*x0 + 1.0)*(0.5*x1 + 1.0)**3 - 0.620875261092142*x0**4*x1*(0.5*x0 + 1.0)*(0.5*x1 + 1.0)**4 + 0.219175156206897*x0**4*(0.5*x0 + 1.0)*(0.5*x1 + 1.0)**5 + 0.0342929325034368*x0**3*x1**5*(0.5*x0 + 1.0)**2 - 0.335026834570705*x0**3*x1**4*(0.5*x0 + 1.0)**2*(0.5*x1 + 1.0) + 1.29510688414621*x0**3*x1**3*(0.5*x0 + 1.0)**2*(0.5*x1 + 1.0)**2 - 2.45425797734338*x0**3*x1**2*(0.5*x0 + 1.0)**2*(0.5*x1 + 1.0)**3 + 2.23577864969541*x0**3*x1*(0.5*x0 + 1.0)**2*(0.5*x1 + 1.0)**4 - 0.747138020536159*x0**3*(0.5*x0 + 1.0)**2*(0.5*x1 + 1.0)**5 - 0.0649982352275076*x0**2*x1**5*(0.5*x0 + 1.0)**3 + 0.608680173858395*x0**2*x1**4*(0.5*x0 + 1.0)**3*(0.5*x1 + 1.0) - 2.3099942479049*x0**2*x1**3*(0.5*x0 + 1.0)**3*(0.5*x1 + 1.0)**2 + 4.2916198431204*x0**2*x1**2*(0.5*x0 + 1.0)**3*(0.5*x1 + 1.0)**3 - 3.78526479374207*x0**2*x1*(0.5*x0 + 1.0)**3*(0.5*x1 + 1.0)**4 + 1.13980752730271*x0**2*(0.5*x0 + 1.0)**3*(0.5*x1 + 1.0)**5 + 0.0606156325401482*x0*x1**5*(0.5*x0 + 1.0)**4 - 0.547379326256262*x0*x1**4*(0.5*x0 + 1.0)**4*(0.5*x1 + 1.0) + 1.91962458065328*x0*x1**3*(0.5*x0 + 1.0)**4*(0.5*x1 + 1.0)**2 - 3.40266427514137*x0*x1**2*(0.5*x0 + 1.0)**4*(0.5*x1 + 1.0)**3 + 2.81495386449012*x0*x1*(0.5*x0 + 1.0)**4*(0.5*x1 + 1.0)**4 - 0.623037268729621*x0*(0.5*x0 + 1.0)**4*(0.5*x1 + 1.0)**5 - 0.0219364595755122*x1**5*(0.5*x0 + 1.0)**5 + 0.185778797077615*x1**4*(0.5*x0 + 1.0)**5*(0.5*x1 + 1.0) - 0.580437016636*x1**3*(0.5*x0 + 1.0)**5*(0.5*x1 + 1.0)**2 + 0.787827976122108*x1**2*(0.5*x0 + 1.0)**5*(0.5*x1 + 1.0)**3 - 0.402989675793996*x1*(0.5*x0 + 1.0)**5*(0.5*x1 + 1.0)**4 - 0.0577596421143753*(0.5*x0 + 1.0)**5*(0.5*x1 + 1.0)**5

	elif x0 > 0 and x1 <= 0:#[5, 5] err 0.0706
		y = 0.000150032123969873*x0**5*x1**5 - 0.00308339119439689*x0**5*x1**4*(0.5*x1 + 1.0) + 0.0164295623165587*x0**5*x1**3*(0.5*x1 + 1.0)**2 - 0.0382896529619009*x0**5*x1**2*(0.5*x1 + 1.0)**3 + 0.0430193573968502*x0**5*x1*(0.5*x1 + 1.0)**4 - 0.0198514695158768*x0**5*(0.5*x1 + 1.0)**5 - 0.000253879367538489*x0**4*x1**5*(1 - 0.5*x0) - 0.0145394309352573*x0**4*x1**4*(1 - 0.5*x0)*(0.5*x1 + 1.0) + 0.128504707298381*x0**4*x1**3*(1 - 0.5*x0)*(0.5*x1 + 1.0)**2 - 0.327604701436284*x0**4*x1**2*(1 - 0.5*x0)*(0.5*x1 + 1.0)**3 + 0.381645250111604*x0**4*x1*(1 - 0.5*x0)*(0.5*x1 + 1.0)**4 - 0.180368510809315*x0**4*(1 - 0.5*x0)*(0.5*x1 + 1.0)**5 - 0.00763621744643007*x0**3*x1**5*(1 - 0.5*x0)**2 + 0.0086057362017819*x0**3*x1**4*(1 - 0.5*x0)**2*(0.5*x1 + 1.0) + 0.238691159214774*x0**3*x1**3*(1 - 0.5*x0)**2*(0.5*x1 + 1.0)**2 - 1.02972383540377*x0**3*x1**2*(1 - 0.5*x0)**2*(0.5*x1 + 1.0)**3 + 1.29748247447734*x0**3*x1*(1 - 0.5*x0)**2*(0.5*x1 + 1.0)**4 - 0.658779058746641*x0**3*(1 - 0.5*x0)**2*(0.5*x1 + 1.0)**5 - 0.0287775695337441*x0**2*x1**5*(1 - 0.5*x0)**3 + 0.150032624704958*x0**2*x1**4*(1 - 0.5*x0)**3*(0.5*x1 + 1.0) - 0.0564482137952344*x0**2*x1**3*(1 - 0.5*x0)**3*(0.5*x1 + 1.0)**2 - 1.08230617762456*x0**2*x1**2*(1 - 0.5*x0)**3*(0.5*x1 + 1.0)**3 + 2.06485615092688*x0**2*x1*(1 - 0.5*x0)**3*(0.5*x1 + 1.0)**4 - 1.14545087677975*x0**2*(1 - 0.5*x0)**3*(0.5*x1 + 1.0)**5 - 0.0443319446475492*x0*x1**5*(1 - 0.5*x0)**4 + 0.332444885751689*x0*x1**4*(1 - 0.5*x0)**4*(0.5*x1 + 1.0) - 0.785933550928012*x0*x1**3*(1 - 0.5*x0)**4*(0.5*x1 + 1.0)**2 + 0.344005620138051*x0*x1**2*(1 - 0.5*x0)**4*(0.5*x1 + 1.0)**3 + 1.08511143886861*x0*x1*(1 - 0.5*x0)**4*(0.5*x1 + 1.0)**4 - 0.748258780456722*x0*(1 - 0.5*x0)**4*(0.5*x1 + 1.0)**5 - 0.0219364595755122*x1**5*(1 - 0.5*x0)**5 + 0.185778797077615*x1**4*(1 - 0.5*x0)**5*(0.5*x1 + 1.0) - 0.580437016636*x1**3*(1 - 0.5*x0)**5*(0.5*x1 + 1.0)**2 + 0.787827976122108*x1**2*(1 - 0.5*x0)**5*(0.5*x1 + 1.0)**3 - 0.402989675793996*x1*(1 - 0.5*x0)**5*(0.5*x1 + 1.0)**4 - 0.0577596421143753*(1 - 0.5*x0)**5*(0.5*x1 + 1.0)**5

	elif x0 <= 0 and x1 > 0:#[5, 5] err 0.099
		y = -1.68237185631749e-5*x0**5*x1**5 - 0.00137265691589281*x0**5*x1**4*(1 - 0.5*x1) - 0.0128738605645146*x0**5*x1**3*(1 - 0.5*x1)**2 - 0.0400875482389648*x0**5*x1**2*(1 - 0.5*x1)**3 - 0.0522590867847153*x0**5*x1*(1 - 0.5*x1)**4 - 0.0242989823616826*x0**5*(1 - 0.5*x1)**5 - 0.000857136995611502*x0**4*x1**5*(0.5*x0 + 1.0) + 0.00365050759497323*x0**4*x1**4*(1 - 0.5*x1)*(0.5*x0 + 1.0) + 0.0636022984063337*x0**4*x1**3*(1 - 0.5*x1)**2*(0.5*x0 + 1.0) + 0.287262593040093*x0**4*x1**2*(1 - 0.5*x1)**3*(0.5*x0 + 1.0) + 0.433276773681636*x0**4*x1*(1 - 0.5*x1)**4*(0.5*x0 + 1.0) + 0.219175156206897*x0**4*(1 - 0.5*x1)**5*(0.5*x0 + 1.0) + 0.00793058201122367*x0**3*x1**5*(0.5*x0 + 1.0)**2 + 0.025894239272097*x0**3*x1**4*(1 - 0.5*x1)*(0.5*x0 + 1.0)**2 - 0.0898508510375479*x0**3*x1**3*(1 - 0.5*x1)**2*(0.5*x0 + 1.0)**2 - 0.642500488572981*x0**3*x1**2*(1 - 0.5*x1)**3*(0.5*x0 + 1.0)**2 - 1.29596615986417*x0**3*x1*(1 - 0.5*x1)**4*(0.5*x0 + 1.0)**2 - 0.747138020536159*x0**3*(1 - 0.5*x1)**5*(0.5*x0 + 1.0)**2 - 0.0299095841895402*x0**2*x1**5*(0.5*x0 + 1.0)**3 - 0.172563602727978*x0**2*x1**4*(1 - 0.5*x1)*(0.5*x0 + 1.0)**3 - 0.189230153834663*x0**2*x1**3*(1 - 0.5*x1)**2*(0.5*x0 + 1.0)**3 + 0.484882952902779*x0**2*x1**2*(1 - 0.5*x1)**3*(0.5*x0 + 1.0)**3 + 1.59618721155199*x0**2*x1*(1 - 0.5*x1)**4*(0.5*x0 + 1.0)**3 + 1.13980752730271*x0**2*(1 - 0.5*x1)**5*(0.5*x0 + 1.0)**3 + 0.0429022583350872*x0*x1**5*(0.5*x0 + 1.0)**4 + 0.328809661605968*x0*x1**4*(1 - 0.5*x1)*(0.5*x0 + 1.0)**4 + 0.827482377803221*x0*x1**3*(1 - 0.5*x1)**2*(0.5*x0 + 1.0)**4 + 0.712366052873993*x0*x1**2*(1 - 0.5*x1)**3*(0.5*x0 + 1.0)**4 - 0.184462714809*x0*x1*(1 - 0.5*x1)**4*(0.5*x0 + 1.0)**4 - 0.623037268729621*x0*(1 - 0.5*x1)**5*(0.5*x0 + 1.0)**4 - 0.0198729702610131*x1**5*(0.5*x0 + 1.0)**5 - 0.172419485487368*x1**4*(1 - 0.5*x1)*(0.5*x0 + 1.0)**5 - 0.552964576638355*x1**3*(1 - 0.5*x1)**2*(0.5*x0 + 1.0)**5 - 0.828644185698245*x1**2*(1 - 0.5*x1)**3*(0.5*x0 + 1.0)**5 - 0.523394158704696*x1*(1 - 0.5*x1)**4*(0.5*x0 + 1.0)**5 - 0.0577596421143753*(1 - 0.5*x1)**5*(0.5*x0 + 1.0)**5

	elif x0 > 0 and x1 > 0:#[5, 5] err 0.096
		y = -0.000861963318768725*x0**5*x1**5 - 0.00838880289222062*x0**5*x1**4*(1 - 0.5*x1) - 0.0326167222357075*x0**5*x1**3*(1 - 0.5*x1)**2 - 0.0613296039754442*x0**5*x1**2*(1 - 0.5*x1)**3 - 0.0559160626338112*x0**5*x1*(1 - 0.5*x1)**4 - 0.0198514695158768*x0**5*(1 - 0.5*x1)**5 - 0.00826681573687151*x0**4*x1**5*(1 - 0.5*x0) - 0.079969372754532*x0**4*x1**4*(1 - 0.5*x0)*(1 - 0.5*x1) - 0.306623718288423*x0**4*x1**3*(1 - 0.5*x0)*(1 - 0.5*x1)**2 - 0.584296310949316*x0**4*x1**2*(1 - 0.5*x0)*(1 - 0.5*x1)**3 - 0.528574914984719*x0**4*x1*(1 - 0.5*x0)*(1 - 0.5*x1)**4 - 0.180368510809315*x0**4*(1 - 0.5*x0)*(1 - 0.5*x1)**5 - 0.0313188237034999*x0**3*x1**5*(1 - 0.5*x0)**2 - 0.29905126993831*x0**3*x1**4*(1 - 0.5*x0)**2*(1 - 0.5*x1) - 1.12719547163565*x0**3*x1**3*(1 - 0.5*x0)**2*(1 - 0.5*x1)**2 - 2.10559425537224*x0**3*x1**2*(1 - 0.5*x0)**2*(1 - 0.5*x1)**3 - 1.95080932166931*x0**3*x1*(1 - 0.5*x0)**2*(1 - 0.5*x1)**4 - 0.658779058746641*x0**3*(1 - 0.5*x0)**2*(1 - 0.5*x1)**5 - 0.0592617106736781*x0**2*x1**5*(1 - 0.5*x0)**3 - 0.551113489069787*x0**2*x1**4*(1 - 0.5*x0)**3*(1 - 0.5*x1) - 2.02194466139783*x0**2*x1**3*(1 - 0.5*x0)**3*(1 - 0.5*x1)**2 - 3.64300900409365*x0**2*x1**2*(1 - 0.5*x0)**3*(1 - 0.5*x1)**3 - 3.28587372300366*x0**2*x1*(1 - 0.5*x0)**3*(1 - 0.5*x1)**4 - 1.14545087677975*x0**2*(1 - 0.5*x0)**3*(1 - 0.5*x1)**5 - 0.0547272187272542*x0*x1**5*(1 - 0.5*x0)**4 - 0.497797167055983*x0*x1**4*(1 - 0.5*x0)**4*(1 - 0.5*x1) - 1.76798959760862*x0*x1**3*(1 - 0.5*x0)**4*(1 - 0.5*x1)**2 - 3.04205965160895*x0*x1**2*(1 - 0.5*x0)**4*(1 - 0.5*x1)**3 - 2.49800606849821*x0*x1*(1 - 0.5*x0)**4*(1 - 0.5*x1)**4 - 0.748258780456722*x0*(1 - 0.5*x0)**4*(1 - 0.5*x1)**5 - 0.0198729702610131*x1**5*(1 - 0.5*x0)**5 - 0.172419485487368*x1**4*(1 - 0.5*x0)**5*(1 - 0.5*x1) - 0.552964576638355*x1**3*(1 - 0.5*x0)**5*(1 - 0.5*x1)**2 - 0.828644185698245*x1**2*(1 - 0.5*x0)**5*(1 - 0.5*x1)**3 - 0.523394158704696*x1*(1 - 0.5*x0)**5*(1 - 0.5*x1)**4 - 0.0577596421143753*(1 - 0.5*x0)**5*(1 - 0.5*x1)**5

	else:
		raise ValueError('undefined partition for Bernstein polynomial approximation')
	return y

if __name__ == '__main__':
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	random_seed = int(sys.argv[2])
	from itertools import count
	import time
	# for trained multuple actors	
	# agent = Agent(state_size=2, action_size=1, random_seed=random_seed, fc1_units=25, fc2_units=None, individual=False)
	# for individual distilled controller
	agent = Agent(state_size=2, action_size=1, random_seed=random_seed, fc1_units=50, fc2_units=None, individual=True)

	# scores = ddpg()
	# assert False

	# random intial state test to generate the scatter plot of safe and unsafe region
	# state_list, fuel_list, _ = test(agent, './ICCAD_models/Individual.pth', renew=True, state_list=[])
	# print(len(fuel_list), np.mean(fuel_list))

	# To compare the individual controller and Bernstein polynomial approximation controlled trajectory
	state_list, _, indi_trajectory = test(agent, './ICCAD_models/Individual.pth', renew=True, state_list=[], 
		EP_NUM=1, random_initial_test=False)
	state_list, _, BP_trajectory = test(agent, './ICCAD_models/Individual.pth', renew=False, state_list=state_list, 
		EP_NUM=1, random_initial_test=False, BP=True)
	plt.plot(indi_trajectory[:, 0], indi_trajectory[:, 1], label='individual')
	plt.plot(BP_trajectory[:, 0], BP_trajectory[:, 1], label='BP')
	plt.legend()
	plt.savefig('Control_Indi_BP.png')

	

