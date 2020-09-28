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
from env import Newenv
import scipy.io as io
from scipy.interpolate import interp2d
from mpl_toolkits.mplot3d import Axes3D

env = Newenv()

import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
	sys.path.append(module_path)
from Agent import Agent

SMOOTH=0.5

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:
		os.makedirs(path)

def save_model(i_episode, score_average):
	print("Model Save...")
	if i_episode > 2000 and i_episode % 200 == 0:
		torch.save(agent.actor_local.state_dict(), './actors/actor_'+str(SMOOTH)+str(i_episode)+ '.pth')

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
			next_state, reward, done = env.step(action, smoothness=SMOOTH)
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
				  .format(i_episode, score_average,  scores[-1], np.max(scores), np.min(scores), agent.epsilon, len(agent.memory)), t, end="\n")     
					
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
			# while True:
			# 	state = env.reset()
			# 	if where_inv_valuebased(state):
			# 		break
			state = env.reset()
			state_list.append(state)
		else: 
			assert len(state_list) == EP_NUM
			state = env.reset(state_list[ep][0], state_list[ep][1], state_list[ep][2])
		# print(state)
		if ep == 0:
			trajectory.append(state)
		for t in range(100):
			action = agent.act(state, add_noise=False)[0]
			BP_action = individual_Bernstein_polynomial(state)
			if ep == 0:
				print(state, abs(action-BP_action))
			fuel += 20 * abs(action)
			if BP:
				next_state, reward, done = env.step(BP_action)
			else:
				next_state, reward, done = env.step(action, smoothness=SMOOTH)
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
	# fig= plt.figure()
	# ax = Axes3D(fig)		
	safe = np.array(safe)
	unsafe = np.array(unsafe)
	if random_initial_test:
		plt.scatter(safe[:, 0], safe[:, 1], c='green')
		if unsafe.shape[0] > 0:
			plt.scatter(unsafe[:, 0], unsafe[:, 1],  c='red')
		plt.savefig(filename+'.png')
	return state_list, fuel_list, np.array(trajectory)

# please refer to ReachNN code for Bernstein Polynomial approximation
# ./models/Indi_exp1.pth
def individual_Bernstein_polynomial(state):
	x0 = state[0]
	x1 = state[1]
	if x0 <= 0 and x1 <= 0: 
		y = 0.000964176741595562*x0**5*x1**5 - 0.00954879236459382*x0**5*x1**4*(0.5*x1 + 1.0) + 0.0374478827332399*x0**5*x1**3*(0.5*x1 + 1.0)**2 - 0.0720764160755354*x0**5*x1**2*(0.5*x1 + 1.0)**3 + 0.067595628978273*x0**5*x1*(0.5*x1 + 1.0)**4 - 0.0241442410379195*x0**5*(0.5*x1 + 1.0)**5 - 0.0096019629450345*x0**4*x1**5*(0.5*x0 + 1.0) + 0.0949719430306287*x0**4*x1**4*(0.5*x0 + 1.0)*(0.5*x1 + 1.0) - 0.371081347390533*x0**4*x1**3*(0.5*x0 + 1.0)*(0.5*x1 + 1.0)**2 + 0.70590765098934*x0**4*x1**2*(0.5*x0 + 1.0)*(0.5*x1 + 1.0)**3 - 0.650623210598839*x0**4*x1*(0.5*x0 + 1.0)*(0.5*x1 + 1.0)**4 + 0.228651881312038*x0**4*(0.5*x0 + 1.0)*(0.5*x1 + 1.0)**5 + 0.0381421574676304*x0**3*x1**5*(0.5*x0 + 1.0)**2 - 0.376759881003129*x0**3*x1**4*(0.5*x0 + 1.0)**2*(0.5*x1 + 1.0) + 1.46505737075418*x0**3*x1**3*(0.5*x0 + 1.0)**2*(0.5*x1 + 1.0)**2 - 2.7632250409727*x0**3*x1**2*(0.5*x0 + 1.0)**2*(0.5*x1 + 1.0)**3 + 2.47391241762492*x0**3*x1*(0.5*x0 + 1.0)**2*(0.5*x1 + 1.0)**4 - 0.839027672589529*x0**3*(0.5*x0 + 1.0)**2*(0.5*x1 + 1.0)**5 - 0.07554667182105*x0**2*x1**5*(0.5*x0 + 1.0)**3 + 0.742760713183827*x0**2*x1**4*(0.5*x0 + 1.0)**3*(0.5*x1 + 1.0) - 2.87830569266639*x0**2*x1**3*(0.5*x0 + 1.0)**3*(0.5*x1 + 1.0)**2 + 5.34023495697006*x0**2*x1**2*(0.5*x0 + 1.0)**3*(0.5*x1 + 1.0)**3 - 4.62214862088532*x0**2*x1*(0.5*x0 + 1.0)**3*(0.5*x1 + 1.0)**4 + 1.49740732121627*x0**2*(0.5*x0 + 1.0)**3*(0.5*x1 + 1.0)**5 + 0.0744507278788396*x0*x1**5*(0.5*x0 + 1.0)**4 - 0.725257123752506*x0*x1**4*(0.5*x0 + 1.0)**4*(0.5*x1 + 1.0) + 2.78009431968883*x0*x1**3*(0.5*x0 + 1.0)**4*(0.5*x1 + 1.0)**2 - 5.01774282604165*x0*x1**2*(0.5*x0 + 1.0)**4*(0.5*x1 + 1.0)**3 + 4.00840355891442*x0*x1*(0.5*x0 + 1.0)**4*(0.5*x1 + 1.0)**4 - 1.0741975678468*x0*(0.5*x0 + 1.0)**4*(0.5*x1 + 1.0)**5 - 0.0289284066112742*x1**5*(0.5*x0 + 1.0)**5 + 0.277983820857859*x1**4*(0.5*x0 + 1.0)**5*(0.5*x1 + 1.0) - 1.04475476104712*x1**3*(0.5*x0 + 1.0)**5*(0.5*x1 + 1.0)**2 + 1.74395348850693*x1**2*(0.5*x0 + 1.0)**5*(0.5*x1 + 1.0)**3 - 1.16077995761166*x1*(0.5*x0 + 1.0)**5*(0.5*x1 + 1.0)**4 + 0.16673994758726*(0.5*x0 + 1.0)**5*(0.5*x1 + 1.0)**5
	elif x0 <= 0 and x1 > 0: 
		y = -3.86719433644732e-6*x0**5*x1**5 - 0.000950177773230977*x0**5*x1**4*(1 - 0.5*x1) - 0.0094174167049687*x0**5*x1**3*(1 - 0.5*x1)**2 - 0.0373183319892392*x0**5*x1**2*(1 - 0.5*x1)**3 - 0.0518532956489086*x0**5*x1*(1 - 0.5*x1)**4 - 0.0241442410379195*x0**5*(1 - 0.5*x1)**5 - 0.000587097274101406*x0**4*x1**5*(0.5*x0 + 1.0) + 0.00399898251302421*x0**4*x1**4*(1 - 0.5*x1)*(0.5*x0 + 1.0) + 0.0512665165121917*x0**4*x1**3*(1 - 0.5*x1)**2*(0.5*x0 + 1.0) + 0.280063144602121*x0**4*x1**2*(1 - 0.5*x1)**3*(0.5*x0 + 1.0) + 0.455199847941385*x0**4*x1*(1 - 0.5*x1)**4*(0.5*x0 + 1.0) + 0.228651881312038*x0**4*(1 - 0.5*x1)**5*(0.5*x0 + 1.0) + 0.0135506693448371*x0**3*x1**5*(0.5*x0 + 1.0)**2 + 0.00624529518472882*x0**3*x1**4*(1 - 0.5*x1)*(0.5*x0 + 1.0)**2 - 0.122502820695042*x0**3*x1**3*(1 - 0.5*x1)**2*(0.5*x0 + 1.0)**2 - 0.774522297311203*x0**3*x1**2*(1 - 0.5*x1)**3*(0.5*x0 + 1.0)**2 - 1.53913532897169*x0**3*x1*(1 - 0.5*x1)**4*(0.5*x0 + 1.0)**2 - 0.839027672589529*x0**3*(1 - 0.5*x1)**5*(0.5*x0 + 1.0)**2 - 0.0522214153570571*x0**2*x1**5*(0.5*x0 + 1.0)**3 - 0.297977281111066*x0**2*x1**4*(1 - 0.5*x1)*(0.5*x0 + 1.0)**3 - 0.192952615759263*x0**2*x1**3*(1 - 0.5*x1)**2*(0.5*x0 + 1.0)**3 + 0.882335079154534*x0**2*x1**2*(1 - 0.5*x1)**3*(0.5*x0 + 1.0)**3 + 2.35977379589333*x0**2*x1*(1 - 0.5*x1)**4*(0.5*x0 + 1.0)**3 + 1.49740732121627*x0**2*(1 - 0.5*x1)**5*(0.5*x0 + 1.0)**3 + 0.0665224810015959*x0*x1**5*(0.5*x0 + 1.0)**4 + 0.543463010098524*x0*x1**4*(1 - 0.5*x1)*(0.5*x0 + 1.0)**4 + 1.31172471547475*x0*x1**3*(1 - 0.5*x1)**2*(0.5*x0 + 1.0)**4 + 0.64286972374349*x0*x1**2*(1 - 0.5*x1)**3*(0.5*x0 + 1.0)**4 - 1.11904517341953*x0*x1*(1 - 0.5*x1)**4*(0.5*x0 + 1.0)**4 - 1.0741975678468*x0*(1 - 0.5*x1)**5*(0.5*x0 + 1.0)**4 - 0.029227615210937*x1**5*(0.5*x0 + 1.0)**5 - 0.270541126333301*x1**4*(1 - 0.5*x1)*(0.5*x0 + 1.0)**5 - 0.900306702504456*x1**3*(1 - 0.5*x1)**2*(0.5*x0 + 1.0)**5 - 1.19023536731864*x1**2*(1 - 0.5*x1)**3*(0.5*x0 + 1.0)**5 - 0.29530907446204*x1*(1 - 0.5*x1)**4*(0.5*x0 + 1.0)**5 + 0.16673994758726*(1 - 0.5*x1)**5*(0.5*x0 + 1.0)**5
	elif x0 > 0 and x1 <= 0: 
		y = -0.00050267909872911*x0**5*x1**5 + 0.00236742806266803*x0**5*x1**4*(0.5*x1 + 1.0) + 0.000608965701221193*x0**5*x1**3*(0.5*x1 + 1.0)**2 - 0.0274798206459645*x0**5*x1**2*(0.5*x1 + 1.0)**3 + 0.0514995953502878*x0**5*x1*(0.5*x1 + 1.0)**4 - 0.0274973929467122*x0**5*(0.5*x1 + 1.0)**5 - 0.00639699364835637*x0**4*x1**5*(1 - 0.5*x0) + 0.0444243568386306*x0**4*x1**4*(1 - 0.5*x0)*(0.5*x1 + 1.0) - 0.0812096760629223*x0**4*x1**3*(1 - 0.5*x0)*(0.5*x1 + 1.0)**2 - 0.0434352174061394*x0**4*x1**2*(1 - 0.5*x0)*(0.5*x1 + 1.0)**3 + 0.336185868808971*x0**4*x1*(1 - 0.5*x0)*(0.5*x1 + 1.0)**4 - 0.242886384483168*x0**4*(1 - 0.5*x0)*(0.5*x1 + 1.0)**5 - 0.0288794600718181*x0**3*x1**5*(1 - 0.5*x0)**2 + 0.235431408200851*x0**3*x1**4*(1 - 0.5*x0)**2*(0.5*x1 + 1.0) - 0.613597627492151*x0**3*x1**3*(1 - 0.5*x0)**2*(0.5*x1 + 1.0)**2 + 0.564083475352052*x0**3*x1**2*(1 - 0.5*x0)**2*(0.5*x1 + 1.0)**3 + 0.558967645258551*x0**3*x1*(1 - 0.5*x0)**2*(0.5*x1 + 1.0)**4 - 0.76307228817612*x0**3*(1 - 0.5*x0)**2*(0.5*x1 + 1.0)**5 - 0.0637896337324303*x0**2*x1**5*(1 - 0.5*x0)**3 + 0.558945968723295*x0**2*x1**4*(1 - 0.5*x0)**3*(0.5*x1 + 1.0) - 1.68692273760726*x0**2*x1**3*(1 - 0.5*x0)**3*(0.5*x1 + 1.0)**2 + 2.17271857871545*x0**2*x1**2*(1 - 0.5*x0)**3*(0.5*x1 + 1.0)**3 - 0.163239262531753*x0**2*x1*(1 - 0.5*x0)**3*(0.5*x1 + 1.0)**4 - 0.96104196902895*x0**2*(1 - 0.5*x0)**3*(0.5*x1 + 1.0)**5 - 0.0691447427071578*x0*x1**5*(1 - 0.5*x0)**4 + 0.639117739665037*x0*x1**4*(1 - 0.5*x0)**4*(0.5*x1 + 1.0) - 2.22772327600715*x0*x1**3*(1 - 0.5*x0)**4*(0.5*x1 + 1.0)**2 + 3.26899438080267*x0*x1**2*(1 - 0.5*x0)**4*(0.5*x1 + 1.0)**3 - 1.29599310512723*x0*x1*(1 - 0.5*x0)**4*(0.5*x1 + 1.0)**4 - 0.295073743754019*x0*(1 - 0.5*x0)**4*(0.5*x1 + 1.0)**5 - 0.0289284066112742*x1**5*(1 - 0.5*x0)**5 + 0.277983820857859*x1**4*(1 - 0.5*x0)**5*(0.5*x1 + 1.0) - 1.04475476104712*x1**3*(1 - 0.5*x0)**5*(0.5*x1 + 1.0)**2 + 1.74395348850693*x1**2*(1 - 0.5*x0)**5*(0.5*x1 + 1.0)**3 - 1.16077995761166*x1*(1 - 0.5*x0)**5*(0.5*x1 + 1.0)**4 + 0.16673994758726*(1 - 0.5*x0)**5*(0.5*x1 + 1.0)**5
	elif x0 > 0 and x1 > 0: 
		y = -0.00097475200862428*x0**5*x1**5 - 0.00973011153499758*x0**5*x1**4*(1 - 0.5*x1) - 0.038781408029841*x0**5*x1**3*(1 - 0.5*x1)**2 - 0.0768830680664916*x0**5*x1**2*(1 - 0.5*x1)**3 - 0.0749748631773671*x0**5*x1*(1 - 0.5*x1)**4 - 0.0274973929467122*x0**5*(1 - 0.5*x1)**5 - 0.00972918018711428*x0**4*x1**5*(1 - 0.5*x0) - 0.0969411700416709*x0**4*x1**4*(1 - 0.5*x0)*(1 - 0.5*x1) - 0.384886622201709*x0**4*x1**3*(1 - 0.5*x0)*(1 - 0.5*x1)**2 - 0.756928894177413*x0**4*x1**2*(1 - 0.5*x0)*(1 - 0.5*x1)**3 - 0.720185853119533*x0**4*x1*(1 - 0.5*x0)*(1 - 0.5*x1)**4 - 0.242886384483168*x0**4*(1 - 0.5*x0)*(1 - 0.5*x1)**5 - 0.038770511301036*x0**3*x1**5*(1 - 0.5*x0)**2 - 0.384883154068398*x0**3*x1**4*(1 - 0.5*x0)**2*(1 - 0.5*x1) - 1.51600505394132*x0**3*x1**3*(1 - 0.5*x0)**2*(1 - 0.5*x1)**2 - 2.93244451997169*x0**3*x1**2*(1 - 0.5*x0)**2*(1 - 0.5*x1)**3 - 2.66114244197267*x0**3*x1*(1 - 0.5*x0)**2*(1 - 0.5*x1)**4 - 0.76307228817612*x0**3*(1 - 0.5*x0)**2*(1 - 0.5*x1)**5 - 0.076945706597258*x0**2*x1**5*(1 - 0.5*x0)**3 - 0.758064113422372*x0**2*x1**4*(1 - 0.5*x0)**3*(1 - 0.5*x1) - 2.93763641664361*x0**2*x1**3*(1 - 0.5*x0)**3*(1 - 0.5*x1)**2 - 5.4946665415092*x0**2*x1**2*(1 - 0.5*x0)**3*(1 - 0.5*x1)**3 - 4.49872422354552*x0**2*x1*(1 - 0.5*x0)**3*(1 - 0.5*x1)**4 - 0.96104196902895*x0**2*(1 - 0.5*x0)**3*(1 - 0.5*x1)**5 - 0.0756408934159844*x0*x1**5*(1 - 0.5*x0)**4 - 0.731589627296397*x0*x1**4*(1 - 0.5*x0)**4*(1 - 0.5*x1) - 2.72106469234712*x0*x1**3*(1 - 0.5*x0)**4*(1 - 0.5*x1)**2 - 4.69359870124181*x0*x1**2*(1 - 0.5*x0)**4*(1 - 0.5*x1)**3 - 3.10293905487454*x0*x1*(1 - 0.5*x0)**4*(1 - 0.5*x1)**4 - 0.295073743754019*x0*(1 - 0.5*x0)**4*(1 - 0.5*x1)**5 - 0.029227615210937*x1**5*(1 - 0.5*x0)**5 - 0.270541126333301*x1**4*(1 - 0.5*x0)**5*(1 - 0.5*x1) - 0.900306702504456*x1**3*(1 - 0.5*x0)**5*(1 - 0.5*x1)**2 - 1.19023536731864*x1**2*(1 - 0.5*x0)**5*(1 - 0.5*x1)**3 - 0.29530907446204*x1*(1 - 0.5*x0)**5*(1 - 0.5*x1)**4 + 0.16673994758726*(1 - 0.5*x0)**5*(1 - 0.5*x1)**5
	else:
		raise ValueError('Undefined Partition')
	return y
def where_inv_valuebased(state):
	invariant = io.loadmat('./inv_value.mat')['V']
	x_loc = state[0]
	y_loc = state[1]
	x1 = np.linspace(-2.4, 2.4, 240)
	y1 = np.linspace(-2.4, 2.4, 240)
	inv1 = interp2d(x1, y1, invariant, kind='linear')(x_loc, y_loc)
	# print(inv1)
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
	agent = Agent(state_size=3, action_size=1, random_seed=random_seed, fc1_units=50, fc2_units=None, individual=True)

	# scores = ddpg()
	# assert False

	#random intial state test to generate the scatter plot of safe and unsafe region
	state_list = np.load('init_state.npy')
	_, fuel_list, trajectory = test(agent, './actors/actor_1.0_2800.pth', renew=False, state_list=state_list)
	print(len(fuel_list), np.mean(fuel_list))
	plt.plot(trajectory[:, 0], trajectory[:, 1])
	plt.savefig('actor_traj.png')
	# np.save('init_state.npy', np.array(state_list))
	# To compare the individual controller and Bernstein polynomial approximation controlled trajectory
	# state_list, _, indi_trajectory = test(agent, './models/Indi_exp1.pth', renew=True, state_list=[], 
	# 	EP_NUM=1, random_initial_test=False)
	# state_list, _, BP_trajectory = test(agent, './models/Indi_exp1.pth', renew=False, state_list=state_list, 
	# 	EP_NUM=1, random_initial_test=False, BP=True)
	# plt.plot(indi_trajectory[:, 0], indi_trajectory[:, 1], label='individual')
	# plt.plot(BP_trajectory[:, 0], BP_trajectory[:, 1], label='BP')
	# plt.legend()
	# plt.savefig('Control_Indi_BP.png')

	

