#this file is going to train the adapter by Double DQN for hard switching and DDPG for weighted average, 
#and test for the safely control rate and energy consumption. 
import numpy as np
import torch
import torch.nn as nn
from Model import Actor, Individualtanh
import time
import torch.optim as optim
import random
from collections import deque, namedtuple
from torch.autograd import Variable
import math
from torch.utils.tensorboard import SummaryWriter
import sys
from env import Osillator
import scipy.io as io
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import gym
import torch.autograd as autograd
from interval import Interval
import os
from Agent import Agent, Weight_adapter

weight = float(sys.argv[2])
print(weight)

EXP1 = True

class ReplayBuffer(object):
	def __init__(self, capacity):
		self.buffer = deque(maxlen=capacity)
	
	def push(self, state, action, reward, next_state, done):
		state      = np.expand_dims(state, 0)
		next_state = np.expand_dims(next_state, 0)
			
		self.buffer.append((state, action, reward, next_state, done))
	
	def sample(self, batch_size):
		state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
		return np.concatenate(state), action, reward, np.concatenate(next_state), done
	
	def __len__(self):
		return len(self.buffer)

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
batch_size = 128
gamma = 0.99
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 3000
replay_buffer = ReplayBuffer(int(5e3))
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_1 = Actor(state_size=2, action_size=1, seed=0, fc1_units=25, fc2_units=None).to(device)
model_1.load_state_dict(torch.load("./models/actor_2800.pth"))
model_1.eval()

model_2 = Actor(state_size=2, action_size=1, seed=0, fc1_units=25).to(device)
if EXP1:
	model_2.load_state_dict(torch.load("./0731actors/actor_2400.pth"))
else:
	model_2.load_state_dict(torch.load("./0801actors/actor_1400.pth"))
model_2.eval()

Individual = Individualtanh(state_size=2, action_size=1, seed=0).to(device)

agent = Agent(state_size=2, action_size=2, random_seed=0, fc1_units=None, fc2_units=None, weighted=True)

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:
		os.makedirs(path)

def update_target(current_model, target_model):
	target_model.load_state_dict(current_model.state_dict())

class DQN(nn.Module):
	def __init__(self, num_inputs, num_actions):
		super(DQN, self).__init__()
		self.num_actions = num_actions
		self.layers = nn.Sequential(
			nn.Linear(num_inputs, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, num_actions)
		)
		
	def forward(self, x):
		return self.layers(x)

	def act(self, state, epsilon):
		if random.random() > epsilon:
			q_value = self.forward(state)
			action  = q_value.max(0)[1].item()
		else:
			action = random.randrange(self.num_actions)
		return action

def compute_td_loss(model, target_model, batch_size, optimizer):
	state, action, reward, next_state, done = replay_buffer.sample(batch_size)

	state      = Variable(torch.FloatTensor(np.float32(state)))
	next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
	action     = Variable(torch.LongTensor(action))
	reward     = Variable(torch.FloatTensor(reward))
	done       = Variable(torch.FloatTensor(done))

	q_values      = model(state)
	next_q_values = model(next_state)
	next_q_state_values = target_model(next_state) 

	q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

	next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
	expected_q_value = reward + gamma * next_q_value * (1 - done)
	
	loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
		
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	
	return loss

# train the adapter by Double DQN for multiple controllers switching
def train_adapter_hard():
	mkdir('./adapter_ab')
	env = Osillator()
	model = DQN(2, 2).to(device)
	target_model = DQN(2, 2).to(device)
	optimizer = optim.Adam(model.parameters())
	EP_NUM = 2001
	frame_idx = 0
	fuel_list = []
	ep_reward = deque(maxlen=100)

	for ep in range(EP_NUM):
		state = env.reset()
		ep_r = 0
		for t in range(200):
			state = torch.from_numpy(state).float().to(device)
			epsilon = epsilon_by_frame(frame_idx)
			action = model.act(state, epsilon)
			with torch.no_grad():
				if action == 0:
					control_action = model_1(state).cpu().data.numpy()[0]
				elif action == 1:
					control_action = model_2(state).cpu().data.numpy()[0]
				else: 
					assert False
					control_action = 0
			next_state, _, done = env.step(control_action)
			reward = 2
			reward -= weight * abs(control_action) * 20
			if done and t <190:
				reward -= 100
			replay_buffer.push(state.cpu().numpy(), action, reward, next_state, done)
			fuel_list.append(abs(control_action) * 20)
			state = next_state
			ep_r += reward
			frame_idx += 1
			if len(replay_buffer) > batch_size:
				loss = compute_td_loss(model, target_model, batch_size, optimizer)
			if frame_idx % 100 == 0:
				update_target(model, target_model)
			if done:
				break
		ep_reward.append(ep_r)
		print('epoch:', ep, 'reward:', ep_r, 'average reward:', np.mean(ep_reward),
					 'fuel cost:', sum(fuel_list[-t - 1:]), 'epsilon:', epsilon, len(replay_buffer)) 
		if ep >= 100 and ep % 100 == 0:
			torch.save(model.state_dict(), './adapter_ab/ddqn_'+str(ep)+'_'+str(weight)+'.pth')

# train the adapter for weight-sum the control inputs by multiple controllers
def train_adapter_weight(EP_NUM=2000):
	mkdir('./adapter_soft')
	env = Osillator()
	scores_deque = deque(maxlen=100)
	scores = []

	for ep in range(EP_NUM):
		state = env.reset()
		agent.reset()
		score = 0
		for t in range(200):
			action = agent.act(state)
			ca1 = model_1(torch.from_numpy(state).float().to(device)).cpu().data.numpy()[0]
			ca2 = model_2(torch.from_numpy(state).float().to(device)).cpu().data.numpy()[0]
			control_action = action[0]*ca1 + action[1]*ca2
			next_state, _, done = env.step(control_action, smoothness=0.5)
			reward = 5
			reward -= weight * abs(control_action) * 20
			reward -= 1 / weight * (abs(next_state[0]) + abs(next_state[1]))
			if done and t < 95:
				reward -= 100
			agent.step(state, action, reward, next_state, done, t)
			score += reward
			state = next_state            
			if done:
				break
		scores_deque.append(score)
		scores.append(score)
		score_average = np.mean(scores_deque)
		if ep % 1 == 0:
			print('\rEpisode {}, Average Score: {:.2f}, Current Score:{:.2f}, Max: {:.2f}, Min: {:.2f}, Epsilon: {:.2f}, Momery:{:.1f}'\
				  .format(ep, score_average,  scores[-1], np.max(scores), np.min(scores), agent.epsilon, len(agent.memory)), end="\n")     
		if ep > 0 and ep % 100 == 0:
			torch.save(agent.actor_local.state_dict(), './adapter_soft/adapter_'+str(ep)+'_'+str(weight)+ '.pth')

def plan(state, ca1, ca2):
	u1 = ca1*20
	u2 = ca2*20
	x0 = state[0] + 0.05 * state[1]
	x1 = state[1] + 0.05*((1-state[0]**2)*state[1] - state[0] + u1)
	
	x2 = state[0] + 0.05 * state[1]
	x3 = state[1] + 0.05*((1-state[0]**2)*state[1] - state[0] + u2)

	if x0**2 + x1**2 < x2**2 + x3**2:
		return ca1 
	else:
		return ca2 

def fgsm(model, X, epsilon=0.2):
	delta = torch.zeros_like(X, requires_grad=True)

	with torch.no_grad():
		y = model(X)
	noise = torch.from_numpy(np.random.uniform(low=-0.01, high=0.01, size=2)).to(device).float()
	loss = -nn.MSELoss()(model(X + delta+noise), y)
	loss.backward()
	return epsilon * delta.grad.detach().sign(), delta.grad.detach()

# test for 500 cases with their safely control rate and energy consumption
def test(adapter_name=None, state_list=None, renew=False, mode='switch', INDI_NAME=None):
	print(mode)
	env = Osillator()
	EP_NUM = 1
	if mode == 'switch':
		model = DQN(2, 2).to(device)
		model.load_state_dict(torch.load(adapter_name))
	if mode == 'weight':
		model = Weight_adapter(2, 2).to(device)
		model.load_state_dict(torch.load(adapter_name))
	if mode == 'individual':
		Individual.load_state_dict(torch.load(INDI_NAME))
	if renew:
		state_list = []
	fuel_list = []
	ep_reward = []
	trajectory = []
	safe = []
	unsafe = []
	control_action_list = []
	for ep in range(EP_NUM):
		if renew:
			state = env.reset()
			state_list.append(state)
		else:
			assert len(state_list) == EP_NUM
			state = env.reset(state_list[ep][0], state_list[ep][1])
		ep_r = 0
		fuel = 0
		if ep == 0:
			trajectory.append(state)
		for t in range(env.max_iteration):
			# attack happens here
			# state += np.random.uniform(low=-0.35, high=0.35, size=state.shape)
			state = torch.from_numpy(state).float().to(device)
			if mode == 'switch':
				action = model.act(state, epsilon=0)
				with torch.no_grad():
					if action == 0:
						control_action = model_1(state).cpu().data.numpy()[0]
					elif action == 1:
						control_action = model_2(state).cpu().data.numpy()[0]
					else:
						assert False
						control_action = 0
			elif mode == 'weight': 
				action = model(state).cpu().data.numpy()
				ca1 = model_1(state).cpu().data.numpy()[0]
				ca2 = model_2(state).cpu().data.numpy()[0]
				control_action = action[0]*ca1 + action[1]*ca2
				if ep == 0:
					print(t, state, control_action, action, ca1, ca2)
			elif mode == 'average':
				ca1 = model_1(state).cpu().data.numpy()[0]
				ca2 = model_2(state).cpu().data.numpy()[0]
				control_action = (ca1 + ca2)/2
			elif mode == 'planning':
				ca1 = model_1(state).cpu().data.numpy()[0]
				ca2 = model_2(state).cpu().data.numpy()[0]
				control_action = plan(state, ca1, ca2) 

			elif mode == 'd1':
				control_action = model_1(state).cpu().data.numpy()[0]

			elif mode == 'd2':
				control_action = model_2(state).cpu().data.numpy()[0]
				
			elif mode == 'individual':
				# delta, original = fgsm(Individual, state)
				# if ep == 0:
				# 	print(delta, original)
				# control_action = Individual(state+delta).cpu().data.numpy()[0]
				control_action = Individual(state).cpu().data.numpy()[0]

			next_state, reward, done = env.step(control_action)
			control_action = np.clip(control_action, -1, 1)
			fuel += abs(control_action) * 20
			state = next_state
			if ep == 0:
				trajectory.append(state)
				control_action_list.append(control_action)
			ep_r += reward
			if done:
				break
		
		ep_reward.append(ep_r)
		if t >= 95:
			fuel_list.append(fuel)
			safe.append(state_list[ep])
		else:
			print(ep, state_list[ep])
			unsafe.append(state_list[ep])
		if ep == 0:
			trajectory = np.array(trajectory)
			# plt.figure()
			plt.plot(trajectory[:, 0], trajectory[:, 1], label=mode)
			plt.legend()
			plt.savefig('trajectory.png')
	# safe = np.array(safe)
	# unsafe = np.array(unsafe)
	# plt.figure()
	# plt.scatter(safe[:, 0], safe[:, 1], c='green')
	# plt.scatter(unsafe[:, 0], unsafe[:, 1], c='red')
	# plt.savefig('./safe_sample_plot/'+ mode +'.png')
	return ep_reward, np.array(fuel_list), state_list, np.array(control_action_list)


def collect_data(adapter_name, INDI_NAME):
	assert EXP1 == True
	env = Osillator()
	model = Weight_adapter(2, 2).to(device)
	model.load_state_dict(torch.load(adapter_name))
	EP_NUM = 1500
	data_set = []
	for ep in range(EP_NUM):
		ep_loss = 0
		state = env.reset()
		for t in range(env.max_iteration):
			state = torch.from_numpy(state).float().to(device)
			action = model(state).cpu().data.numpy()
			with torch.no_grad():
				ca1 = model_1(state)
				ca2 = model_2(state)
			control_action = ca1*action[0] + ca2*action[1]

			next_state, reward, done = env.step(control_action.cpu().data.numpy()[0])
			data_set.append([state.cpu().data.numpy()[0], state.cpu().data.numpy()[1], control_action.cpu().data.numpy()[0]])
			state = next_state
			if done:
				break
		print(ep_loss, t)
	return np.array(data_set)
if __name__ == '__main__':
	# train_adapter_weight()
	# train_adapter_hard()
	# assert False
	state_list = np.load('init_state.npy')
	if EXP1:
		ADAPTER_NAME = './0731adapter/ddqn_1200_1.0_exce.pth'
		_, weight_fuel, _, w_action  = test('./0731adapter/adapter_300_exce.pth', state_list=state_list, renew=False, mode='weight')
		# lipschitz constant 34.8		
		_, indi_fuel, _, indi_action = test(None, state_list=state_list, renew=False, mode='individual', INDI_NAME='./direct_distill_tanh.pth')
		# lipschitz constant 15.4
		_, robust_fuel, _, robust_action = test(None, state_list=state_list, renew=False, mode='individual', INDI_NAME='./robust_distill_l2_new0824.pth')	
		_, sw_fuel, _, _  = test(ADAPTER_NAME, state_list=state_list, renew=False, mode='switch')
	else:
		ADAPTER_NAME = './0801adapter/ddqn_300_1.0.pth'
		_, weight_fuel, state_list  = test('./0801adapter/adapter_1700_1.0_extreme.pth', state_list=[], renew=True, mode='weight')
		_, sw_fuel, _  = test(ADAPTER_NAME, state_list=state_list, renew=False, mode='switch')
	
	# _, plan_fuel, _ = test(None, state_list=state_list, renew=False, mode='planning')
	# _, avg_fuel, _ = test(None, state_list=state_list, renew=False, mode='average')
	_, d1_fuel, _, _  = test(None, state_list=state_list, renew=False, mode='d1')
	_, d2_fuel, _, _  = test(None, state_list=state_list, renew=False, mode='d2')
	print(np.mean(weight_fuel), np.mean(indi_fuel), np.mean(robust_fuel),np.mean(sw_fuel), np.mean(d1_fuel), np.mean(d2_fuel), 
		 len(weight_fuel), len(indi_fuel), len(robust_fuel), len(sw_fuel), len(d1_fuel), len(d2_fuel))
	
	# print(np.mean(indi_fuel), np.mean(robust_fuel), len(indi_fuel), len(robust_fuel))
	
	plt.figure()
	plt.plot(w_action, label='weighted')
	plt.plot(indi_action, label='regular_distill')
	plt.plot(robust_action, label='robust_distill')
	plt.legend()
	plt.savefig('control_action.png')
	
	# np.save('init_state.npy', np.array(state_list))