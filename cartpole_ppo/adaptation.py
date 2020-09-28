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
import scipy.io as io
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import gym
import torch.autograd as autograd
from interval import Interval
import os
from Agent import Agent
from ppo import PPO
from env import ContinuousCartPoleEnv

weight = 1
print(weight)

ATTACK = False
SCALE1 = 0.24
SCALE2 = 0.02

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

model_1 = Actor(state_size=4, action_size=1, seed=0).to(device)
model_1.load_state_dict(torch.load("./actor5000_1.pth"))
model_1.eval()

model_2 = Actor(state_size=4, action_size=1, seed=0).to(device)
model_2.load_state_dict(torch.load("./actor4850_1.pth"))
model_2.eval()

Individual = Individualtanh(state_size=4, action_size=1, seed=0, fc1_units=50).to(device)

agent = Agent(state_size=4, action_size=2, random_seed=0)

ppo = PPO(4, 2, method = 'penalty')
ppo.load_model(5499, 1)

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

# train the switcher by Double DQN for multiple controllers switching
def train_switcher_DDQN():
	mkdir('./switch')
	env = ContinuousCartPoleEnv()
	model = DQN(4, 2).to(device)
	target_model = DQN(4, 2).to(device)
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
					control_action = model_1(state).cpu().data.numpy()
				elif action == 1:
					control_action = model_2(state).cpu().data.numpy()
				else: 
					assert False
					control_action = 0
			next_state, _, done, _ = env.step(control_action)
			reward = 5
			reward -= weight * abs(control_action)
			if done and t != 199:
				reward -= 50
			replay_buffer.push(state.cpu().numpy(), action, reward, next_state, done)
			fuel_list.append(abs(control_action))
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
			torch.save(model.state_dict(), './switch/ddqn_'+str(ep)+'_'+str(weight)+'.pth') 

def fgsm(model, X, epsilon1=SCALE1, epsilon2=SCALE2):
	delta = torch.zeros_like(X, requires_grad=True)

	with torch.no_grad():
		y = model(X)
	noise = torch.from_numpy(np.random.uniform(low=-0.002, high=0.002, size=4)).to(device).float()
	loss = -nn.MSELoss()(model(X + delta+noise), y)
	loss.backward()
	indicator = delta.grad.detach().sign()
	indicator[0] *= epsilon1 
	indicator[1] *= 0
	indicator[2] *= epsilon2
	indicator[-1] *= 0
	return indicator
# test for 500 cases with their safely control rate and energy consumption
def test(adapter_name=None, state_list=None, renew=False, mode='switch', INDI_NAME=None):
	print(mode)
	env = ContinuousCartPoleEnv()
	EP_NUM = 500
	if mode == 'switch':
		model = DQN(4, 2).to(device)
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
			state = env.reset(state=state_list[ep], set_state=True)
		ep_r = 0
		fuel = 0
		if ep == 0:
			trajectory.append(state)
		for t in range(200):
			state = torch.from_numpy(state).float().to(device)
			if mode == 'switch':
				action = model.act(state, epsilon=0)
				with torch.no_grad():
					if action == 0:
						control_action = model_1(state).cpu().data.numpy()
					elif action == 1:
						control_action = model_2(state).cpu().data.numpy()
					else:
						assert False
						control_action = 0
			
			elif mode == 'ppo':
				action = ppo.choose_action(state.cpu().data.numpy(), True)
				ca1 = model_1(state).cpu().data.numpy()[0]
				ca2 = model_2(state).cpu().data.numpy()[0]
				control_action = np.array([action[0]*ca1 + action[1]*ca2])
				control_action = np.clip(control_action, -1, 1)
				if ep == 0:
					print(t, state, control_action, action, ca1, ca2)				

			elif mode == 'd1':
				control_action = model_1(state).cpu().data.numpy()

			elif mode == 'd2':
				control_action = model_2(state).cpu().data.numpy()
				
			elif mode == 'individual':
				if ATTACK:
					if t % 15 == 0:
						delta = fgsm(Individual, state)
						# ele1 = np.random.uniform(low=-SCALE1, high=SCALE1, size=1)[0]
						# ele2 = np.random.uniform(low=-SCALE2, high=SCALE2, size=1)[0]
						# delta = torch.from_numpy(np.array([ele1, 0, ele2, 0])).float().to(device)
					control_action = Individual(state+delta).cpu().data.numpy()
				else:
					control_action = Individual(state).cpu().data.numpy()
			
			control_action = np.clip(control_action, -1, 1)
			next_state, reward, done, _ = env.step(control_action)
			fuel += abs(control_action)
			state = next_state
			if ep == 99:
				trajectory.append(state)
				control_action_list.append(control_action)
			ep_r += reward
			if done:
				break
		
		ep_reward.append(ep_r)
		if t == 199:
			fuel_list.append(fuel)
			safe.append(state_list[ep])
		else:
			print(ep, state_list[ep])
			unsafe.append(state_list[ep])
	safe = np.array(safe)
	unsafe = np.array(unsafe)
	np.save('./plot/'+mode+'_safe.npy', safe)
	np.save('./plot/'+mode+'_unsafe.npy', unsafe)
	return ep_reward, np.array(fuel_list), state_list, np.array(control_action_list)


def collect_data():
	env = ContinuousCartPoleEnv()
	EP_NUM = 1000
	data_set = []
	for ep in range(EP_NUM):
		ep_loss = 0
		state = env.reset()
		for t in range(200):
			state = torch.from_numpy(state).float().to(device)
			action = ppo.choose_action(state.cpu().data.numpy(), True)
			with torch.no_grad():
				ca1 = model_1(state)
				ca2 = model_2(state)
			control_action = ca1*action[0] + ca2*action[1]
			control_action = np.clip(control_action.cpu().data.numpy(), -1, 1)
			next_state, reward, done, _ = env.step(control_action)
			data_set.append([state.cpu().data.numpy()[0], state.cpu().data.numpy()[1], state.cpu().data.numpy()[2],
				state.cpu().data.numpy()[3], control_action[0]])
			state = next_state
			if done:
				break
		print(t)
	return np.array(data_set)

if __name__ == '__main__':
	# dataset = collect_data()
	# np.save('dataset.npy', dataset)
	# assert False

	# train_switcher_DDQN()
	# assert False

	state_list = np.load('init_state.npy')
	# switching
	_, sw_fuel, _, _  = test('./switcher.pth', state_list=state_list, renew=False, mode='switch')

	# direct distillation lipschitz constant 126.1		
	_, indi_fuel, _, indi_action = test(None, state_list=state_list, renew=False, mode='individual', INDI_NAME='./direct_distill.pth')

	# robust distillation lipschitz constant 72.5
	_, robust_fuel, _, robust_action = test(None, state_list=state_list, renew=False, mode='individual', INDI_NAME='./robust_distill.pth')	
	
	# hierarchical control by learned adapter
	_, ppo_fuel, _, ppo_action = test(None, state_list=state_list, renew=False, mode='ppo')
	_, d1_fuel, _, _  = test(None, state_list=state_list, renew=False, mode='d1')
	_, d2_fuel, _, _  = test(None, state_list=state_list, renew=False, mode='d2')
	print(np.mean(ppo_fuel), np.mean(indi_fuel), np.mean(robust_fuel), np.mean(d1_fuel), np.mean(d2_fuel), 
		 len(ppo_fuel), len(indi_fuel), len(robust_fuel), len(d1_fuel), len(d2_fuel))
	
	
	plt.figure()
	plt.figure(figsize=(6, 4))
	plt.plot(indi_action, label='regular_distill')
	plt.plot(robust_action, label='robust_distill')
	plt.legend()
	plt.savefig('./plot/OS_attack_u.pdf', bbox='tight')