import numpy as np
import torch
import torch.nn as nn
from Model import Actor
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

weight = float(sys.argv[2])
print(weight)

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

model_1 = Actor(state_size=2, action_size=1, seed=0, fc1_units=25).to(device)
model_1.load_state_dict(torch.load("actor_2800.pth"))
model_1.eval()

model_2 = Actor(state_size=2, action_size=1, seed=0, fc1_units=25).to(device)
model_2.load_state_dict(torch.load("actor_2900.pth"))
model_2.eval()

# single_model = Actor(state_size=2, action_size=1, seed=0, fc1_units=25).to(device)
# single_model.load_state_dict(torch.load("actor_single_2400.pth"))
# print(model_1, model_2, single_model)

# invariant_1 = io.loadmat('m1_inv.mat')['V']
# invariant_2 = io.loadmat('m2_inv.mat')['V']
# single_inv = io.loadmat('data/single.mat')['V']

# def is_invariant(state):
	 
# 	x_loc = state[0] + 150
# 	y_loc = state[1] + 40
# 	x = np.linspace(115, 185, 150)
# 	y = np.linspace(22, 58, 150)
# 	inv1 = interp2d(x, y, single_inv, kind='cubic')(x_loc, y_loc)
# 	return int(inv1<1e-6)

# def where_inv(state):
	 
# 	x_loc = state[0]
# 	y_loc = state[1]
# 	x1 = np.linspace(-2.5, 1.5, 150)
# 	y1 = np.linspace(-2.5, 2.5, 150)
# 	inv1 = interp2d(x1, y1, invariant_1, kind='cubic')(x_loc, y_loc)
	
# 	x2 = np.linspace(-1.5, 2.5, 150)
# 	y2 = np.linspace(-2.5, 2.5, 150)
# 	inv2 = interp2d(x2, y2, invariant_2, kind='cubic')(x_loc, y_loc)

# 	return np.array([int(inv1<1e-6), int(inv2<1e-6)])

# whole space
# def where_inv(state):
# 	x0 = state[0]
# 	x1 = state[1]
# 	return np.array([int(x0 in Interval(-2, 1)), int(x0 in Interval(-1, 2))]) 

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
			# state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
			q_value = self.forward(state)
			# print(q_value)
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

def train_adapter():
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
			next_state, reward, done = env.step(control_action)

			reward += 2
			# reward -= weight * (abs(next_state[0]) + abs(next_state[1]))
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
			torch.save(model.state_dict(), './ddqn_models/ddqn_'+str(ep)+'_'+str(weight)+'.pth')

def test(adapter_name=None, state_list=None, renew=False, mode='switch'):
	env = Osillator()
	model = DQN(2, 2).to(device)
	EP_NUM = 200
	if mode == 'switch':
		model.load_state_dict(torch.load(adapter_name))
	if renew:
		state_list = []
	fuel_list = []
	ep_reward = []

	for ep in range(EP_NUM):
		if renew:
			state = env.reset()
			state_list.append(state)
		else:
			assert len(state_list) == EP_NUM
			state = env.reset(state_list[ep][0], state_list[ep][1])
		ep_r = 0
		fuel = 0
		
		for t in range(env.max_iteration):
			state = torch.from_numpy(state).float().to(device)
			# flag = where_inv(state.cpu().numpy())
			if mode == 'switch':
				action = model.act(state, epsilon=0)
				with torch.no_grad():
					if action == 0:
						control_action = model_1(state).cpu().data.numpy()[0]
					elif action == 1:
						control_action = model_2(state).cpu().data.numpy()[0]
				if ep == 0:
					print(t, state, action, control_action*20)
			
			elif mode == 'random':
				action = np.random.randint(2)
				if action == 0:
					control_action = model_1(state).cpu().data.numpy()[0]
				elif action == 1:
					control_action = model_2(state).cpu().data.numpy()[0]
			
			elif mode == 'd1':
				control_action = model_1(state).cpu().data.numpy()[0]
		
			elif mode == 'd2':
				control_action = model_2(state).cpu().data.numpy()[0]

			next_state, reward, done = env.step(control_action)
			fuel += abs(control_action) * 20
			state = next_state
			ep_r += reward
			if done:
				break
	
		ep_reward.append(ep_r)
		if t >= 90:
			fuel_list.append(fuel)
	return ep_reward, np.array(fuel_list), state_list

def distill(model_name):
	env = Osillator()
	model = DQN(2, 2).to(device)
	EP_NUM = 2

	model.load_state_dict(torch.load(model_name))

	for ep in range(EP_NUM):
		state = env.reset()
		
		for  t in range(env.max_iteration):
			state = torch.from_numpy(state).float().to(device)
			action = model.act(state, epsilon=0)
			with torch.no_grad():
				if action == 0:
					control_action = model_1(state).cpu().data.numpy()[0]
				elif action == 1:
					control_action = model_2(state).cpu().data.numpy()[0]
			if ep == 0:
				print(t, state, action, control_action*20)

			next_state, reward, done = env.step(control_action)
			state = next_state
			if done:
				break

if __name__ == '__main__':

	# sw_reward, sw_fuel, sw_state  = test('./ddqn_200_1.0_good.pth', state_list=None, renew=True, mode='switch')
	# d1_reward, d1_fuel, _ = test(model_name=None, state_list=sw_state, renew=False, mode='d1')
	# ran_reward, ran_fuel, _ = test(model_name=None, state_list=sw_state, renew=False, mode='random')
	# d2_reward, d2_fuel, _ = test(model_name=None, state_list=sw_state, renew=False, mode='d2')
	# print(np.mean(sw_fuel), np.mean(d1_fuel), np.mean(d2_fuel), np.mean(ran_fuel), 
	# 	len(sw_fuel), len(d1_fuel), len(d2_fuel), len(ran_fuel))

	distill('./ddqn_200_1.0_good.pth')