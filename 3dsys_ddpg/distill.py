import numpy as np
import torch
from Model import  IndividualModel
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn as nn

dataset = np.load('dataset.npy')
y = torch.from_numpy(np.reshape(dataset[:, -1], (len(dataset[:, -1]), 1))).float()
x = torch.from_numpy(dataset[:, :3]).float()

Individual = IndividualModel(state_size=3, action_size=1, seed=0)

def train(inputdata, label, net):
	optimizer = torch.optim.Adam(net.parameters(), weight_decay=1e-5)
	criterion = torch.nn.MSELoss()  

	BATCH_SIZE = 100
	EPOCH = 100

	torch_dataset = Data.TensorDataset(inputdata, label)

	loader = Data.DataLoader(
		dataset=torch_dataset, 
		batch_size=BATCH_SIZE, 
		shuffle=True, num_workers=2,)

	for epoch in range(EPOCH):
		loss_list = []
		for step, (batch_x, batch_y) in enumerate(loader, 0): 
			prediction = net(batch_x)    
			loss = criterion(prediction, batch_y)     
			loss_list.append(loss.data.numpy())
			optimizer.zero_grad()   
			loss.backward()         
			optimizer.step()       
		print(np.sum(loss_list), len(loss_list))
	torch.save(net.state_dict(), './l2_only.pth')

def fgsm(model, X, y, epsilon=0.04):
    delta = torch.zeros_like(X, requires_grad=True)
    loss = -nn.MSELoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()

def robust_train(inputdata, label, net):
	optimizer = torch.optim.Adam(net.parameters(), weight_decay=1e-5)
	criterion = torch.nn.MSELoss()  

	BATCH_SIZE = 100
	EPOCH = 100

	torch_dataset = Data.TensorDataset(inputdata, label)

	loader = Data.DataLoader(
		dataset=torch_dataset, 
		batch_size=BATCH_SIZE, 
		shuffle=True, num_workers=2,)

	for epoch in range(EPOCH):
		loss_list = []
		for step, (batch_x, batch_y) in enumerate(loader, 0):
			if np.random.uniform(low=0, high=1, size=1)[0] > 0.8:
				delta = fgsm(net, batch_x, batch_y)
				prediction = net(batch_x+delta)
			else:
				prediction = net(batch_x)    
			loss = criterion(prediction, batch_y)     
			loss_list.append(loss.data.numpy())
			optimizer.zero_grad()   
			loss.backward()         
			optimizer.step()       
		print(np.sum(loss_list), len(loss_list))
	torch.save(net.state_dict(), './robust_distill_0824.pth')

# train(x, y, Individual)
robust_train(x, y, Individual)