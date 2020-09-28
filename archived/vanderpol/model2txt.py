# this file is to record the NN controller parameters into a txt file to be used 
# for Bernstein polynomial approximation by the tool of ReachNN
from Model import IndividualModel
import torch
import numpy as np

trained_model = IndividualModel(state_size=2, action_size=1, seed=0, fc1_units=50)
trained_model.load_state_dict(torch.load('./models/Individual.pth'))
trained_model.eval()
bias_list = []
weight_list = []
for name, param in trained_model.named_parameters():
	if 'bias' in name:
		bias_list.append(param.detach().cpu().numpy())
		
	if 'weight' in name:
		weight_list.append(param.detach().cpu().numpy())

all_param = []

for i in range(len(bias_list)):
	for j in range(len(bias_list[i])):
		for k in range(weight_list[i].shape[1]):
			all_param.append(weight_list[i][j, k])
		all_param.append(bias_list[i][j])

np.savetxt('nn_individual_relu_tanh', np.array(all_param))
print('done')