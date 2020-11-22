import numpy as np


def MController(state):
	action = 0.634*state[0] - 0.296*state[1] - 0.153*state[2] + 0.053*state[0]**2 - 1.215*state[0]**3
	return action

Lips_list = []
for _ in range(5000000):

	a = np.random.uniform(low=-0.5, high=0.5, size=(3,))
	b = np.random.uniform(low=-0.5, high=0.5, size=(3,))

	Lips_list.append(abs(MController(a) - MController(b)) / np.linalg.norm(a-b))


print(max(Lips_list))