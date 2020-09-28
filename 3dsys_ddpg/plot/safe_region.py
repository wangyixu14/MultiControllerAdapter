import matplotlib.pyplot as plt
import numpy as np

d1_safe = np.load('d1_safe.npy')
d2_safe = np.load('d2_safe.npy')
switch_safe = np.load('switch_safe.npy')
individual_safe = np.load('individual_safe.npy')

d1_unsafe = np.load('d1_unsafe.npy')
d2_unsafe = np.load('d2_unsafe.npy')
switch_unsafe = np.load('switch_unsafe.npy')
individual_unsafe = np.load('individual_unsafe.npy')

plt.subplots(2, 2, figsize=(6, 4))
plt.subplot(221)
plt.title('$\kappa_{1}$')
plt.scatter(d1_safe[:, 0], d1_safe[:, 1], c='green',s=10)
plt.scatter(d1_unsafe[:, 0], d1_unsafe[:, 1], c='red',marker='x',s=10)
plt.xticks(color='w')
# plt.yticks(color='w')

plt.subplot(222)
plt.title('$\kappa_{2}$')
plt.scatter(d2_safe[:, 0], d2_safe[:, 1], c='green',s=10)
plt.scatter(d2_unsafe[:, 0], d2_unsafe[:, 1], c='red',marker='x',s=10)
plt.xticks(color='w')
plt.yticks(color='w')

plt.subplot(223)
plt.title('$Switching$')
plt.scatter(switch_safe[:, 0], switch_safe[:, 1], c='green',s=10)
plt.scatter(switch_unsafe[:, 0], switch_unsafe[:, 1], c='red',marker='x',s=10)


plt.subplot(224)
plt.title('$\kappa^{*}$')
plt.scatter(individual_safe[:, 0], individual_safe[:, 1], c='green',s=10)
plt.scatter(individual_unsafe[:, 0], individual_unsafe[:, 1], c='red',marker='x',s=10)
# plt.xticks(color='w')
plt.yticks(color='w')

plt.savefig('OS_region.pdf', bbox_inches='tight')