import numpy as np
import matplotlib.pyplot as plt

os_direct = np.load('os_direct.npy')
os_robust = np.load('os_robust.npy')
ex_direct = np.load('3d_direct.npy')
ex_robust = np.load('3d_robust.npy')

cartpole_direct = np.load('cartpole_direct.npy')
cartpole_robust = np.load('cartpole_robust.npy')

fig = plt.figure(figsize=(18, 4))
ax = plt.subplot("131")
ax.set_title("Van der Pol's oscillator", fontsize=16)
ax.plot(os_direct)
ax.plot(os_robust)

# plt.subplots(1, 3, figsize=(18,4))
# plt.subplot(131)
# plt.plot(os_direct)
# plt.plot(os_robust)

ax = plt.subplot(132)
ax.set_title("3D system", fontsize=16)
ax.plot(ex_direct[:100])
ax.plot(ex_robust[:100])

ax = plt.subplot(133)
ax.set_title("Cartpole", fontsize=16)
ax.plot(cartpole_direct, label='$\kappa_{D}$')
ax.plot(cartpole_robust, label='$\kappa^{*}$')
plt.legend(fontsize=15)
plt.savefig('attack_u.pdf', bbox_inches='tight')