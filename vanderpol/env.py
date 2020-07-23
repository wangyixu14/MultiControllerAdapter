import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import math
import random
import math
import gym
from interval import Interval

def main():
    env = Osillator_square()
    for ep in range(1):
        state = env.reset()

        for i in range(env.max_iteration):
            action = np.random.uniform(low=-1, high=1)
            next_state, reward, done = env.step(action)
            print(i, state,action, next_state, reward, done)
            state = next_state
            if done:
                break

class Osillator:
    deltaT = 0.05
    u_range = 20
    max_iteration = 100
    error = 1e-5
    x0_low = -2
    x0_high = 2  
    x1_low = -2 
    x1_high = 2
    def __init__(self, x0=None, x1=None):
        if x0 is None or x1 is None:
            x0 = np.random.uniform(low=self.x0_low, high=self.x0_high, size=1)[0]
            x1 = np.random.uniform(low=self.x1_low, high=self.x1_high, size=1)[0]
            self.x0 = x0
            self.x1 = x1
        else:
            self.x0 = x0
            self.x1 = x1
        
        self.t = 0
        self.state = np.array([self.x0, self.x1])
        self.u_last = 0

    def reset(self, x0=None, x1=None):
        if x0 is None or x1 is None:
            x0 = np.random.uniform(low=self.x0_low, high=self.x0_high, size=1)[0]
            x1 = np.random.uniform(low=self.x1_low, high=self.x1_high, size=1)[0]
            self.x0 = x0
            self.x1 = x1
        else:
            self.x0 = x0
            self.x1 = x1
        
        self.t = 0
        self.state = np.array([self.x0, self.x1])
        self.u_last = 0
        return self.state

    def step(self, action):
        # np.random.seed(1)
        disturbance = np.random.uniform(-0.05, 0.05)
        # disturbance = 0
        u = action * self.u_range
        x0_tmp = self.state[0] + self.deltaT * self.state[1]
        x1_tmp = self.state[1] + self.deltaT*((1-self.state[0]**2)*self.state[1] - self.state[0] + u) + disturbance
        
        self.t = self.t + 1
        reward = self.design_reward(u, self.u_last, smoothness=0.2)
        self.u_last = u
        self.state = np.array([x0_tmp, x1_tmp])
        done = self.if_unsafe() or self.t == self.max_iteration
        return self.state, reward, done

    def design_reward(self, u, u_last, smoothness):
        r = 0
        # tarining actor2800 and actor2900
        # actor
        r -= 5 * abs(self.state[0])
        r -= 5 * abs(self.state[1])
        r -= 0.2 * abs(u)
        r -= smoothness * abs(u - u_last)
        if self.if_unsafe():
            r -= 50
        else:
            r += 10        
        return r

    def if_unsafe(self):
        if self.state[0] in Interval(self.x0_low, self.x0_high) and self.state[1] in Interval(self.x1_low, self.x1_high):
            return 0
        else:
            return 1


if __name__ == '__main__':
    main()
