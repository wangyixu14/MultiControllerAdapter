# This file define the Oscillator dynamics, reward function and safety property
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import math
import random
import math
import gym
from interval import Interval

def main():
    env = Newenv()
    for ep in range(2):
        state = env.reset()
        for i in range(env.max_iteration):
            action = np.random.uniform(low=-1, high=1)
            next_state, reward, done = env.step(action)
            print(i, state,action, next_state, reward, done)
            state = next_state
            if done:
                break

class Newenv:
    deltaT = 0.05
    u_range = 10
    max_iteration = 100
    error = 1e-5
    range_low = -0.5
    range_high = 0.5  
    def __init__(self, x0=None, x1=None, x2=None):
        if x0 is None or x1 is None or x2 is None:
            x0 = np.random.uniform(low=self.range_low, high=self.range_high, size=1)[0]
            x1 = np.random.uniform(low=self.range_low, high=self.range_high, size=1)[0]
            x2 = np.random.uniform(low=self.range_low, high=self.range_high, size=1)[0]
            self.x0 = x0
            self.x1 = x1
            self.x2 = x2
        else:
            self.x0 = x0
            self.x1 = x1
            self.x2 = x2
        
        self.t = 0
        self.state = np.array([self.x0, self.x1, self.x2])
        self.u_last = 0

    def reset(self, x0=None, x1=None, x2=None):
        if x0 is None or x1 is None or x2 is None:
            x0 = np.random.uniform(low=self.range_low, high=self.range_high, size=1)[0]
            x1 = np.random.uniform(low=self.range_low, high=self.range_high, size=1)[0]
            x2 = np.random.uniform(low=self.range_low, high=self.range_high, size=1)[0]
            self.x0 = x0
            self.x1 = x1
            self.x2 = x2
        else:
            self.x0 = x0
            self.x1 = x1
            self.x2 = x2
        
        self.t = 0
        self.state = np.array([self.x0, self.x1, self.x2])
        self.u_last = 0
        return self.state

    def step(self, action, smoothness):
        # disturbance = np.random.uniform(-0.05, 0.05)
        u = action * self.u_range
        u = np.clip(u, -10, 10)
        Ori = True
        if Ori:
            x0_tmp = self.state[0] + self.deltaT * (self.state[1] + 0.5*self.state[2]**2)
            x1_tmp = self.state[1] + self.deltaT * self.state[2]
            x2_tmp = self.state[2] + self.deltaT * u
        else:
            disturbance = np.random.uniform(low=-0.01, high=0.01, size=(3,))
            x0_tmp = self.state[0] + self.deltaT * (self.state[1] + 0.5*self.state[2]**2) + disturbance[0]
            x1_tmp = self.state[1] + self.deltaT * self.state[2] + disturbance[1]
            x2_tmp = self.state[2] + self.deltaT * u + disturbance[2]           
        
        self.t = self.t + 1
        reward = self.design_reward(u, self.u_last, smoothness)
        self.u_last = u
        self.state = np.array([x0_tmp, x1_tmp, x2_tmp])
        done = self.is_unsafe() or self.t == self.max_iteration
        return self.state, reward, done

    def design_reward(self, u, u_last, smoothness):
        r = 0
        r -= 1 / smoothness * abs(self.state[0])
        r -= 1 / smoothness * abs(self.state[1])
        r -= 1 / smoothness * abs(self.state[2])
        r -= smoothness * abs(u)
        r -= smoothness * abs(u - u_last)
        if self.is_unsafe():
            r -= 50
        else:
            r += 5       
        return r

    def is_unsafe(self):
        if self.state[0] in Interval(self.range_low, self.range_high) and self.state[1] in Interval(self.range_low, self.range_high) and self.state[2] in Interval(self.range_low, self.range_high):
            return 0
        else:
            return 1


if __name__ == '__main__':
    main()
