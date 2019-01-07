# Inspired by https://keon.io/deep-q-learning/

import random
import gym
import math
import numpy as np
from collections import deque
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from gym.envs.classic_control import *

from cartpole_env import *


kp_cart = 2
kd_cart = 50
kp_pole = 8
kd_pole = 100
DIRECT_MAG=True
RANDOM_NOISE=True

if DIRECT_MAG:
    env=CartPoleEnv()
else:
    env = gym.make('CartPole-v1')

class CartPoleControl:

    def __init__(self, kp_cart, kd_cart, kp_pole, kd_pole):
        self.kp_cart = kp_cart
        self.kd_cart = kd_cart
        self.kp_pole = kp_pole
        self.kd_pole = kd_pole
        self.bias_cart_1 = 0
        self.bias_pole_1 = 0
        self.i=0

    def pid_cart(self, position):
        bias = position  # 这句可能有问题
        #bias=self.bias_cart_1*0.8+bias*0.2
        d_bias = bias - self.bias_cart_1
        balance = self.kp_cart * bias + self.kd_cart * d_bias
        self.bias_cart_1 = bias
        return balance

    def pid_pole(self, angle):
        bias = angle  # 这句可能有问题
        d_bias = bias - self.bias_pole_1
        balance = -self.kp_pole * bias - self.kd_pole * d_bias
        self.bias_pole_1 = bias
        return balance

    def control_output(self, control_cart, control_pole):
        if DIRECT_MAG:
            return -10*(control_pole - control_cart)
        else:
            return 1 if (control_pole - control_cart) < 0 else 0

if __name__ == '__main__':

    control=CartPoleControl(kp_cart, kd_cart, kp_pole, kd_pole)


    rewards=0
    state = env.reset()
    done = False
    i=0
    while abs(state[2]<2):
        env.render()
        control_pole = control.pid_pole(state[2])
        control_cart = control.pid_cart(state[0])
        if RANDOM_NOISE and random.random()>0.99:
            i=2

        if i>0:
            if DIRECT_MAG:
                action = 10
            else:
                action=1
            i-=1
        else:
            action = control.control_output(control_cart, control_pole)

        next_state, reward, done, _ = env.step(action)
        state = next_state
        rewards+=reward
        print(state)
        print(action)
    print('total rewards:'+str(rewards))
    env.close()


