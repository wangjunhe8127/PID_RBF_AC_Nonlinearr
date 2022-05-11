import sys
import gym
import pandas as pd
from sympy import *
import math
import numpy as np
from .utility import PID, System
import os
os.path.join('~/PID_RBF_AC/GymEnv/PARMAS.py')
os.path.join('~/PID_RBF_AC/GymEnv/PARMAS_SN.py')
from GymEnv.PARAMS import AC_PARAMS
from GymEnv.PARAMS_SN import TARGET
gym.logger.set_level(40)
class Env(gym.Env):
    # 初始化参数
    def __init__(self):
        self.System = System()
        self.PID = PID()
        self.TH1 = 0.8
        self.TH2 = 0.5
        self.TH3 = 0.15
        self.TH4 = 0.05
        self.TH5 = 0.01
        self.is_sin = AC_PARAMS()['is_sin']
        self.current_value = 0.0 # 系统当前输出
        self.done = False
        self.cloud = 0
        self.num = 0
        self.now_max = 0
        self.last_max = 0
        self.mul_num = AC_PARAMS()['mul_num']
        self.up_num = AC_PARAMS()['up_num']
        self.value_limit = AC_PARAMS()['is_value_limit']
        self.is_increase_PID =  AC_PARAMS()['is_increase_PID']
        self.observation_space = gym.spaces.Box(low=-20, high=20, shape=(3,))
        self.action_space = gym.spaces.Box(low=-5, high=5, shape=(3,))
    # 奖励函数
    def get_reward(self):
        if abs(self.PID.err)>self.TH1:
            r1 = 1
        elif abs(self.PID.err)>self.TH2:
            r1 = 3
        elif abs(self.PID.err)>self.TH3:
            r1 = 5
        elif abs(self.PID.err)>self.TH4:
            r1 = 9
        elif abs(self.PID.err)>self.TH5:
            r1 = 12
        else:
            r1 = 15
        r2 = 10
        if abs(self.PID.err) > abs(self.PID.real_past):
            if abs(self.PID.err_err)>abs(self.PID.real_past_past):
                r2 = -10
        reward = 0.0003*r1+0.0002*r2
        self.cloud = self.cloud+1 if abs(self.PID.err)<self.TH5 else 0
        if self.could == 3:
            reward = (self.up_num - self.num)*0.008
            self.done = True
        if self.value_limit:
            if self.up_num == 400:
                done_num = 10*math.exp(-0.02230489*self.num+1.3321368e-5*self.num*self.num)
            elif self.up_num == 300:
                done_num = 10*math.exp(-0.024168*self.num+5.996152e-6*self.num*self.num)
            elif self.up_num == 250:
                done_num = 10*math.exp(-0.02539448*self.num-6.786689e-6*self.num*self.num)
            elif self.up_num == 100:
                done_num = 10*math.exp(-0.12236*self.num+0.000542207*self.num*self.num)
            if self.num>self.up_num or abs(self.PID.err)>done_num:
                reward = -0.8
                self.done = True
        else:
            if self.num>self.up_num:
                reward = -0.8
                self.done = True
        return reward
    def get_state(self,observation):
        state = np.array(observation).reshape(1,3)
        state = state*self.mul_num
        return state
    # 主程序
    def step(self, action,is_test=False,random_pid = None):
        self.num = self.num+1
        # 执行动作
        action = np.array([action]).tolist()
        PID_value, observation = self.PID.compute_out(self.current_value,action,is_test,random_pid)
        self.current_value = self.System.sim_step(PID_value)
        # 获取动作对应奖励
        reward = self.get_reward()
        # 获取下一状态
        state = self.get_state(observation)
        return state, reward, self.done, {}
    # 重置环境
    def reset(self):
        self.num = 0
        self.could = 0
        self.now_max = 0
        self.last_max = 0
        self.done = False
        self.current_value = self.System.reset()
        mode = 'increase' if self.is_increase_PID else 'position'
        observation = self.PID.reset(is_sin=self.is_sin,control_mode=mode)
        state = self.get_state(observation)
        return state
# if __name__ == '__main__':