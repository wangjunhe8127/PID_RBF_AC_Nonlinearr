import gym
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import random
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import os
os.path.join('~/PID_RBF_AC/GymEnv/PARMAS.py')
os.path.join('~/PID_RBF_AC/GymEnv/PARMAS_SN.py')
from GymEnv.PARAMS_SN import TARGET
from GymEnv.PARAMS import AC_PARAMS
from GymEnv import env
class CustomMLP(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box,features_dim:int):
        super(CustomMLP, self).__init__(observation_space, features_dim)
        self.net = th.nn.Sequential(
                             th.nn.Linear(3, 17),
                             th.nn.LeakyReLU(),
                             th.nn.Linear(17,35),
                             th.nn.PReLU())
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)
def fixed_seed(i):
    random.seed(i)
    os.environ['PYTHONHASHSEED'] = str(i)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(i)
    th.manual_seed(i)
    th.cuda.manual_seed(i)
    th.cuda.manual_seed_all(i)  # if you are using multi-GPU.
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
if __name__=='__main__':
    fixed_seed(1)#固定随机种子
    params = AC_PARAMS()
    env = make_vec_env("PID_RBF_AC-v0", n_envs=40,seed = 1)
    policy_kwargs =(
    dict(
        net_arch=[dict(
        pi=[1],
        vf=[1])],
        log_std_init=params['log_std_init'],
        hidden_num=params['hidden_num'],
        RBF_init = params['rbf_init']))
    model = A2C(
        "MlpPolicy",
        env,
        gamma=0.99,
        learning_rate=params['lr'],
        n_steps=1,
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log="../tf-logs/",)
    model.learn(
        total_timesteps=2.5*10**5,
        n_eval_episodes=10,)
    
    model.save('%s.pkl'%repr(TARGET))