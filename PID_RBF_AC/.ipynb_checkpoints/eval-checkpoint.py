import gym
import GymEnv
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import os
import numpy as np
import torch as th
import random
os.path.join('~/PID_RBF_AC/GymEnv/PARMAS.py')
os.path.join('~/PID_RBF_AC/GymEnv/PARMAS_SN.py')
from GymEnv.PARAMS_SN import TARGET
from GymEnv.PARAMS import AC_PARAMS
env = gym.make('PID_RBF_AC-v0')
model = A2C.load('%s.pkl'%repr(TARGET))# 导入模型
score = 0
steps = 0
def fixed_seed(i):
    random.seed(i)
    os.environ['PYTHONHASHSEED'] = str(i)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(i)
    th.manual_seed(i)
    th.cuda.manual_seed(i)
    th.cuda.manual_seed_all(i)  # if you are using multi-GPU.
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
print('开始测试')
fixed_seed(1)#固定随机种子
state = env.reset()
done = False
while not done:
    action,_ = model.predict(state,deterministic=True)
    action = action.squeeze(0)
    kp = (action[0]+5)*0.02
    ki = (action[0]+5)*0.04
    kd = (action[0]+5)*0.01
    with open('PID.txt', 'a+') as f:
        f.write(str(kp)+' '+str(ki)+' '+str(kd))
        f.write('\r\n')
    state,reward,done,_ = env.step(action)
    score = score+reward
    steps =steps+1
    with open('model_value.txt', 'a+') as f:
        f.write(str(env.current_value))
        f.write('\r\n')
    with open('hope_value.txt', 'a+') as f:
        f.write(str(env.PID.hope_value))
        f.write('\r\n')
env.close()
avg_score = score
avg_steps = steps
print("avg_score=",avg_score)
print("avg_steps=",avg_steps)
print('开始生成随机结果')
fixed_seed(1)#固定随机种子
state = env.reset()
done = False
while not done:
    state,reward,done,_ = env.step(action = None,is_test = True,random_pid = [0.03,0.01,0.03])
    score = score+reward
    steps =steps+1
    with open('random_value.txt', 'a+') as f:
        f.write(str(env.current_value))
        f.write('\r\n')
env.close()
print('测试结束')