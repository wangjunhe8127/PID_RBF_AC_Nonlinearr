from gym.envs.registration import register

register(
    id = 'PID_RBF_AC-v0', # 环境名,版本号v0必须有
    entry_point = 'GymEnv.env:Env'
    # entry_point='env:RobotEnv'#换成自己代码后，将上句代码注释掉，然后使用这一句
)