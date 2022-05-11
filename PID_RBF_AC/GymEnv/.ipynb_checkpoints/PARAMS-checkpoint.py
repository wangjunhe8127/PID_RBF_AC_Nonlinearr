from .PARAMS_SN import TARGET
import torch
def AC_PARAMS(num=TARGET):
    res={
    #测试时需要把is_value_limit设置为False
    #测试时需要把up_num设置为4000
    #测试时需要把is_sin设置为True
    #训练时记得再改回来
    1:{
        'log_std_init':      2,#actor初始高斯分布log标准差
        'hidden_num':        13,#RBF个数
        'is_value_limit':    True,#当误差大于阈值后退出当前回合
        'up_num':            400,#单个回合最大步数
        'mul_num':           5,#状态乘以系数
        'rbf_init':          10,#rbf方差初始值
        'is_sin':            False,#是否测试方波
        'is_increase_PID':   False,#是否为增量式PID
        'gamma':             0.99,#强化学习折扣率
        'lr':                0.0003,#学习率
    },
    }
    return res.get(num,None)
