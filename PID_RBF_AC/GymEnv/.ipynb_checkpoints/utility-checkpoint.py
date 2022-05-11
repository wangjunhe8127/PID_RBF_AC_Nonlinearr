import math
import numpy
# 控制系统PID
class PID:
    def __init__(self):
        self.hope_value = 0.0
        self.control_mode = 0.0
        self.err = 0.0
        self.err_err = 0.0
        self.err_past = 0.0
        self.err_err_err = 0.0
        self.err_err_past = 0.0
        self.err_sum = 0.0
        self.uCurent = 0.0
        self.uPrevious = 0.0
        self.real_past = 0.0
    def Position_mode(self, current_value, kp, ki, kd):
        self.err = self.hope_value - current_value
        self.real_past = self.err_past
        self.err_err = self.err - self.err_past
        self.err_past = self.err
        self.real_past_past = self.err_err_past
        self.err_err_err = self.err_err - self.err_err_past
        self.err_err_past = self.err_err
        self.err_sum += self.err
        outPID = kp* self.err+ ki* self.err_sum+ kd* self.err_err
        return outPID

    def Increase_mode(self, curValue, kp, ki, kd):
        self.uCurent = self.Position_mode(curValue, kp, ki, kd)
        outPID = self.uCurent - self.uPrevious
        self.uPrevious = self.uCurent
        return outPID

    def compute_out(self, curValue, action,is_test,random_pid):
        if not is_test:
            kp = (action[0][0]+5)*0.02
            ki = (action[0][1]+5)*0.04
            kd = (action[0][2]+5)*0.01
        else:
            kp = random_pid[0]
            ki = random_pid[1]
            kd = random_pid[2]
        self.hope_value = numpy.sign(math.sin(self.num*0.001*2*math.pi)) if self.is_sin else 1
        self.num+=1
        if self.control_mode == 'position':
            res = self.Position_mode(curValue, kp, ki, kd),[self.err, self.err_err, self.err_err_err]
        else:
            res = self.Increase_mode(curValue, kp, ki, kd),[self.err, self.err_err, self.err_err_err]
        res = list(res)
        if self.num%150 == 0:
            res[0] = res[0] + numpy.random.uniform(low=-0.1, high=0.1)
        return res

    def reset(self,is_sin, control_mode):
        self.num = 0
        self.is_sin = is_sin
        self.hope_value = 0 if is_sin else 1
        self.control_mode = control_mode
        self.err = 0.0
        self.err_err = 0.0
        self.err_past = 0.0
        self.err_err_err = 0.0
        self.err_err_past = 0.0
        self.err_sum = 0.0
        self.uCurent = 0.0
        self.uPrevious = 0.0
        self.real_past = 0.0
        return [self.err, self.err_err, self.err_err_err]

# 被控对象系统，传递函数离散化
class System:
    def __init__(self):
        self.In_past = 0
        self.OUT_past = 0
    def sim_step(self, In):
        output = (self.In_past-0.1*self.OUT_past)/(1+self.OUT_past*self.OUT_past)
        self.In_past = In
        self.OUT_past = output
        return output
    def reset(self):
        self.In_past = 0
        self.OUT_past = 0
        return 0.0