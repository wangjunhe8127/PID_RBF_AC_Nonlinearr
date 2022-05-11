from torch import nn
import torch as th

class MyLayer(nn.Module):

    def __init__(self,in_channel,out_channel,rbf_init):
        super(MyLayer,self).__init__()
        self.input_size=in_channel
        self.hidden_size=out_channel
        self.w = nn.Parameter(th.Tensor(self.hidden_size, self.input_size), requires_grad=True)#这里一定不能加.to(device)会导致没有梯度
        nn.init.normal_(self.w, 0, 0.1)
        self.log_sigmas = nn.Parameter(th.Tensor(self.hidden_size), requires_grad=True)
        nn.init.constant_(self.log_sigmas, rbf_init)
    def forward(self,data):
        size = (data.size(0), self.hidden_size, self.input_size)
        input_kernel_w = th.unsqueeze(self.w,0).expand(size)
        x = th.unsqueeze(data,1).expand(size)
        distances = th.sqrt(th.sum(th.pow((x-input_kernel_w),2),-1)) / th.exp(self.log_sigmas)
        y = th.exp(-1*th.pow(distances,2))
        return y