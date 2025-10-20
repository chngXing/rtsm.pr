from  torch import nn
import torch
import numpy as np
import pylab as pl


class SNNLayer(nn.Module):
    '''
    构建一个 SNN 层，一个 SNN 层由多个 神经元（neurons）构成；
    我们假定：
    1）每个 dendrite（树突，用于输入） 没有 branches，
    2）每个 neuron 只有一个 axon （轴突，用于输出）
    3）attention （也称为衰减因子，attention越大，衰减越小）和输入信号有关（optional）
    4）threshold 和输入信号有关（optional）
    '''
    def __init__(self, 
                 num_dendrites,
                 num_neurons,
                 p=10.,
                ):
        '''
        Args:
            num_dendrites (int): 输入维度数（树突数目）
            num_neurons (int): 输出维度数 (神经元/轴突数目)
            p (float): 极化率 polarizability （值越大，sigmoid 跳跃越陡峭）
        '''
        
        super(SNNLayer, self).__init__()
        self.num_dendrites = num_dendrites
        self.num_neurons = num_neurons
        self.inp = nn.Linear(num_dendrites, num_neurons) # dendrites,用于聚合输入信号
        self.att = nn.Linear(num_dendrites, num_neurons) # attention，用于膜电势的保持
        self.thr = nn.Linear(num_dendrites, num_neurons) # threshold，用于比较以决定是否发生放电
        self.p = p

    def forward(self, s, v=None):
        '''
        Args:
            s (Tensor): 输入信号（signals），类似于 RNN 的输入特征/状态
            v (Tensor): 膜电势 (membrance potential)，类似于 RNN 隐态
        '''
        if v is None:
            v = torch.zeros(self.num_neurons)
        s_ = self.inp(s)
        a = torch.sigmoid(self.att(s))
        v_ = a*v + s_
        th = self.thr(s)
        s = torch.sigmoid(self.p*(v_ - th))
        v = (1-s)*v_ + s*v
        return s, v
        
        
class SNN(nn.Module):
    '''
    最简单的三层 Spiking Neural Network
    '''
    def __init__(self, num_inputs, num_outputs, num_hiddens, p=10):
        '''
        Args:
            num_inputs (int): 输入维度数
            num_outputs （int）: 输出维度数
            num_hiddens (int): 隐藏层维度数
            p (float): 极化率 polarizability （值越大，sigmoid 跳跃越陡峭）
        '''
        super(SNN, self).__init__()
        self.inp = SNNLayer(num_inputs, num_hiddens, p=p)
        self.hid = SNNLayer(num_hiddens, num_hiddens, p=p)
        self.out = SNNLayer(num_hiddens, num_outputs, p=p)

    def forward(self, s, v=(None, None, None)):
        '''
        Args:
            s (Tensor): 输入信号（signals），类似于 RNN 的输入特征/状态
            v ((Tensor, Tensor, Tensor)): 三层 SNNLayer 的膜电势 (membrance potential)，类似于 RNN 隐态
        '''
            
        v1, v2, v3 = v
        s, v1 = self.inp(s, v1)
        s, v2 = self.hid(s, v2)
        s, v3 = self.out(s, v3)
        v = v1, v2, v3
        return s, v 
        

def binary(num):
    '''输入 int，返回 list of binary digits'''
    
    return list(map(float, np.binary_repr(num, width=8)))


def get_batch(batch_size):
    '''随机生成一个 batch 的训练数据'''
    
    decimals = np.random.randint(0, 2**8, batch_size)
    return torch.FloatTensor(list(map(binary, decimals)))

if __name__ == '__main__':
    models = []
    losses, accuracies, steps = [], [], []
    polarizabilities = [1, 2, 4, 8, 16, 32] 

    for p in polarizabilities:
        snn = SNN(1, 8, 16, p=p)
        optimizer = torch.optim.Adam(snn.parameters())
        loss_func = nn.MSELoss()

        max_steps = 5000
        batch_size = 100

        steps_p, losses_p, accuracies_p = [], [], []

        for step in range(1, max_steps + 1):
            x = get_batch(batch_size=batch_size)
            v_t = (None, None, None)
            for t in range(8):
                x_t = x[:, t:t+1]
                y_t, v_t = snn(x_t, v_t)
            loss = loss_func(y_t, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                acc = (((y_t.detach() > 0.5).type(torch.float)-x).abs().sum(dim=-1) == 0).type(torch.float).mean()
                steps_p.append(step)
                losses_p.append(loss.item())
                accuracies_p.append(acc)
                #print(' '*100, end='\r')
                print(f'p={p}, step {step}: loss = {loss.item():.3f}, accuracy={acc:.3f}', end='\r')
                
        steps.append(steps_p)
        losses.append(losses_p)
        accuracies.append(accuracies_p)
        models.append(snn)
    for x, y, p in zip(steps, losses, polarizabilities):
        pl.plot(x, y, label='p='+str(p))
    pl.xlabel('step')
    pl.ylabel('loss')
    pl.legend()
    pl.savefig('loss.png')
    pl.show()