import torch
import torch.nn as nn

'''
    自定义网络对象，结构为: input(customized) -> hidden(100) -> ReLU -> output(2) -> LogSoftmax
    forward() 方法实现网络的前向传播
    后向传播由网络自己计算
    save 和 load 提供了存储和读取网络参数的功能
'''
class Network(nn.Module):
    def __init__(self, input_size):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)  # 5*5 from image dimension
        # self.fc11 = nn.Dropout(p=0.2)
        self.fc2 = nn.ReLU()
        self.fc3 = nn.Linear(100, 2)
        self.fc4 = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.fc11(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x
    
    def save(self, ckpt_path):
        # with open(ckpt_path, 'w') as f:
        torch.save(self.state_dict(), ckpt_path)

    def load(self, ckpt_path):
        # with open(ckpt_path, 'w') as f:
        checkpoint = torch.load(ckpt_path)
        self.load_state_dict(checkpoint)
        self.eval()