from collections import OrderedDict

import torch
import torch.nn as nn

'''
    自定义网络对象，结构为: input(customized) -> hidden(100) -> ReLU -> output(2) -> LogSoftmax
    forward() 方法实现网络的前向传播
    后向传播由网络自己计算
    save 和 load 提供了存储和读取网络参数的功能
'''
class NN(nn.Module):
    def __init__(self, input_size):
        super(NN, self).__init__()
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


'''
    自定义网络对象，结构为: input(customized) -> hidden(100) -> ReLU -> output(2) -> LogSoftmax
    forward() 方法实现网络的前向传播
    后向传播由网络自己计算
    save 和 load 提供了存储和读取网络参数的功能
'''


class DNN(nn.Module):
    """
        Multi-layer perceptron with single hidden layer.
        """

    def __init__(self,
                 input_dim=2,
                 num_classes=1,
                 num_hidden_nodes=20):
        super(DNN, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_hidden_nodes = num_hidden_nodes
        activ = nn.ReLU(True)
        self.feature_extractor = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(self.input_dim, self.num_hidden_nodes)),
            ('relu1', activ)]))
        self.size_final = self.num_hidden_nodes

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.size_final, self.num_classes))]))

    def forward(self, input):
        features = self.feature_extractor(input)
        logits = self.classifier(features.view(-1, self.size_final))
        return logits