import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import roc_auc_score, f1_score

try:
    from termcolor import cprint
except ImportError:
    cprint = None

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

'''
Use the average meter to keep track of average of the loss or 
the test accuracy! Just call the update function, providing the
quantities being added, and the counts being added
'''
class AveMeter():
    reals = []
    predicts = []

    def __init__(self):
        pass

    def update(self, real, predict):
        self.reals.append(real.astype(np.float))
        self.predicts.append(predict.astype(np.float))
    
    def get_auc(self):
        real_res = np.array([])
        for i in self.reals:
            real_res = np.concatenate((real_res, i))
        predicts_res = np.array([])
        for i in self.predicts:
            predicts_res = np.concatenate((predicts_res, i))
        score = roc_auc_score(real_res, predicts_res)
        return score

    def get_f1(self):
        real_res = np.array([])
        for i in self.reals:
            real_res = np.concatenate((real_res, i))
        predicts_res = np.array([])
        for i in self.predicts:
            predicts_res = np.concatenate((predicts_res, i))
        score = f1_score(real_res, predicts_res)
        return score

'''
    使用指定网路net 和数据加载器loader 运行一遍网络。如果Train==True 则训练网络，否则测试网络准确性
'''
def run(net, epoch, loader, optimizer, criterion, run_type="Train"):
    # Initialize
    test_accuracy = None
    if run_type is not "Train":
        test_accuracy = AveMeter()

    ave_loss = torch.zeros(1)
    batch_count = 0

    # Run one epoch
    for i, (datapoint, labels) in enumerate(loader):
        outputs = net(datapoint.float())        # 前向传播，获取网络输出
        labels = labels.type(torch.LongTensor)      # 获取数据label
        loss = criterion(outputs, labels)       # 使用 损失函数 计算label 和网络输出的差
        ave_loss += loss
        batch_count += 1

        if run_type is "Train":   # 如果是训练模式，则对网络计算后向传播并使用optimizer优化网络参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:       # 如果是测试模式，则记录 accuracy
            out_predict = torch.argmax(outputs, dim=1)
            test_accuracy.update(labels.numpy(), out_predict.numpy())

    ave_loss /= batch_count

    if run_type is "Train":
        return ave_loss, None
    elif run_type is "Validate":
        return ave_loss, test_accuracy.get_f1()
    return ave_loss, test_accuracy.get_auc()

        
'''
    使用指定 训练数据集train_loader 和 测试数据集 test_loader 训练网络net
    训练过程的参数由cfg传入
'''
def train(net, train_loader, test_loader, cfg):
    # Define the SGD optimizer here. Use hyper-parameters from cfg
    # 优化器Optimizer, 使用SGD优化方法
    optimizer = optim.SGD(net.parameters(), lr=cfg["lr"], nesterov=cfg["nesterov"],
                                momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    # Define the criterion (Objective Function) that you will be using
    criterion = nn.NLLLoss()        # 损失函数 Lossfunction
    # Define the ReduceLROnPlateau scheduler for annealing the learning rate
    # 调度器scheduler，在学习效果不好时降低learning rate以提高准确度
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min",  patience=cfg["patience"], factor=cfg["lr_decay"])

    weight1 = []
    weight2 = []

    # 训练一共进行 cfg['epochs'] 次
    for i in range(cfg['epochs']):
        # Run the network on the entire train dataset. Return the average train loss
        # Note that we don't have to calculate the accuracy on the train set.
        loss, _ = run(net, i, train_loader, optimizer, criterion)
        
        # Logs the training loss on the screen, while training
        if i % cfg['log_every'] == 0:
            log_text = "Epoch: [%d/%d], Training Loss:%2f" % (i, cfg['epochs'], loss)
            log_print(log_text, color='green', attrs=['bold'])

        # Evaluate our model and add visualizations on tensorboard
        if i % cfg['val_every'] == 0:
            # HINT - you might need to perform some step before and after running the network
            # on the test set
            # Run the network on the test set, and get the loss and accuracy on the test set 
            loss, acc = run(net, i, test_loader, optimizer, criterion, run_type="Validate")
            log_text = "Epoch: %d, Test Accuracy:%2f" % (i, acc)
            log_print(log_text, color='red', attrs=['bold'])

            # Perform a step on the scheduler, while using the Accuracy on the test set
            scheduler.step(acc)

