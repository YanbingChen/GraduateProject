'''
    神经网络参数
'''

cfg = {
    'lr': 0.01,         # 学习率 learning rate
    'weight_decay': 0.0001,
    'momentum': 0.9,    # gradient momentum
    'nesterov': True,
    'epochs': 60,       # 训练 epoch 总数
    'batch_size': 64,   # 训练时每 batch 的大小
    'log_every': 1,
    'val_every': 1,
    'num_workers': 0,
    'patience': 5,
    'lr_decay': 0.5     # 学习率 减少的速度
}