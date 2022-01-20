import csv

import torch
from nn.network import Network
from nn.data import MYDATASET
from nn.train import train
from nn.config import cfg

import numpy as np

import datetime

'''
    Prerequest: Store dataset file in ../out/ directory:
        fullDataset.npy
'''
def train_network(train_dataset, test_dataset, feature_number):
    # Create a network object
    net = Network(feature_number)

    # Create train data loader
    train_dataset = MYDATASET(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg["batch_size"],
                                       num_workers=cfg["num_workers"], shuffle=True)

    # Create test data loader
    test_dataset = MYDATASET(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg["batch_size"],
                                               num_workers=cfg["num_workers"], shuffle=True)

    # Run the training!
    train(net, train_loader, test_loader, cfg)

    return net

if __name__ == "__main__":
    np.random.seed(52)

    starttime = datetime.datetime.now()
    # Load dataset
    orig_dataset = np.load("../out/fullDataset.npy")

    # Normalize dataset
    featureAve = np.average(orig_dataset, axis=0)
    featureStd = np.std(orig_dataset, axis=0)
    orig_dataset[:, 2:] -= featureAve[2:]
    orig_dataset[:, 2:] /= featureStd[2:]

    # Split dataset
    rand_index = orig_dataset.shape[0]
    rand_index = np.arange(rand_index)
    np.random.shuffle(rand_index)
    orig_dataset = orig_dataset[rand_index]

    split_point = int(orig_dataset.shape[0] / 5)
    test_dataset = orig_dataset[0:split_point]
    train_dataset = orig_dataset[split_point:]

    # Train network
    network = train_network(train_dataset, test_dataset, orig_dataset.shape[1] - 2)

    network.save("../out/network.pt")

    np.save("../out/test_dataset", test_dataset)
    np.save("../out/train_dataset", train_dataset)

    endtime = datetime.datetime.now()

    print(endtime-starttime)


