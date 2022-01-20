import csv
import json

import torch
from nn.network import Network
from nn.data import MYDATASET
from nn.train import train
from nn.config import cfg

import numpy as np

'''
    Prerequest: Store dataset file in ../out/ directory:
        fullDataset.npy
'''
def test_predict(nn, test_data):
    # Create test data loader
    test_dataset = MYDATASET(test_data)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg["batch_size"],
                                              num_workers=cfg["num_workers"], shuffle=False)

    test_reals = np.array([])
    test_predicts = np.array([])
    for i, (datapoint, labels) in enumerate(test_loader):
        outputs = nn(datapoint)
        labels = labels.type(torch.LongTensor)
        out_predict = torch.argmax(outputs, dim=1)
        test_reals = np.concatenate((test_reals, labels.numpy()))
        test_predicts = np.concatenate((test_predicts, out_predict.numpy()))

    return test_predicts, test_reals

if __name__ == "__main__":
    np.random.seed(52)

    # Load dataset
    test_data = np.load("../out/test_dataset.npy")
    id_record_dict = {}
    with open("../out/id_feature.txt", 'r') as f:
        id_record_dict = json.load(f)

    feature_num = test_data.shape[1] - 2

    nn = Network(feature_num)
    nn.load("../out/network.pt")

    # Predict output
    predict, real = test_predict(nn, test_data)

    # Output predict result on test_dataset
    # Output as .csv file

    with open("../out/test_predict.csv", 'w') as f:
        csv_out = csv.writer(f)
        csv_out.writerow(['uid', 'real', 'predict'])
        for i, (r, p) in enumerate(zip(real, predict)):
            id = int(test_data[i, 1])
            uid = id_record_dict[str(id)]
            csv_out.writerow([uid, r, p])
        

