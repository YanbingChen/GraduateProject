import csv

import torch
from nn.network import NN
from nn.data import MYDATASET
from nn.train import train
from nn.config import cfg
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

import numpy as np

import datetime
from dataset import ConnlogsDataset

'''
    Prerequest: Run ConnlogsDataset.process_connlogs
'''
def train_network(dataset_dir):
    # Create train data loader
    train_dataset = ConnlogsDataset(dataset_dir, nn_type='nn', tvt_type="train")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg["batch_size"],
                                       num_workers=cfg["num_workers"], shuffle=True)

    # Create test data loader
    test_dataset = ConnlogsDataset(dataset_dir, nn_type='nn', tvt_type="validate")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg["batch_size"],
                                               num_workers=cfg["num_workers"], shuffle=True)

    feat_num = train_dataset.feat_num
    # Create a network object
    net = NN(feat_num)

    # Run the training!
    train(net, train_loader, test_loader, cfg)

    return net

'''
    Prerequest: Run ConnlogsDataset.process_connlogs
'''
def run_test(nn, dataset_dir):
    # Create train data loader
    test_dataset = ConnlogsDataset(dataset_dir, nn_type='nn', tvt_type="test", timestamp=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg["batch_size"],
                                              num_workers=cfg["num_workers"], shuffle=True)

    test_reals = np.array([])
    test_predicts = np.array([])
    test_timestamps = np.array([])
    for i, (datapoint, labels, ts) in enumerate(test_loader):
        outputs = nn(datapoint.float())
        labels = labels.type(torch.LongTensor)
        out_predict = torch.argmax(outputs, dim=1)
        test_reals = np.concatenate((test_reals, labels.numpy()))
        test_predicts = np.concatenate((test_predicts, out_predict.numpy()))
        test_timestamps = np.concatenate((test_timestamps, ts.numpy()))

    # Calculate AUC score
    auc_score = roc_auc_score(1 - test_reals, 1 - test_predicts)
    prec, recall, f1_score, _ = precision_recall_fscore_support(
        1 - test_reals, 1 - test_predicts, average="binary")

    return test_timestamps, test_predicts, test_reals, auc_score, prec, recall, f1_score

if __name__ == "__main__":
    np.random.seed(52)

    # Parse params
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--datasetPath', default="../out/dataset",
                        help='Input dataset files')
    parser.add_argument('--modelPath', default="../out/models/nn_model.pt",
                        help='Output network params')
    parser.add_argument('--resultPath', default="../out/test_result",
                        help='Output network params')

    args = parser.parse_args()

    # Load dataset
    dataset_dir = args.datasetPath
    model_path = args.modelPath
    result_path = args.resultPath

    starttime = datetime.datetime.now()
    network = train_network(dataset_dir)
    network.save(model_path)
    endtime = datetime.datetime.now()
    print("Train time: " + str(endtime - starttime))

    # Run test
    starttime = datetime.datetime.now()
    ts, predict, gt, auc_score, prec, recall, f1_score = run_test(network, dataset_dir)
    endtime = datetime.datetime.now()
    print("Test time: " + str(endtime - starttime))
    print("Test AUC Score: " + str(auc_score))
    print("Test F1 Score: " + str(f1_score))
    print("Test Precision: " + str(prec))
    print("Test Recall: " + str(recall))

    with open(os.path.join(result_path, "nn.csv"), 'w') as f_csv:
        f_csv.write("timestamp,real,predict")
        for i in range(len(ts)):
            f_csv.write(str(ts[i]) + ',' + str(predict[i]) + ',' + str(gt[i]) + '\n')