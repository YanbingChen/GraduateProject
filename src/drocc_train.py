from __future__ import print_function

import datetime
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
import numpy as np
from DROCC.drocc_trainer import DROCCTrainer
from dataset import ConnlogsDataset
from nn.network import DNN

# Main method for Anomaly-Detection-module

def adjust_learning_rate(epoch, total_epochs, only_ce_epochs, learning_rate, optimizer):
    """Adjust learning rate during training.

    Parameters
    ----------
    epoch: Current training epoch.
    total_epochs: Total number of epochs for training.
    only_ce_epochs: Number of epochs for initial pretraining.
    learning_rate: Initial learning rate for training.
    """
    # We dont want to consider the only ce
    # based epochs for the lr scheduler
    epoch = epoch - only_ce_epochs
    drocc_epochs = total_epochs - only_ce_epochs
    # lr = learning_rate
    if epoch <= drocc_epochs:
        lr = learning_rate * 0.001
    if epoch <= 0.90 * drocc_epochs:
        lr = learning_rate * 0.01
    if epoch <= 0.60 * drocc_epochs:
        lr = learning_rate * 0.1
    if epoch <= 0.30 * drocc_epochs:
        lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return torch.from_numpy(self.data[idx]), (self.labels[idx]), torch.tensor([0])


def load_data(path):
    train_data = ConnlogsDataset(path, "nn_oneclass_period", "train")
    validate_data = ConnlogsDataset(path, "nn_oneclass_period", "validate")
    return train_data, validate_data, train_data.feat_num


def main():
    train_dataset, test_dataset, num_features = load_data(args.data_path)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=True)

    model = DNN(num_features, num_hidden_nodes=args.hd).to(device)
    if args.optim == 1:
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=args.mom)
        print("using SGD")
    else:
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr)
        print("using Adam")

    trainer = DROCCTrainer(model, optimizer, args.lamda, args.radius, args.gamma, device)

    if args.eval == 0:
        # Training the model
        trainer.train(train_loader, test_loader, args.lr, adjust_learning_rate, args.epochs,
                      metric=args.metric, ascent_step_size=args.ascent_step_size, only_ce_epochs=args.only_ce_epochs)

        trainer.save(args.model_dir)

    else:
        # Evaluating the existing model (this F1 is aiming at Normal data)
        if os.path.exists(os.path.join(args.model_dir)):
            trainer.load(args.model_dir)
            print("Saved Model Loaded")
        else:
            print('Saved model not found. Cannot run evaluation.')
            exit()
        score = trainer.test(test_loader, 'F1')
        print('Test F1: {}'.format(score))

    return model


'''
    Prerequest: Run ConnlogsDataset.process_connlogs
'''


def run_test(nn, dataset_dir):
    # Create train data loader
    test_dataset = ConnlogsDataset(dataset_dir, nn_type='nn_oneclass_period', tvt_type="test", timestamp=True)
    # shuffle should equal to false if the nn is memory, but nevermind now
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=True)

    nn.eval()
    label_score = []
    batch_idx = -1
    for data, target, ts in test_loader:
        batch_idx += 1
        data = data.to(torch.float)
        target = target.to(torch.float)
        target = torch.squeeze(target)

        logits = nn(data)
        logits = torch.squeeze(logits, dim=1)
        sigmoid_logits = torch.sigmoid(logits)
        scores = sigmoid_logits
        label_score += list(zip(target.cpu().data.numpy().tolist(),
                                scores.cpu().data.numpy().tolist(),
                                ts))

    labels, scores, ts = zip(*label_score)
    labels = np.array(labels)
    scores = np.array(scores)
    ts = np.array(ts)

    # Calculate AUC score
    a = 1 - labels
    b = 1 - scores
    auc_score = roc_auc_score(a, b)
    # Evaluation based on https://openreview.net/forum?id=BJJLHbb0-   --referred from DROCC (bullshit but beautiful)
    thresh = np.percentile(b, 20)
    y_pred = np.where(b >= thresh, 1, 0)
    prec, recall, f1_score, _ = precision_recall_fscore_support(
        a, y_pred, average="binary")

    return ts, scores, labels, auc_score, prec, recall, f1_score


if __name__ == '__main__':
    torch.set_printoptions(precision=5)

    parser = argparse.ArgumentParser(description='PyTorch Simple Training')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size for training')
    # may be shorter if loss is too small
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('-oce,', '--only_ce_epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train with only CE loss')
    parser.add_argument('--ascent_num_steps', type=int, default=50, metavar='N',
                        help='Number of gradient ascent steps')
    parser.add_argument('--hd', type=int, default=128, metavar='N',
                        help='Number of hidden nodes for LSTM model')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--ascent_step_size', type=float, default=0.001, metavar='LR',
                        help='step size of gradient ascent')
    parser.add_argument('--mom', type=float, default=0.99, metavar='M',
                        help='momentum')
    parser.add_argument('--model_dir', default='../out/models/drocc.pt',
                        help='path where to save checkpoint')
    parser.add_argument('--one_class_adv', type=int, default=1, metavar='N',
                        help='adv loss to be used or not, 1:use 0:not use(only CE)')
    # sqrt(dimension)/2 may be good
    parser.add_argument('--radius', type=float, default=2.40, metavar='N',
                        help='radius corresponding to the definition of set N_i(r)')
    parser.add_argument('--lamda', type=float, default=1, metavar='N',
                        help='Weight to the adversarial loss')
    parser.add_argument('--reg', type=float, default=0, metavar='N',
                        help='weight reg')
    parser.add_argument('--eval', type=int, default=0, metavar='N',
                        help='whether to load a saved model and evaluate (0/1)')
    parser.add_argument('--optim', type=int, default=0, metavar='N',
                        help='0 : Adam 1: SGD')
    parser.add_argument('--gamma', type=float, default=2.0, metavar='N',
                        help='r to gamma * r projection for the set N_i(r)')
    parser.add_argument('-d', '--data_path', type=str, default='../out/dataset')
    parser.add_argument('--metric', type=str, default='F1')
    parser.add_argument('--result_path', default="../out/test_result", type=str)
    args = parser.parse_args()

    # settings
    # Checkpoint store path
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    starttime = datetime.datetime.now()
    model = main()
    endtime = datetime.datetime.now()
    print("Train time: " + str(endtime - starttime))
    torch.save(model.state_dict(), "../out/models/drocc.pt")

    # Run test
    starttime = datetime.datetime.now()
    ts, predict, gt, auc_score, prec, recall, f1_score = run_test(model, args.data_path)
    endtime = datetime.datetime.now()
    print("Test time: " + str(endtime - starttime))
    print("Test AUC Score: " + str(auc_score))
    print("Test F1 Score: " + str(f1_score))
    print("Test Precision: " + str(prec))
    print("Test Recall: " + str(recall))

    with open(os.path.join(args.result_path, "drocc.csv"), 'w') as f_csv:
        f_csv.write("timestamp,real,predict")
        for i in range(len(ts)):
            f_csv.write(str(ts[i]) + ',' + str(predict[i]) + ',' + str(gt[i]) + '\n')

