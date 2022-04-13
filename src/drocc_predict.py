import torch
import datetime
import os
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from nn.network import DNN
from dataset import ConnlogsDataset
import numpy as np

def run_test(dataset_dir):
    # Create train data loader
    test_dataset = ConnlogsDataset(dataset_dir, nn_type='nn_oneclass_period', tvt_type="test", timestamp=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128,
                                              shuffle=True)
    # Load nn
    nn = DNN(test_dataset.feat_num, num_hidden_nodes=128)
    nn.load_state_dict(torch.load("../out/models/drocc.pt"))

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
    # Evaluation based on https://openreview.net/forum?id=BJJLHbb0-
    thresh = np.percentile(b, 20)
    y_pred = np.where(b >= thresh, 1, 0)
    prec, recall, f1_score, _ = precision_recall_fscore_support(
        a, y_pred, average="binary")

    return ts, scores, labels, auc_score, prec, recall, f1_score

# Run test
starttime = datetime.datetime.now()
ts, predict, gt, auc_score, prec, recall, f1_score = run_test("../out/dataset")
endtime = datetime.datetime.now()
print("Test time: " + str(endtime - starttime))
print("Test AUC Score: " + str(auc_score))
print("Test F1 Score: " + str(f1_score))
print("Test Precision: " + str(prec))
print("Test Recall: " + str(recall))

with open(os.path.join("../out/test_result", "test_predict.csv"), 'w') as f_csv:
    f_csv.write("timestamp,real,predict")
    for i in range(len(ts)):
        f_csv.write(str(ts[i]) + ',' + str(gt[i]) + ',' + str(predict[i]) + '\n')
