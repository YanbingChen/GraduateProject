import os
import random

from torch.utils.data import Dataset
import torch
import numpy as np

from log_reader import LogReader
from gen_anomal import generate_anormal

# Prepare dataset

class ConnlogsDataset(Dataset):
    def __init__(self, dataset_root, nn_type, tvt_type, timestamp=False):
        if tvt_type == "train":
            tvt_type = "Train"
        elif tvt_type == "test":
            tvt_type = "Test"
        elif tvt_type == "validate":
            tvt_type = "Validate"

        oneclass_flag = "oneclass" in nn_type
        period_flag = "period" in nn_type
        if oneclass_flag and period_flag:
            dataset_root = os.path.join(dataset_root, "nn_oneclass_period", tvt_type + ".npz")
            nn_initializer = self.nn_init
        elif oneclass_flag:
            dataset_root = os.path.join(dataset_root, "nn_oneclass", tvt_type+".npz")
            nn_initializer = self.nn_init
        elif period_flag:
            dataset_root = os.path.join(dataset_root, "nn_period", tvt_type + ".npz")
            nn_initializer = self.nn_init
        else:
            dataset_root = os.path.join(dataset_root, "nn", tvt_type + ".npz")
            nn_initializer = self.nn_init

        super(ConnlogsDataset, self).__init__()

        self.nn_type = nn_type
        self.tvt_type = tvt_type
        self.needtimestamp = timestamp

        data_x, data_y = nn_initializer(dataset_root)

        self.data_x = data_x
        self.data_y = data_y
        self.feat_num = data_x.size(dim=-1)

    def __len__(self):
        return self.data_x.shape[0]

    def __getitem__(self, index):
        if self.needtimestamp:
            return self.data_x[index], self.data_y[index], self.timestamp[index]

        return self.data_x[index], self.data_y[index]

    '''
        process_connlogs()
        This function reads all conn.log files, splits, and creates Train, Validate and Test data file
        
        @params:
            log_path: Stores all conn.log created by zeek. 
                    MUST be named in chronological order!
            dataset_path: Stores the generated dataset. The following files will be generated
                $(dataset_path)/nn/Test.txt
                $(dataset_path)/nn/Train.txt
                $(dataset_path)/nn/Validate.txt
                $(dataset_path)/nn_oneclass/Test.txt
                $(dataset_path)/nn_oneclass/Train.txt
                $(dataset_path)/nn_oneclass/Validate.txt
            test_portion: The portion for Test data
            val_portion: The portion for Validate data
            abnormal_rate: The global abnormal rate.
    '''
    @staticmethod
    def process_connlogs(log_file, dataset_path,
                         test_portion=0.15, val_portion=0.15, abnormal_rate=0.1,
                         nn_fields=[]):

        # 1. Read in all log entries (as LogReaders)
        log_reader = LogReader(log_file)
        entry_nums = len(log_reader)

        test_nums = int(entry_nums * test_portion)
        val_nums = int(entry_nums * val_portion)
        train_nums = entry_nums - test_nums - val_nums

        nn_fields_idx = [log_reader.fields[name] for name in nn_fields]

        # Create general dataset
        test_dataset = {'label': [], 'ts': [], 'feature': []}
        val_dataset = {'label': [], 'ts': [], 'feature': []}
        train_dataset = {'label': [], 'ts': [], 'feature': []}
        for i, log_record in enumerate(log_reader):
            if i < train_nums:
                train_dataset['label'].append(1)
                train_dataset['ts'].append(log_record.timestamp)
                train_dataset['feature'].append([log_record.values[idx] for idx in nn_fields_idx])
            elif i < train_nums + val_nums:
                val_dataset['label'].append(1)
                val_dataset['ts'].append(log_record.timestamp)
                val_dataset['feature'].append([log_record.values[idx] for idx in nn_fields_idx])
            else:
                test_dataset['label'].append(1)
                test_dataset['ts'].append(log_record.timestamp)
                test_dataset['feature'].append([log_record.values[idx] for idx in nn_fields_idx])

            rand_val = random.uniform(0, 1)
            if rand_val < abnormal_rate:
                # Gen abnormal data
                anormal_record = generate_anormal(log_reader)
                if i < train_nums:
                    train_dataset['label'].append(0)
                    train_dataset['ts'].append(log_record.timestamp)
                    train_dataset['feature'].append([anormal_record.values[idx] for idx in nn_fields_idx])
                elif i < train_nums + val_nums:
                    val_dataset['label'].append(0)
                    val_dataset['ts'].append(log_record.timestamp)
                    val_dataset['feature'].append([anormal_record.values[idx] for idx in nn_fields_idx])
                else:
                    test_dataset['label'].append(0)
                    test_dataset['ts'].append(log_record.timestamp)
                    test_dataset['feature'].append([anormal_record.values[idx] for idx in nn_fields_idx])

        # Transform to numerical
        for i in range(len(test_dataset['feature'])):
                test_dataset['feature'][i] = [ConnlogsDataset.log_numerical_mapping(idx, test_dataset['feature'][i][j]) for j, idx in enumerate(nn_fields)]

        for i in range(len(val_dataset['feature'])):
                val_dataset['feature'][i] = [ConnlogsDataset.log_numerical_mapping(idx, val_dataset['feature'][i][j]) for j, idx in enumerate(nn_fields)]

        for i in range(len(train_dataset['feature'])):
                train_dataset['feature'][i] = [ConnlogsDataset.log_numerical_mapping(idx, train_dataset['feature'][i][j]) for j, idx in enumerate(nn_fields)]

        def gen_dataset(_train, _val, _test, _outPath, _oneclass=False, _period=True, _fields=[]):
            _test_dataset = {'label': np.asarray(_test['label']),
                               'ts': np.asarray(_test['ts']),
                               'feature': np.asarray(_test['feature'])}
            _val_dataset = {'label': np.asarray(_val['label']),
                              'ts': np.asarray(_val['ts']),
                              'feature': np.asarray(_val['feature'])}
            _train_dataset = {'label': np.asarray(_train['label']), 'ts': np.asarray(_train['ts']),
                                'feature': np.asarray(_train['feature'])}

            if not _period:
                _period_fields = ['period', 'period_length0', 'period_length1', 'period_length2', 'period_length3',
                                 'period_length4']
                _period_idx = [_fields.index(_name) for _name in _period_fields]

                _train_dataset['feature'] = np.delete(_train_dataset['feature'], _period_idx, 1)
                _val_dataset['feature'] = np.delete(_val_dataset['feature'], _period_idx, 1)
                _test_dataset['feature'] = np.delete(_test_dataset['feature'], _period_idx, 1)

            if _oneclass:
                _anormal_indicies = np.where(_val_dataset['label'] == 0)[0]
                _val_dataset['label'] = np.delete(_val_dataset['label'], _anormal_indicies, 0)
                _val_dataset['ts'] = np.delete(_val_dataset['ts'], _anormal_indicies, 0)
                _val_dataset['feature'] = np.delete(_val_dataset['feature'], _anormal_indicies, 0)
                _anormal_indicies = np.where(_train_dataset['label'] == 0)[0]
                _train_dataset['label'] = np.delete(_train_dataset['label'], _anormal_indicies, 0)
                _train_dataset['ts'] = np.delete(_train_dataset['ts'], _anormal_indicies, 0)
                _train_dataset['feature'] = np.delete(_train_dataset['feature'], _anormal_indicies, 0)

            _nn_mean = np.average(_train_dataset['feature'], axis=0)
            _nn_std = np.std(_train_dataset['feature'], axis=0)

            _del_cols = np.where(_nn_std == 0)[0]
            if len(_del_cols) != 0:
                _nn_std[_del_cols] = 1.0

            _test_dataset['mean'] = _val_dataset['mean'] = _train_dataset['mean'] = _nn_mean
            _test_dataset['std'] = _val_dataset['std'] = _train_dataset['std'] = _nn_std

            _test_dataset['feature'] -= _test_dataset['mean']
            _val_dataset['feature'] -= _val_dataset['mean']
            _train_dataset['feature'] -= _train_dataset['mean']

            _test_dataset['feature'] /= _test_dataset['std']
            _val_dataset['feature'] /= _val_dataset['std']
            _train_dataset['feature'] /= _train_dataset['std']

            np.savez(os.path.join(_outPath, "Train.npz"), label=_train_dataset['label'],
                     ts=_train_dataset['ts'], feature=_train_dataset['feature'],
                     mean=_train_dataset['mean'], std=_train_dataset['std'])
            np.savez(os.path.join(_outPath, "Validate.npz"), label=_val_dataset['label'],
                     ts=_val_dataset['ts'], feature=_val_dataset['feature'],
                     mean=_val_dataset['mean'], std=_val_dataset['std'])
            np.savez(os.path.join(_outPath, "Test.npz"), label=_test_dataset['label'],
                     ts=_test_dataset['ts'], feature=_test_dataset['feature'],
                     mean=_test_dataset['mean'], std=_test_dataset['std'])

            return _train_dataset, _val_dataset, _test_dataset
        # Generate NN period

        nn_period_train, nn_period_val, nn_period_test = \
            gen_dataset(train_dataset, val_dataset, test_dataset, os.path.join(dataset_path, "nn_period"),
                        _oneclass=False, _period=True, _fields=nn_fields)

        nn_train, nn_val, nn_test = \
            gen_dataset(train_dataset, val_dataset, test_dataset, os.path.join(dataset_path, "nn"),
                        _oneclass=False, _period=False, _fields=nn_fields)

        nn_oneclass_train, nn_oneclass_val, nn_oneclass_test = \
            gen_dataset(train_dataset, val_dataset, test_dataset, os.path.join(dataset_path, "nn_oneclass"),
                        _oneclass=True, _period=False, _fields=nn_fields)

        nn_oneclass_period_train, nn_oneclass_period_val, nn_oneclass_period_test = \
            gen_dataset(train_dataset, val_dataset, test_dataset, os.path.join(dataset_path, "nn_oneclass_period"),
                        _oneclass=True, _period=True, _fields=nn_fields)

        '''
            Visualize params above
        '''
        def write_datasets(_train, _val, _test, _path, _fields):
            with open(os.path.join(_path, "Train.txt"), "w") as _f_train, \
                    open(os.path.join(_path, "Validate.txt"), "w") as _f_val, \
                    open(os.path.join(_path, "Test.txt"), "w") as _f_test:
                _nn_mean = _train['mean']
                _nn_std = _train['std']

                def write_fileheader(_file_obj, _fields, _mean=None, _std=None):
                    if _mean is not None:
                        _file_obj.write("#mean\t" + '\t'.join(['%.5f' % num for num in _mean]) + "\n")
                    if _std is not None:
                        _file_obj.write("#std\t" + '\t'.join(['%.5f' % num for num in _std]) + "\n")

                    _file_obj.write("#fields\tlabel\tts\t")
                    for field in _fields:
                        _file_obj.write(field + "\t")
                    _file_obj.write("\n")

                def write_entry(_file_obj, _label, _ts, _data, sep="\t"):
                    _file_obj.write(str(_label) + sep)
                    _file_obj.write(str(_ts) + sep)
                    for val in _data:
                        _file_obj.write(str(val) + sep)
                    _file_obj.write("\n")

                # 1. Write column names
                write_fileheader(_f_train, _fields, _nn_mean, _nn_std)
                write_fileheader(_f_val, _fields, _nn_mean, _nn_std)
                write_fileheader(_f_test, _fields, _nn_mean, _nn_std)

                # Write entries
                for i, data in enumerate(_train['feature']):
                    write_entry(_f_train, _train['label'][i], _train['ts'][i], data)
                for i, data in enumerate(_val['feature']):
                    write_entry(_f_val, _val['label'][i], _val['ts'][i], data)
                for i, data in enumerate(_test['feature']):
                    write_entry(_f_test, _test['label'][i], _test['ts'][i], data)

        write_datasets(nn_train, nn_val, nn_test, os.path.join(dataset_path, "nn"), nn_fields)
        write_datasets(nn_period_train, nn_period_val, nn_period_test, os.path.join(dataset_path, "nn_period"), nn_fields)
        write_datasets(nn_oneclass_train, nn_oneclass_val, nn_oneclass_test, os.path.join(dataset_path, "nn_oneclass"), nn_fields)
        write_datasets(nn_oneclass_period_train, nn_oneclass_period_val, nn_oneclass_period_test, os.path.join(dataset_path, "nn_oneclass_period"), nn_fields)

    '''
        nn_init
        Initialize dataset for nn.
    '''
    def nn_init(self, file_obj):
        npz_object = np.load(file_obj)

        label = npz_object['label']
        data = npz_object['feature']
        self.timestamp = npz_object['ts']

        label = torch.LongTensor(label)
        data = torch.from_numpy(data)

        return data, label

    '''
        rnn_init
        Initialize dataset for rnn.
    '''
    def rnn_init(self, file_obj):
        # Format dictionary
        npz_object = np.load(file_obj)

        label = npz_object['label']
        seqs = npz_object['feature']
        self.timestamp = npz_object['ts']

        l_list = [len(row) for row in seqs]

        fin_seqs = []
        fin_label = []
        for i in range(len(label)):
            temp_seq = []
            temp_label = 1
            for tempI in range(i, i+1):
                temp_seq.append(seqs[tempI])
                temp_label = temp_label & label[i]
            fin_seqs.append(temp_seq)
            fin_label.append(temp_label)

        l_list = torch.LongTensor(l_list)
        label = torch.LongTensor(fin_label)
        data = torch.zeros((len(fin_seqs), len(fin_seqs[0]), len(fin_seqs[0][0])), dtype=torch.float32)
        for i, seq_id in enumerate(fin_seqs):
            for j, word_ids in enumerate(seq_id):
                for k, word_id in enumerate(word_ids):
                    data[i][j][k] = word_id

        return (data, l_list), label

    @staticmethod
    def log_numerical_mapping(field, value):
        translated_value = 0.0
        if field == "id.orig_h" or field == "id.resp_h":
            splits = value.split(".")
            if len(splits) > 1:
                translated_value = 0
                for s in splits:
                    translated_value *= 256
                    translated_value += int(s)
            else:
                translated_value = 3232235777
        elif field == "proto":
            if value == "tcp":
                translated_value = 0
            elif value == "udp":
                translated_value = 1
            else:
                translated_value = 2
        elif field == "service":
            if value == "http":
                translated_value = 1
            elif value == "ssl":
                translated_value = 2
            elif value == "dns":
                translated_value = 3
            elif value == "ntp":
                translated_value = 4
            elif "dce_rpc" in value:
                translated_value = 5
            elif value == "dhcp":
                translated_value = 6
            elif "smb" in value:
                translated_value = 7
            else:
                translated_value = 0
        elif field == "conn_state":
            if value == "SF":
                translated_value = 1
            elif value == "S0":
                translated_value = 2
            elif value == "RSTO":
                translated_value = 3
            elif value == "REJ":
                translated_value = 4
            elif value == "OTH":
                translated_value = 5
            elif value == "RSTR":
                translated_value = 6
            elif value == "S1":
                translated_value = 7
            elif value == "S2":
                translated_value = 8
            elif value == "SH":
                translated_value = 9
            elif value == "SHR":
                translated_value = 10
            else:
                translated_value = 0
        elif field == "land":
            if value == "F":
                translated_value = 0
            else:
                translated_value = 1
        elif field == "label":
            label = int(value)
            translated_value = label
        else:
            if value == '-':
                value = "0"
            translated_value = float(value)

        return translated_value
class Dictionary(object):
    def __init__(self):
        self.word2idx = {'PAD': 0}
        self.idx2word = {0: 'PAD'}
        self.pad = 0
        self.idx = 1

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word[self.idx] = word
            self.word2idx[word] = self.idx  # pad, sos, end
            self.idx += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word.keys())


if __name__ == "__main__":
    # Parse params
    import argparse

    parser = argparse.ArgumentParser(description='Format dataset fom conn_log.')
    parser.add_argument('--datafile', default='../out/period_output/SummaryWithPeriod.tsv',
                        help='Input log files.')
    parser.add_argument('--datasetPath', default='../out/dataset',
                        help='output dataset path.')

    args = parser.parse_args()

    dataset_path = args.datasetPath
    data_file = args.datafile

    ConnlogsDataset.process_connlogs(data_file, dataset_path,
                                     nn_fields=["id.orig_h", "id.resp_h", "id.resp_p", "proto", "service", "duration",  "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes", "orig_ttl",
                    "dest_ttl", "land", "pkts_count", "min_IAT", "max_IAT", "mean_IAT"
                  , 'period', 'period_length0', 'period_length1',
                  'period_length2', 'period_length3', 'period_length4'
                  ])

    connDs1 = ConnlogsDataset(dataset_path, nn_type='nn', tvt_type='train')
    connDs2 = ConnlogsDataset(dataset_path, nn_type='nn_oneclass', tvt_type='train')
    pass