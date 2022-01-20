import torch
from torch.utils.data import Dataset, DataLoader

'''
    自定义数据集对象，实现__len__()方法和__getitem__()方法
'''
class MYDATASET(Dataset):
    def __init__(self, data_array):
        '''
        Args:
            set_name: Decide which dataset is loaded. For this project, only "train" and "test" are supported.
        '''
        super(MYDATASET, self).__init__()
        self.raw_data = torch.tensor(data_array[:, 2:], dtype=torch.float)

        self.label = torch.tensor(data_array[:, 0], dtype=torch.float)
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        raw_data = self.raw_data[index].flatten()
        label = self.label[index]

        return raw_data, label
