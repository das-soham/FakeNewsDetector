import torch
from torch.utils.data import Dataset, IterableDataset
import pandas as pd
import os
#from itertools import cycle


class NewsProducer(Dataset):
    def __init__(self, path, filename):
        self.data = pd.read_csv(os.path.join(path, filename), delimiter=',', error_bad_lines=False)

    def __len__(self):
        return (len(self.data))

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        news = self.data.loc[item, 'text']
        label = self.data.loc[item, 'Label']
        return news, label


class IterableNewsProducer(IterableDataset):
    def __init__(self, path, filename):
        super(IterableNewsProducer).__init__()
        self.data = pd.read_csv(os.path.join(path, filename), delimiter=',', error_bad_lines=False)

    def datastream(self):
        for i in self.data.index:
            news = self.data.loc[i,'text']
            label = self.data.loc[i, 'Label']
            yield from (news, label)
    def __iter__(self):
        return self.datastream()








