import torch
import os
import sys
import pandas as pd
from torch.utils.data import Dataset,DataLoader

class  DataProducer(Dataset):
    def __init__(self, data_dir, data):
        self.truth = pd.read_csv(os.path.join(data_dir, data[0]),delimiter = ',', error_bad_lines=False)
        self.false = pd.read_csv(os.path.join(data_dir, data[1]),delimiter = ',', error_bad_lines=False)
        self.__blend()
    def __len__(self):
        return (len(self.truth), len(self.false))
    def __blend(self):
        self.truth['Label'] = 1
        self.false['Label'] = 0
        truth_train = self.truth[['text','Label']].sample(frac=0.8, random_state=1)
        false_train = self.false[['text', 'Label']].sample(frac=0.8, random_state=1)
        truth_test = self.truth.drop(truth_train.index)
        false_test = self.false.drop(false_train.index)
        self.train = truth_train.append(false_train, ignore_index=True).sample(frac=1, random_state=1).reset_index(drop=True)
        self.test = truth_test.append(false_test, ignore_index=True).sample(frac=1, random_state=1).reset_index(drop=True)
        return
    def __getitem__(self,idx):
        pass



files = ['True.csv','Fake.csv']
prod = DataProducer('data',files)
print(prod.__len__())
print(prod.train.head(5))
