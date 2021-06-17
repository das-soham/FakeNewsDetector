import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

class NewsProducer(Dataset):
    def __init__(self,path, filename):
        self.data = pd.read_csv(os.path.join(path,filename), delimiter = ',', error_bad_lines=False)
    def __len__(self):
        return (len(self.data))
    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        news = self.data.loc[item, 'text']
        label = self.data.loc[item, 'Label']
        return news,label


training_dataset = NewsProducer('data', 'Train.csv')

torch.manual_seed(1)
train_loader = DataLoader(training_dataset,batch_size=10,shuffle=True, num_workers=0)


