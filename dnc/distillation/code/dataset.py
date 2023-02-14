import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm


class EmbDataset(Dataset):
    def __init__(self, data_path, tokenizer, size=None, text_col="text", target_col="sentiment", mean=True):
        self.X, self.y = zip(*pd.read_csv(data_path)[[text_col, target_col]][:size].to_numpy())
        self.tokenizer = tokenizer
        self.X = [self.tokenizer(x) for x in tqdm(self.X, desc="tokenizer")]

        non_empty_mask = {i for i,x in enumerate(self.X) if x.nelement()}
        self.X = [x for i,x in enumerate(self.X) if i in non_empty_mask]
        self.y = [y for i,y in enumerate(self.y) if i in non_empty_mask]
        
        if mean:
            self.X = [x.mean(0) for x in tqdm(self.X, desc="mean")]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def __len__(self):
        return len(self.X)

    

class Collator:
    def __init__(self, pad_value=0):
        self.pad_value = pad_value

    def __call__(self, batch):
        feats, labels = zip(*batch)
        lens = [len(x) for x in feats]
        feats = pad_sequence(feats, batch_first=True, padding_value=self.pad_value)
        return feats, torch.tensor(labels), lens
