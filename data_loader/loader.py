


import random
import warnings
import argparse

import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore') 

class PT_Dataset(Dataset):
    def __init__(self, args, is_valid):
        super().__init__()
        self.item_feat = np.load('data/CF_feat.npy')
        if is_valid:
            self.pos_pairs = np.load('data/CF_valid_pos_pairs.npy')[:3000]
            self.hist = np.load('data/CF_valid_hist.npy')
        else:
            self.pos_pairs = np.load('data/CF_train_pos_pairs.npy')[:3000]
            self.hist = np.load('data/CF_train_hist.npy')
        self.args = args


    def __len__(self):
        return self.pos_pairs.shape[0]
        
    
    def __getitem__(self, ix):
        batch_data_dict = dict()
        user_ix = self.pos_pairs[ix, 0]

        hist = torch.tensor(self.hist[user_ix, :], dtype=torch.int)
        att_mask = torch.tensor(hist == 0, dtype=torch.bool)
        aug_hist = self.drop_seq(hist).type(torch.int)
        aug_att_mask = torch.tensor(aug_hist == 0, dtype=torch.bool)
        item_ix = torch.tensor(self.pos_pairs[ix, 1], dtype=torch.int)

        batch_data_dict.update({'hist': self.ix2feat(hist)})
        batch_data_dict.update({'att_mask': att_mask})

        batch_data_dict.update({'aug_hist': self.ix2feat(aug_hist)})
        batch_data_dict.update({'aug_att_mask': aug_att_mask})

        batch_data_dict.update({'item_ix': self.ix2feat(item_ix)})

        return batch_data_dict
    
    def drop_seq(self, hist):
        hist_len = sum(hist != 0)
        drop_cnt = int(hist_len * self.args.WordDrop)
        drop_ix = random.sample(range(hist_len), drop_cnt)
        ## drop item
        aug_hist = hist.clone().detach()
        aug_hist[drop_ix] = 0
        aug_hist = aug_hist[aug_hist > 0]

        return torch.cat([aug_hist, torch.zeros(50 - len(aug_hist))])
    
    def ix2feat(self, item):
        return torch.tensor(self.item_feat[item,:], dtype=torch.float32)
    


class FT_Dataset(Dataset):
    def __init__(self, args, is_valid):
        super().__init__()
        self.item_feat = np.load('data/Kindle_feat.npy')
        if is_valid:
            self.pos_pairs = np.load('data/Kindle_valid_pos_pairs.npy')[:3000]
            self.hist = np.load('data/Kindle_valid_hist.npy')
        else:
            self.pos_pairs = np.load('data/Kindle_train_pos_pairs.npy')[:3000]
            self.hist = np.load('data/Kindle_train_hist.npy')
        self.args = args


    def __len__(self):
        return self.pos_pairs.shape[0]
        
    
    def __getitem__(self, ix):
        batch_data_dict = dict()
        user_ix = self.pos_pairs[ix, 0]

        hist = torch.tensor(self.hist[user_ix, :], dtype=torch.int)
        att_mask = torch.tensor(hist == 0, dtype=torch.bool)
        item_ix = torch.tensor(self.pos_pairs[ix, 1], dtype=torch.int)
        neg_item_ix = torch.tensor(random.choice(range(1, self.args.ft_item_size)), dtype=torch.int)

        batch_data_dict.update({'hist': self.ix2feat(hist)})
        batch_data_dict.update({'att_mask': att_mask})

        batch_data_dict.update({'item_ix': self.ix2feat(item_ix)})
        batch_data_dict.update({'neg_item_ix': self.ix2feat(neg_item_ix)})

        return batch_data_dict
    
    def ix2feat(self, item):
        return torch.tensor(self.item_feat[item,:], dtype=torch.float32)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument('--loss_type', dest='loss_type', type=str, default='seq2seq')
    # basic settings
    parser.add_argument('--WordDrop', dest='WordDrop', type=int, default=0.2)
    args = parser.parse_args()
    pt_ds = PT_Dataset(args, False)
    pt_ld = DataLoader(dataset=pt_ds,
                           batch_size=2,
                           shuffle=True)

    for batch in pt_ld:
        print(batch['hist'].dtype)
        break
    