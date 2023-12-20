import sys
from tqdm import tqdm
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
sys.path.append("C:/Users/user/Documents/uinsrec_playground")
from data_loader.loader import PT_Dataset
from model.lm_model import UnisRec



def run_pretrain(args):
    # random_seed(args.seed)
    ## define model & optimizer
    model = UnisRec(args)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-6)
    ## dataloader
    tr_dataset = PT_Dataset(args, is_valid=False)
    tr_loader = DataLoader(dataset=tr_dataset,
                           batch_size=args.batch_size,
                           shuffle=True)
    val_dataset = PT_Dataset(args, is_valid=True)
    val_loader = DataLoader(dataset=val_dataset,
                           batch_size=args.batch_size)

    prev_loss, cnt = 10000, 0
    for i in range(args.epochs):
        tr_loss, val_loss = [], []

        model.train()
        for tr_batch in tqdm(tr_loader):
            loss = model.pretrain(tr_batch)

            loss.backward()
            optim.step()
            optim.zero_grad()

            tr_loss.append(loss.detach().numpy())

        model.eval()
        with torch.no_grad():
            for val_batch in tqdm(val_loader):
                loss = model.pretrain(val_batch)

                val_loss.append(loss.detach().numpy())

        
        if prev_loss <= np.mean(val_loss):
            cnt += 1
        else:
            prev_loss = np.mean(val_loss)
            cnt = 0

        if cnt == 5:
            break

        print(f"epochs: {i} train loss: {np.mean(tr_loss)}, val loss: {np.mean(val_loss)}")

    torch.save({'model_state_dict':model.state_dict(),
                'args':args}, f'{args.model_path}/best_model.pt') 

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    ##path
    parser.add_argument('--model_path', dest='model_path', type=str, default='parameters')

    # model argument
    parser.add_argument('--WordDrop', dest='WordDrop', type=float, default=0.2)
    parser.add_argument('--nhead', dest='nhead', type=int, default=2)
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', dest='num_layers', type=int, default=2)

    parser.add_argument('--bert_dim', dest='bert_dim', type=int, default=768)
    parser.add_argument('--MTL_weight', dest='MTL_weight', type=float, default=1e-3)
    parser.add_argument('--n_exps', dest='n_exps', type=int, default=8)
    parser.add_argument('--temperature', dest='temperature', type=float, default=0.07)

    ## train argument 
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=512)
    parser.add_argument('--lr', dest='lr', type=float, default=0.0001)
    parser.add_argument('--epochs', dest='epochs', type=int, default=10)

    args = parser.parse_args()

    run_pretrain(args)



        
   



