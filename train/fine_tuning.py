import sys
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

sys.path.append("C:/Users/user/Documents/uinsrec_playground")
from data_loader.loader import FT_Dataset
from model.UnisRec import UnisRec

## pretrain에 모델 파라미터 저장 코드 작성 - 완료
## 파일 실행 해봐 - 완료
## finetuning dataset 생성
## finetuning에서 loader은 변경할 필요없을것같긴한데...
## train 부분 함수로 변경해서 추가해도 깔--끔할듯


def finetune(args):
    ## load pretrained model
    checkpoint = torch.load(f'{args.model_path}/best_model.pt')
    model = UnisRec(checkpoint['args'])
    model.load_state_dict(checkpoint['model_state_dict'])

    ## freezing encoder
    for params in model.behavior_enc.parameters():
        params.requires_grad = False

    ## define optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-6)

    tr_dataset = FT_Dataset(args, is_valid=False)
    tr_loader = DataLoader(dataset=tr_dataset,
                           batch_size=args.batch_size,
                           shuffle=True)
    val_dataset = FT_Dataset(args, is_valid=True)
    val_loader = DataLoader(dataset=val_dataset,
                           batch_size=args.batch_size)

    prev_loss, cnt = 10000, 0
    for i in range(args.epochs):
        tr_loss, val_loss = [], []

        model.train()
        for tr_batch in tr_loader:
            loss = model(tr_batch)

            loss.backward()
            optim.step()
            optim.zero_grad()

            tr_loss.append(loss.detach().numpy())

        model.eval()
        with torch.no_grad():
            for val_batch in val_loader:
                loss = model(val_batch)

                val_loss.append(loss.detach().numpy())

        
        if prev_loss <= np.mean(val_loss):
            cnt += 1
        else:
            prev_loss = np.mean(val_loss)
            cnt = 0

        if cnt == 5:
            break

        print(f"epochs: {i} train loss: {np.mean(tr_loss)}, val loss: {np.mean(val_loss)}")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    ##path
    parser.add_argument('--model_path', dest='model_path', type=str, default='parameters')

    ##loader argument
    parser.add_argument('--WordDrop', dest='WordDrop', type=float, default=0.2)
    parser.add_argument('--ft_item_size', dest='ft_item_size', type=int, default=98112)

    # model argument
    parser.add_argument('--nhead', dest='nhead', type=int, default=2)
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', dest='num_layers', type=int, default=2)

    parser.add_argument('--bert_dim', dest='bert_dim', type=int, default=768)
    parser.add_argument('--MTL_weight', dest='MTL_weight', type=float, default=1e-3)
    parser.add_argument('--n_exps', dest='n_exps', type=int, default=8)
    parser.add_argument('--temperature', dest='temperature', type=float, default=0.07)

    ## train argument 
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=512)
    parser.add_argument('--lr', dest='lr', type=float, default=0.001)
    parser.add_argument('--epochs', dest='epochs', type=int, default=10)

    args = parser.parse_args()
    finetune(args)


    
     




            
                                      

