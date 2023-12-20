import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_loader.loader import U2I_Infer_Dataset
from model import EmbTransformer


def infer(args):
    checkpoint = torch.load('parameters/best_model.pt')
    model = EmbTransformer(checkpoint['args'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    infer_dataset = U2I_Infer_Dataset(args)
    infer_loader = DataLoader(dataset=infer_dataset,
                        batch_size=args.batch_size,
                        shuffle=False)
    
    item_pool = torch.arange(1, args.vocab_size)
    item_feats = model(item_id=item_pool, item_feats=None)  

    rs = []
    for infer_batch in infer_loader:
        user_feats = model(user_hist=infer_batch['hist'], att_mask=infer_batch['att_mask'])
        logit = torch.mm(user_feats, torch.t(item_feats))
        top5 = torch.topk(logit, 5)
        rs.append(top5)

    return {ix:rs_li for ix, rs_li in enumerate(rs)}
        




            
                                      

