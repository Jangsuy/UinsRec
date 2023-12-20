
import random
import warnings

import torch
import torch.nn.functional as F

from model import EmbTransformer


warnings.filterwarnings("ignore")


class PT:
    def __init__(self, args):
        self.args = args
        # self.model = EmbTransformer(args).to('cuda')
        self.model = EmbTransformer(args)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=args.lr, eps=1e-6)

    def rec_loss(self, results, batch_dict):
        user_feats = self.model(user_hist=batch_dict['hist'], att_mask=batch_dict['att_mask'])
        item_feats = self.model(item_id=batch_dict['item_ix'], item_feats=None)
        # print(user_feats.shape, item_feats.shape,batch_dict['y'] ,(user_feats * item_feats).shape)
        logits = (user_feats * item_feats).mean(dim=-1)
        loss = F.binary_cross_entropy_with_logits(logits, batch_dict['y'].float())
        results.update({'loss': loss.mean()})
        return results
    
    def evaluation(self, results, batch_dict):
        true_ix = batch_dict['y'] == 1
        if sum(true_ix) >= 1: 
            neg_set = list(set(range(self.args.vocab_size)) - set(batch_dict['item_ix'][true_ix]))
            ed_ix = random.sample(range(100, len(neg_set)), 1)[0]
            neg_sample = torch.tensor(neg_set[ed_ix - 99 : ed_ix], dtype=torch.int)

            user_feats = self.model(user_hist=batch_dict['hist'][true_ix], att_mask=batch_dict['att_mask'][true_ix])    
            true_feats = self.model(item_id=batch_dict['item_ix'][true_ix], item_feats=None)
            neg_feats = self.model(item_id=neg_sample, item_feats=None)

            neg_logit = torch.mm(user_feats, torch.t(neg_feats))
            pos_logit = (user_feats * true_feats).mean(dim=-1).unsqueeze(dim=1)
            full_logit = torch.cat([pos_logit, neg_logit], dim=1)
            results['logits'] = full_logit
            


        return results

    def run_fwd_bwd(self, batch_dict, _results, is_valid):

        _results = self.rec_loss(_results, batch_dict)

        if not is_valid:
            _results['loss'].backward()
        else:
            _results = self.evaluation(_results, batch_dict)

        return _results

    def iteration(self, batch_dict, is_valid): 
        _results = dict()
        if not is_valid:
            self.model.train()
            _results = self.run_fwd_bwd(
                batch_dict, _results, is_valid)
            self.optim.step()
            self.optim.zero_grad()

        else:
            self.model.eval()
            with torch.no_grad():
                _results = self.run_fwd_bwd(batch_dict, _results, is_valid)
        return _results

                
    # def rec_result(self, batch_dict):
    #     user_feats = self.model(
    #         user_hist=batch_dict['hist'], att_mask=batch_dict['att_mask'])
    #     neg_items = random.sample(range(1, args.vocab_size + 1))
    #     item_feats = self.model(item_id=batch_dict['item_ix'], item_feats=None)

