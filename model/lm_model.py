
import torch
import torch.nn as nn
import torch.nn.functional as F

 
## mask value 확인 - padding이 True - 완료
## aug_seq 수정 - 완료
## contrastive learning loss 수정 - 완료
## positional encoding
## temperature 추가 - 완료

class PWLayer(nn.Module):
    """Single Parametric Whitening Layer
    """
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)


class MoEAdaptorLayer(nn.Module):
    """MoE-enhanced Adaptor
    """
    def __init__(self, args, hiddendropout=0.0, noise=True):
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = args.n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList([PWLayer(args.bert_dim, args.hidden_dim, hiddendropout) for i in range(self.n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(args.bert_dim, self.n_exps)
                                   ,requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(args.bert_dim, self.n_exps)
                                    ,requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training) # (B, n_E)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)] # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2) # (B, n_E, D)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2) # (B, D)
    

class UnisRec(nn.Module):
    def __init__(self, args):
        super().__init__()
        # behavior encoder 
        # moe adaptor 정의
        # pretrain 함수 안에 정의
        self.args = args 
        self.moe_adaptor = MoEAdaptorLayer(args)
        self.tf_enc = torch.nn.TransformerEncoderLayer(d_model=args.hidden_dim,
                                                       nhead=args.nhead,
                                                       batch_first=True)
        self.behavior_enc = torch.nn.TransformerEncoder(self.tf_enc,
                                                        num_layers=args.num_layers)
        
    def forward(self, batch_dict):
        hist = self.moe_adaptor(batch_dict['hist'])
        hist_emb = self.behavior_enc(hist, src_key_padding_mask=batch_dict['att_mask']).sum(dim=-2) # b s d -> b d
        
        pos_pairs = self.moe_adaptor(batch_dict['item_ix'])  # b d
        
        logits = hist_emb.mul(pos_pairs).sum(dim=-1)
        
        y = torch.ones(len(logits), dtype=int)
        
        return F.cross_entropy(logit, y)

    def pretrain(self, batch_dict):
        hist = self.moe_adaptor(batch_dict['hist'])
        hist_emb = self.behavior_enc(hist, src_key_padding_mask=batch_dict['att_mask']).sum(dim=-2) # b s d -> b d

        aug_hist = self.moe_adaptor(batch_dict['aug_hist'])
        aug_hist_emb = self.behavior_enc(aug_hist, src_key_padding_mask=batch_dict['aug_att_mask']).sum(dim=-2) # b s d -> b d
        
        pos_pairs = self.moe_adaptor(batch_dict['item_ix'])

        SeqSeq_loss = self.contrastive_loss(hist_emb, aug_hist_emb)
        SeqItem_loss = self.contrastive_loss(hist_emb, pos_pairs)
        return SeqItem_loss + self.args.MTL_weight * SeqSeq_loss


    
    def contrastive_loss(self, hist_emb, pairs_emb):
        logit = torch.matmul(hist_emb, pairs_emb.T) / self.args.temperature # b b
        y = torch.arange(len(logit)) # 대각원소 추출
        return F.cross_entropy(logit, y)
    





        

        


    

