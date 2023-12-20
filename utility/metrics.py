import torch

def hit_rate(k, logits):
    topk = torch.topk(logits, k)
    hits = topk.indices == 0
    return (hits.sum() / hits.shape[0]).numpy()


def mrr(logits):
    sort_ix = torch.argsort(logits, dim=1, descending=True)
    return (1 / (torch.where(sort_ix == 0)[1] + 1)).mean().numpy()


