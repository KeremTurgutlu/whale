import torch
import torch.nn as nn
from tqdm import tqdm_notebook

def accuracy_score(input, target, thresh=0.5):
    """calculate accuracy with single dimension input, e.g. after sigmoid"""
    return torch.mean(((input > thresh).view(-1) == target.view(-1).byte()).float())

def pairwise_l2_dist(m1, m2):
    """calculate pairwise p=2 distance of 2 matrices: m1 and m2"""
    return (m1.unsqueeze(1) - m2).pow(2).sum(dim=-1)

def pairwise_l1_dist(m1, m2):
    """calculate pairwise p=2 distance of 2 matrices: m1 and m2"""
    return (m1.unsqueeze(1) - m2).abs().sum(dim=-1)

def pairwise_rel(a1, a2):
    """calculate pairwise relevance of 2 target/label vectors"""
    return a1.unsqueeze(1) == a2

def pairwise_metric_dist(m1, m2, W):
    """
    calculate pairwise distance between m1 and m2 
    given a learnable W transformation matrix
    (x1 - x2).T @ W @ (x1-x2) : postive or zero 
    https://en.wikipedia.org/wiki/Definiteness_of_a_matrix
    """
    d = m1.unsqueeze(1) - m2
    return torch.einsum("ijk,kk,ijk->ij",d, W, d)

class MetricLearning(nn.Module):
    """
    https://en.wikipedia.org/wiki/Similarity_learning
    (x1 - x2).T @ W @ (x1-x2), where W is positive semi-definite
    """
    def __init__(self):
        super().__init__()
        
    def forward(self):
        pass


def mapk_from_rel(topk, k):
    """
    Given a topk binary relevance matrix and k calculate 
    mean average precision.
    """
    return torch.max(topk.float()*torch.linspace(1,0,k), dim=1)[0].mean()

def mapk(pwise_distances, pwise_targets, k=5):
    """
    Calculate mean average precision from 
    pwise_distances: pairwise distance (N_query x N_vocab)
    pwise_targets: pairwise binary target match (N_query x N_vocab)
    k: k of map@k
    """
    indices = torch.argsort(pwise_distances, dim=1)
    # sorted by pwise distances, smallest to greatest
    sorted_distances = torch.zeros_like(pwise_distances)
    sorted_targets = torch.zeros_like(pwise_targets)
    for i, idx in enumerate(indices):
        sorted_distances[i] = pwise_distances[i][idx]
        sorted_targets[i] = pwise_targets[i][idx]
    topk_rel = sorted_targets[:, :k]
    return mapk_from_rel(topk_rel, k)

def module_predict_dl(module, dl): 
    """
    given a torch module and dataloader return 
    output from the module
    """
    model = module.eval()
    preds = []
    with torch.no_grad():
        for xb, yb in tqdm_notebook(dl):
            preds.append(model(xb).cpu())
    return torch.cat(preds, dim=0)




























# tests
def pairwise_dist_tests():
    vocab_emb = torch.rand((10,3))
    query_emb = torch.rand((5,3))
    vocab_target = torch.randint(high=2, size=(10,))
    query_target = torch.randint(high=2, size=(5,))
    euc_dist = pairwise_l2_dist(vocab_emb, query_emb)
    metric_dist = pairwise_metric_dist(vocab_emb, query_emb, W=torch.eye(3, 3))
    assert torch.equal(euc_dist, metric_dist)