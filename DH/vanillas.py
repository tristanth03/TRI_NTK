import torch

def C(S:torch.Tensor,K=1,i=0):
    return torch.clamp(S[...,i]-K,min=0)

def P(S:torch.Tensor,K=1,i=0):
    return torch.clamp(K-S[...,i],min=0)