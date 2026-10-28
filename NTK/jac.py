import torch

def F(f:torch.nn.Sequential,theta:dict[str,torch.Tensor],x:torch.Tensor):
    return torch.func.functional_call(f,theta,x)

class Jac:
    def __init__(self):
        self.F = F
        self.DF = torch.func.jacrev(self.F,argnums=1,) # args: (f,theta,x)

    def jac_eval(self,f:torch.nn.Sequential,x:torch.Tensor):
        theta = dict(f.named_parameters())
        return self.DF(f,theta,x)

    def jac_batch(self,f:torch.nn.Sequential,X:torch.Tensor):
        """
        X.shape[0] <-- Number of datapoints (N) have the first index

        X.shape[1] <-- Datapoint dimension (n_L)
        """
        j_batch = torch.func.vmap(self.jac_eval,in_dims=(None,0))(f,X)
        J =  torch.cat([j.flatten(2) for j in j_batch.values()],dim=2).flatten(0,1) # dict to N (n_L x P)
        return J
