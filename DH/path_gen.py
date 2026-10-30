import torch


def gen_simple_path(
        S0=1.0,
        sigma=0.2,
        T=1,
        M=10,
        N=1,
        n_L=1,
        seed=0,
        dtype_=torch.float32):
    torch.manual_seed(seed)
    delta_t = T/M
    S = torch.zeros(M+1,N,n_L,dtype=dtype_)
    S[0,:,:] = S0
    for k in range(1,M+1):
        eps_k = torch.randint(-1,1+1,(1,N,n_L))
        S[k,:,:] = S[k-1,:,:]*(1+delta_t*sigma*eps_k)

    t_ = (T-torch.linspace(0, T, M+1))[:, None, None].expand(-1, N, -1)
    X_partial = torch.cat((S,t_),dim=2)
    return X_partial

def A(f:torch.nn.Sequential,X:torch.Tensor):
    M_,_ = X.shape
    W = 0
    for m in range(1,M_):
        W+=f(X[m-1,:])*(X[m,0]-X[m-1,0])
    return W



