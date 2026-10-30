from . import path_gen
from . import vanillas
import torch
import matplotlib.pyplot as plt
from NTK.ntk import NTK


SEED = 0


dtype_ = torch.double
S0=0.9;K=1.0;sigma=0.2;T=2
M=100;N=100;n_L=1
epochs=100

X_partial = path_gen.gen_simple_path(S0=S0,sigma=sigma,T=T,M=M,N=N,n_L=n_L,seed=SEED,dtype_=dtype_)
G = vanillas.C(X_partial,K=K)[-1,:]
print(G.shape)
K_ = torch.full((M+1, N, 1), K)

X = torch.cat((X_partial,K_),dim=2) # (M+1,N,n_0)

n_0 = X.shape[2]

f = torch.nn.Sequential(
    torch.nn.Linear(n_0,100),torch.nn.Tanh(),
    torch.nn.Linear(100,n_L)).to(dtype_)

K_op = NTK()

for t in range(M+1):
    print(K_op.spectrum(f,X[t,:,:])[-1])
print("\n")
for i in range(N):
    print(K_op.spectrum(f,X[:,i,:])[-1])

opt = torch.optim.SGD(f.parameters())
L_fn = torch.nn.MSELoss()
L_coll = []
for _ in range(1,epochs+1):
    opt.zero_grad()
    L = 0
    for i in range(N):        
        L+=L_fn(G[i],path_gen.A(f,X[:,i,:]))

    L /= N
    L.backward()
    opt.step()

    L_coll.append(L.detach().item())

plt.plot(L_coll)
plt.yscale("log")
plt.show()

diff = []
for i in range(N):
    diff.append((G[i]-path_gen.A(f,X[:,i,:])).detach().item())
print(diff)
plt.hist(diff)
plt.show()