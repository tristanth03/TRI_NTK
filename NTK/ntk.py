import torch
from . import jac
from time import perf_counter

### Note: the last two methods are not verified

class NTK:
    def __init__(self):
        pass

    def direct_jac(self,f:torch.nn.Sequential,X1:torch.Tensor,X2:torch.Tensor):
        J_op = jac.Jac()
        J1 = J_op.jac_batch(f,X1).detach()

        if X1 is X2:
            J2 = J1
        else:
            J2 = J_op.jac_batch(f,X2).detach()

        K = J1@J2.T
        return K

    def layerwise(self,f:torch.nn.Sequential,X1:torch.Tensor,X2:torch.Tensor,chunk_size=512):
        def factors(X):
            A,D,W=[],[],[]
            a=X

            for layer in f:
                if isinstance(layer,torch.nn.Linear):
                    A.append(a)
                    W.append(layer.weight)
                    a=layer(a)
                    D.append(torch.ones_like(a))
                elif isinstance(layer,torch.nn.Tanh):
                    a=layer(a)
                    D[-1]=1-a.square()

            S=[None]*len(W)

            if len(W)>1:
                s=W[-1].unsqueeze(0).expand(X.shape[0],-1,-1)

                for l in range(len(W)-2,-1,-1):
                    s=s*D[l][:,None,:]
                    S[l]=s
                    if l:
                        s=s@W[l]

            return A,S

        with torch.no_grad():
            A1,S1=factors(X1)
            A2,S2=(A1,S1) if X1 is X2 else factors(X2)

            n1,n2=X1.shape[0],X2.shape[0]
            m=f[-1].out_features

            K=torch.zeros(n1*m,n2*m,dtype=X1.dtype,device=X1.device)

            for i in range(0,n1,chunk_size):
                b=min(chunk_size,n1-i)
                block=K[i*m:(i+b)*m].view(b,m,n2,m)

                for l,(a1,a2,s1,s2) in enumerate(zip(A1,A2,S1,S2)):
                    c=a1[i:i+b]@a2.T+1

                    if l==len(A1)-1:
                        block.diagonal(dim1=1,dim2=3).add_(c.unsqueeze(-1))
                    else:
                        v=(s1[i:i+b].flatten(0,1)@s2.flatten(0,1).T).view(b,m,n2,m)
                        v.mul_(c[:,None,:,None])
                        block.add_(v)

            return K

    def structured(self,f:torch.nn.Sequential,X1:torch.Tensor,X2:torch.Tensor,chunk_size=64):
        layers=[layer for layer in f if isinstance(layer,torch.nn.Linear)]

        def factors(X):
            A,D=[],[]
            a=X

            for layer in f:
                if isinstance(layer,torch.nn.Linear):
                    A.append(a)
                    a=layer(a)
                    D.append(None)
                elif isinstance(layer,torch.nn.Tanh):
                    a=layer(a)
                    D[-1]=1-a.square()

            S=[None]*(len(layers)-1)
            s=layers[-1].weight.unsqueeze(0).expand(X.shape[0],-1,-1)

            for l in range(len(S)-1,-1,-1):
                if D[l] is not None:
                    s=s*D[l][:,None,:]
                S[l]=s
                if l:
                    s=s@layers[l].weight

            return A,S

        with torch.no_grad():
            A1,S1=factors(X1)
            A2,S2=(A1,S1) if X1 is X2 else factors(X2)

            C=[
                a1@a2.T+int(layer.bias is not None)
                for a1,a2,layer in zip(A1,A2,layers)
            ]

            n1,n2=X1.shape[0],X2.shape[0]
            m=layers[-1].out_features
            symmetric=X1 is X2

            K=torch.empty((n1*m,n2*m),device=X1.device,dtype=X1.dtype)

            bi=min(chunk_size,n1)
            bj=min(chunk_size,n2)

            tmp=torch.empty((bi*m,bj*m),device=X1.device,dtype=X1.dtype)

            for i in range(0,n1,chunk_size):
                ni=min(chunk_size,n1-i)

                for j in range(i if symmetric else 0,n2,chunk_size):
                    nj=min(chunk_size,n2-j)

                    dst=K[i*m:(i+ni)*m,j*m:(j+nj)*m]
                    block=dst.view(ni,m,nj,m)

                    if S1:
                        for l in range(len(S1)):
                            a=S1[l][i:i+ni].reshape(ni*m,-1)
                            b=S2[l][j:j+nj].reshape(nj*m,-1)

                            t=tmp[:ni*m,:nj*m]
                            torch.mm(a,b.T,out=t)

                            t.view(ni,m,nj,m).mul_(
                                C[l][i:i+ni,j:j+nj][:,None,:,None]
                            )

                            if l==0:
                                dst.copy_(t)
                            else:
                                dst.add_(t)
                    else:
                        dst.zero_()

                    block.diagonal(dim1=1,dim2=3).add_(
                        C[-1][i:i+ni,j:j+nj].unsqueeze(-1)
                    )

                    if symmetric and j>i:
                        K[j*m:(j+nj)*m,i*m:(i+ni)*m].copy_(dst.T)

            return K
    

    def spectrum(self,f,X):
        Q=X.shape[0]*f[-1].out_features
        P=sum(p.numel() for p in f.parameters())

        if P<Q:
            J=jac.Jac().jac_batch(f,X).detach()
            lam=torch.linalg.eigvalsh(J.T@J)
            zeros=lam.new_zeros(Q-P)
            return torch.cat((lam,zeros)).sort().values

        K=self.structured(f,X,X)
        return torch.linalg.eigvalsh(K)    
