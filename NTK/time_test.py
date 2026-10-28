import sys
import subprocess
import torch
import ntk
from time import perf_counter

def TIME_test():
    torch.manual_seed(0)
    torch.set_num_threads(8)

    M,N=100,100

    f=torch.nn.Sequential(
        torch.nn.Linear(M,M*2),torch.nn.Tanh(),
        torch.nn.Linear(M*2,M),torch.nn.Tanh(),
        torch.nn.Linear(M,M)
    ).double()

    X=torch.randn(N,M,dtype=torch.double)
    method=sys.argv[1]
    compute=getattr(ntk.NTK(),method)

    compute(f,X,X)

    start=perf_counter()
    K=compute(f,X,X)

    print(f"{method}: {perf_counter()-start:.4f} s")

def main():
    for method in ("direct_jac","layerwise","structured"):
        subprocess.run([sys.executable,__file__,method],check=True)

if __name__=="__main__":
    if len(sys.argv)>1:
        TIME_test()
    else:
        main()