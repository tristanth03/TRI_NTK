import sys
import subprocess
import psutil
import torch
import ntk
from memory_profiler import memory_usage

def RAM_test():
    torch.manual_seed(0)
    torch.set_num_threads(8)

    M,N=100,100

    f=torch.nn.Sequential(
        torch.nn.Linear(M,M),torch.nn.Tanh(),
        torch.nn.Linear(M,M),torch.nn.Tanh(),
        torch.nn.Linear(M,M)
    ).double()

    X=torch.randn(N,M,dtype=torch.double)
    method=sys.argv[1]

    baseline=psutil.Process().memory_info().rss/2**20

    peak=memory_usage(
        (getattr(ntk.NTK(),method),(f,X,X)),
        interval=0.01,
        max_usage=True
    )

    print(f"{method}:")
    print(f"Peak RAM: {peak:.2f} MiB")
    print(f"Additional RAM: {peak-baseline:.2f} MiB")

def main():
    for method in ("direct_jac","layerwise","structured"):
        subprocess.run([sys.executable,__file__,method],check=True)

if __name__=="__main__":
    if len(sys.argv)>1:
        RAM_test()
    else:
        main()