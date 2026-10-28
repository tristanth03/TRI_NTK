import ntk
import torch
from time import perf_counter
from memory_profiler import memory_usage


def RAM_test():
    




def main():
    torch.manual_seed(0)
    M = 100
    N = 100
    f = torch.nn.Sequential(torch.nn.Linear(M,M),torch.nn.Tanh(),
                            torch.nn.Linear(M,M),torch.nn.Tanh(),
                            torch.nn.Linear(M,M))

    X = torch.randn(N,M)
    X_ = torch.randn(N*2,M)
    NTK = ntk.NTK()

    torch.set_num_threads(12)

    start = perf_counter()
    K2 = NTK.structured(f,X,X_)
    print("Kernel:",perf_counter()-start)

    start = perf_counter()
    K1 = NTK.layerwise(f,X,X_)
    print("Kernel:",perf_counter()-start)

    start = perf_counter()
    K = NTK.direct_jac(f,X,X_)
    print("Kernel:",perf_counter()-start)

    print((K-K1).abs().max().item())
    print((K-K2).abs().max().item())

    print(torch.linalg.vector_norm(K1-K2)/torch.linalg.vector_norm(K1))

if __name__ == "__main__":
    main()