import ntk
import torch
from time import perf_counter
from memory_profiler import memory_usage


    

def main():
    torch.manual_seed(0)
    M = 10
    N = 1000
    f = torch.nn.Sequential(torch.nn.Linear(M,M),torch.nn.Tanh(),
                            torch.nn.Linear(M,M),torch.nn.Tanh(),
                            torch.nn.Linear(M,M)).double()

    X = torch.randn(N,M,dtype=torch.double)
    X_ = X
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

    start = perf_counter()
    eig2 = NTK.spectrum(f,X)
    print("Eig via spectrum:",perf_counter()-start)
    print(eig2[0],eig2[-1])

    start = perf_counter()
    eig = torch.linalg.eigvalsh(K)
    print("Eig via eigvalsh:",perf_counter()-start)
    print(eig[0],eig[-1])

if __name__ == "__main__":
    main()