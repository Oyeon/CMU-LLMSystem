import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent   # …/llmsys_s25_hw3
if str(ROOT) not in sys.path:                          # put project root on sys.path
    sys.path.insert(0, str(ROOT))


"""
test_layernorm_fw.py

Tests a row-block LN kernel in minitorch (forward only).

We define:
 - custom() => uses minitorch.layernorm(...) (your kernel)
 - baseline() => manual LN forward in Python with sample variance

We compare custom vs baseline. This version prints partial logs from baseline
so you can see if the data or variance is degenerate or not.
"""

import ctypes
import numpy as np
import time

import torch
import torch.nn.functional as F

from pycuda import gpuarray, driver
import pycuda.autoinit

from test_utils import TestDecorator
kt = TestDecorator()

import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps
import os

backend = minitorch.TensorBackend(CudaKernelOps)

# Load the row-block LN .so
lib_layernorm = ctypes.CDLL("minitorch/cuda_kernels/layernorm_kernel.so")
print("lib_layernorm loaded from =", lib_layernorm._name)


@kt.case(atol=1e-2, rtol=1e-3, ntest=5)
def test_launch_layernorm():
    # 1) random shape
    batch_size, seq_len = kt.bs_sl()
    bsz_seq = batch_size * seq_len
    hidden_dim = kt.hidden_dim
    print(f"(batch_token_num, hidden_dim): ({bsz_seq}, {hidden_dim})")

    # 2) random input
    inp   = kt.rand((bsz_seq, hidden_dim))
    gamma = kt.rand((hidden_dim,))
    beta  = kt.rand((hidden_dim,))

    def custom():
        """
        custom => minitorch.layernorm(...) => calls your row-block LN kernel
        """
        inp_mt   = minitorch.tensor(inp.clone().tolist(),   backend=backend, requires_grad=True)
        gamma_mt = minitorch.tensor(gamma.clone().tolist(), backend=backend, requires_grad=True)
        beta_mt  = minitorch.tensor(beta.clone().tolist(),  backend=backend, requires_grad=True)

        start_time = time.time()
        out_mt = inp_mt.layernorm(gamma_mt, beta_mt)
        end_time = time.time()

        out = torch.tensor(out_mt._tensor._storage).cuda()
        return [out], end_time - start_time

    def baseline():
        """
        baseline => row-wise LN:
          1) sum => mean
          2) sum of squares => sample variance => / (d - 1)
          3) LN => (x - mean)/ sqrt(var+eps)
          4) multiply by gamma, add beta
        Also logs partial results to see if data or var is broken.
        """
        inp_mt   = minitorch.tensor(inp.clone().tolist(), backend=backend, requires_grad=True)
        gamma_mt = minitorch.tensor(gamma.clone().tolist(), backend=backend, requires_grad=True)
        beta_mt  = minitorch.tensor(beta.clone().tolist(),  backend=backend, requires_grad=True)

        start_time = time.time()

        x = inp_mt.contiguous()
        b, d = x.shape

        print(">> Baseline: x first 25 elems =", x._tensor._storage[:25])

        # 1) sum => mean (sample mean)
        sum_x = x.sum(dim=1)  # shape: [b]
        mean_ = sum_x / float(d)
        mean_ = mean_.view(b, 1)

        print(">> Baseline: mean_ first 5 elems =", mean_._tensor._storage[:5])

        # 2) sum of squares => var
        diff   = x - mean_
        sqdiff = diff * diff
        sum_sq = sqdiff.sum(dim=1)  # shape: [b]
        var_   = sum_sq / float(d - 1)  # sample var
        var_   = var_.view(b, 1)

        print(">> Baseline: var_ first 5 elems =", var_._tensor._storage[:5])

        # 3) LN => (x - mean)/ sqrt(var + eps)
        x = (x - mean_) / ((var_ + kt.epsilon)**0.5)
        print(">> LN x first 25 elems =", x._tensor._storage[:25])

        # 4) gamma,beta
        x = gamma_mt * x + beta_mt

        end_time = time.time()
        base = torch.tensor(x._tensor._storage).cuda()
        return [base.contiguous()], end_time - start_time

    return custom, baseline


if __name__ == "__main__":
    kt.init(device='cuda:0', nhead=8)
    kt.run('test_launch_layernorm')


# import ctypes
# import numpy as np
# import time

# import torch
# import torch.nn.functional as F

# from pycuda import gpuarray, driver
# import pycuda.autoinit

# from test_utils import TestDecorator
# kt = TestDecorator()

# import minitorch
# from minitorch.cuda_kernel_ops import CudaKernelOps
# import os
# backend = minitorch.TensorBackend(CudaKernelOps)

# # Load the shared library
# # lib = ctypes.CDLL("minitorch/cuda_kernels/layernorm_kernel.so")
# lib_layernorm = ctypes.CDLL("minitorch/cuda_kernels/layernorm_debug.so")
# print("lib_layernorm loaded from =", lib_layernorm._name)


# @kt.case(atol=1e-2, rtol=1e-3, ntest=5)
# def test_launch_layernorm():
#     batch_size, seq_len = kt.bs_sl()
#     bsz_seq = batch_size * seq_len
#     hidden_dim = kt.hidden_dim
#     print(
#         "(batch_token_num, hidden_dim): "
#         f"({bsz_seq}, {hidden_dim})"
#     )
    
#     custom_res = kt.rand((bsz_seq, hidden_dim))
#     inp = kt.rand((bsz_seq, hidden_dim))
#     gamma = kt.rand((hidden_dim))
#     beta = kt.rand((hidden_dim))
#     var = kt.rand((bsz_seq))
#     means = kt.rand((bsz_seq))
    
#     def custom():
#       inp_mt = minitorch.tensor(inp.clone().tolist(), backend=backend, requires_grad=True)
#       gamma_mt = minitorch.tensor(gamma.clone().tolist(), backend=backend, requires_grad=True)
#       beta_mt = minitorch.tensor(beta.clone().tolist(), backend=backend, requires_grad=True)

#       start_time = time.time()
#       out_mt = inp_mt.layernorm(gamma_mt, beta_mt)
#       end_time = time.time()
#       out = torch.tensor(out_mt._tensor._storage).cuda()
#       return [out], end_time - start_time
    
#     def baseline():
#       inp_mt = minitorch.tensor(inp.clone().tolist(), backend=backend, requires_grad=True)
#       gamma_mt = minitorch.tensor(gamma.clone().tolist(), backend=backend, requires_grad=True)
#       beta_mt = minitorch.tensor(beta.clone().tolist(), backend=backend, requires_grad=True)

#       start_time = time.time()

#       x = inp_mt.contiguous()
#       batch, dim = x.shape

#       mean = x.mean(dim=1).view(batch, 1)
#       variance = x.var(dim=1).view(batch, 1)
#       x = (x - mean) / ((variance + kt.epsilon) ** 0.5)
#       x = gamma_mt * x + beta_mt
#       end_time = time.time()

#       base = torch.tensor(x._tensor._storage).cuda()
#       return [
#           base.contiguous(),
#       ], end_time - start_time
    
#     return custom, baseline


# kt.init(device='cuda:0', nhead=8)
# kt.run(
#   'test_launch_layernorm'
# )