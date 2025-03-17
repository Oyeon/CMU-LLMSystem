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


lib_layernorm = ctypes.CDLL("minitorch/cuda_kernels/layernorm_kernel.so")
print("lib_layernorm loaded from =", lib_layernorm._name)


@kt.case(atol=1e-2, rtol=1e-3, ntest=5)
def test_launch_layernorm():
    # 1) 
    batch_size, seq_len = kt.bs_sl()
    bsz_seq = batch_size * seq_len
    hidden_dim = kt.hidden_dim
    print(f"(batch_token_num, hidden_dim): ({bsz_seq}, {hidden_dim})")

    # 2) 
    inp   = kt.rand((bsz_seq, hidden_dim))
    gamma = kt.rand((hidden_dim,))
    beta  = kt.rand((hidden_dim,))

    def custom():
        """custom() => minitorch .layernorm(...) """
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
        baseline() =>  mean,var (row-wise),
                      LN => (x - mean)/sqrt(var + eps),
                      gamma,beta 
        """
        inp_mt   = minitorch.tensor(inp.clone().tolist(), backend=backend, requires_grad=True)
        gamma_mt = minitorch.tensor(gamma.clone().tolist(), backend=backend, requires_grad=True)
        beta_mt  = minitorch.tensor(beta.clone().tolist(),  backend=backend, requires_grad=True)

        start_time = time.time()

        x = inp_mt.contiguous()
        b, d = x.shape

        print(">> Baseline: x first 25 elems =", x._tensor._storage[:25])

        # 1) sum => mean
        sum_x = x.sum(dim=1)  # shape: [b]
        mean_ = sum_x / float(d)  # sample mean
        mean_ = mean_.view(b, 1)

        print(">> Baseline: mean_ first 5 elems =", mean_._tensor._storage[:5])

        # 2) sum of squares => var
        diff   = x - mean_
        sqdiff = diff * diff
        sum_sq = sqdiff.sum(dim=1)   # shape: [b]
        # sample variance => / (d-1)
        var_   = sum_sq / float(d - 1)
        var_   = var_.view(b, 1)

        print(">> Baseline: var_ first 5 elems =", var_._tensor._storage[:5])

        # 3) LN => (x - mean)/sqrt(var+eps)
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


# import os
# import ctypes
# import numpy as np
# import pycuda.autoinit
# import pycuda.driver as cuda
# import torch

# import minitorch
# from minitorch.tensor_functions import tensor
# from minitorch.cuda_kernel_ops import CudaKernelOps

# # This debug .so has your device-side prints in the LN kernel
# lib_layernorm = ctypes.CDLL("minitorch/cuda_kernels/layernorm_debug.so")
# print("lib_layernorm loaded from =", lib_layernorm._name)

# backend = minitorch.TensorBackend(CudaKernelOps)

# def tiny_debug_test():
#     """
#     1) shape=(1,4) => hidden_dim_float4=1
#     2) no partial indexing => we do out.view(4) to produce a (4,) shape
#     3) var,mean each have shape (1,) => we do .view(()) to create a scalar
#     """

#     print("\n[DEBUG] Doing a TINY LN test: shape=(1,4) so hidden_dim_float4=1")

#     batch_size, width = 1, 4
#     inp_data   = np.random.randn(batch_size, width).astype(np.float32)
#     gamma_data = np.random.randn(width).astype(np.float32)
#     beta_data  = np.random.randn(width).astype(np.float32)

#     print("[DEBUG] CPU inp_data:\n",  inp_data)
#     print("[DEBUG] CPU gamma_data:\n", gamma_data)
#     print("[DEBUG] CPU beta_data:\n",  beta_data)

#     # Convert to nested lists so shape is preserved by `tensor(...)`
#     inp_list   = inp_data.tolist()  
#     gamma_list = gamma_data.tolist()
#     beta_list  = beta_data.tolist()

#     inp   = tensor(inp_list,   backend=backend, requires_grad=False)
#     gamma = tensor(gamma_list, backend=backend, requires_grad=False)
#     beta  = tensor(beta_list,  backend=backend, requires_grad=False)

#     print("[DEBUG] inp.shape   =", inp.shape)   # (1,4)
#     print("[DEBUG] gamma.shape =", gamma.shape) # (4,)
#     print("[DEBUG] beta.shape  =", beta.shape)  # (4,)

#     # Actually call LN => calls your debug kernel in layernorm_debug.so
#     print("[DEBUG] Now calling LN forward on shape=(1,4). Expect hidden_dim=1 in kernel.")
#     out = inp.layernorm(gamma, beta)

#     # out has shape (1,4), var/mean each have shape (1,)
#     # If we do out[0], that triggers partial indexing error. Instead:
#     #  - reshape out to (4,)
#     #  - reshape var to () for a scalar, or keep shape(1,) and do var[0]
#     out_1d  = out.view(width)   # from (1,4) -> (4,)
#     # var_1d  = var.view(1)       # from (1,)  -> (1,) still, so we can do var_1d[0]
#     # mean_1d = mean.view(1)      # from (1,)  -> (1,)

#     # Alternatively, we can do var_0d=var.view(()) => shape=()
#     # then do var_0d.item()

#     print("\n[DEBUG] LN Output shape =>", out_1d.shape)
#     print("[DEBUG] LN Output =>", out_1d)       # no partial indexing error
#     # print("[DEBUG] LN Var => shape", var_1d.shape, "value =>", var_1d[0])  # (1,) => index [0]
#     # print("[DEBUG] LN Mean=> shape", mean_1d.shape,"value =>", mean_1d[0]) # or .item()

# if __name__ == "__main__":
#     os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#     os.environ["PYTHONUNBUFFERED"] = "1"

#     tiny_debug_test()
