"""
test_layernorm_bw.py

Tests rowBlock-based LayerNorm backward code.

We define:
 - custom() => LN forward + .backward (row-block LN kernel)
 - baseline() => row-wise LN forward/backward in Python (partial sums),
                 storing [dGamma, dBeta, dInp].

Then we compare the results.

Expect small numeric differences if the kernel uses a slightly different
"divide by (n) or (n-1)" logic in backward.
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

# Set up the minitorch backend
backend = minitorch.TensorBackend(CudaKernelOps)

# Load your .so that has row-block LN forward/backward
lib_layernorm = ctypes.CDLL("minitorch/cuda_kernels/layernorm_kernel.so")
print("lib_layernorm loaded from =", lib_layernorm._name)


@kt.case(atol=1e-3, rtol=1e-2, ntest=5)
def test_launch_layernorm_bw():
    """
    We'll do up to 5 random tests for LN backward.
    The 'hidden_dim' is chosen randomly by kt.hidden_dim, which depends on nhead, etc.
    """
    # 1) random shape from test_utils
    batch_size, seq_len = kt.bs_sl()
    bsz_seq = batch_size * seq_len
    hidden_dim = kt.hidden_dim
    print(f"(batch_token_num, hidden_dim): ({bsz_seq}, {hidden_dim})")

    # 2) random input/out_grad/gamma/beta
    ln_input = kt.rand((bsz_seq, hidden_dim))
    out_grad = kt.rand((bsz_seq, hidden_dim))
    gamma    = kt.rand((hidden_dim,))
    beta     = kt.rand((hidden_dim,))

    def custom():
        """
        custom => LN forward + .backward (rowBlock LN kernel).
                  We'll read [dGamma,dBeta,dInp] from the final grads.
        """
        # wrap minitorch Tensors
        inp_mt   = minitorch.tensor(ln_input.clone().tolist(), backend=backend, requires_grad=True)
        gamma_mt = minitorch.tensor(gamma.clone().tolist(),    backend=backend, requires_grad=True)
        beta_mt  = minitorch.tensor(beta.clone().tolist(),     backend=backend, requires_grad=True)

        # forward LN => rowBlock kernel
        out_mt = inp_mt.layernorm(gamma_mt, beta_mt)

        # out_grad
        out_grad_mt = minitorch.tensor(out_grad.clone().tolist(), backend=backend, requires_grad=True)

        start_time = time.time()
        # backward => triggers row-block LN backward kernel
        out_mt.backward(out_grad_mt)
        end_time = time.time()

        # read grads => torch Tensors
        d_inp   = torch.tensor(inp_mt.grad.to_numpy(),   dtype=torch.float32).cuda()
        d_gamma = torch.tensor(gamma_mt.grad.to_numpy(), dtype=torch.float32).cuda()
        d_beta  = torch.tensor(beta_mt.grad.to_numpy(),  dtype=torch.float32).cuda()

        return [d_gamma, d_beta, d_inp], end_time - start_time

    def baseline():
        """
        baseline => row-wise LN forward/backward in Python using partial sums.

        LN forward:
          mean  = sum(x)/d
          var   = sum((x-mean)^2)/(d-1)
          xhat  = (x-mean)/ sqrt(var + eps)
          out   = xhat* gamma + beta

        LN backward:
          dBeta   = sum(dOut)
          dGamma  = sum(dOut * xhat)
          dxhat   = dOut * gamma
          dX = (1/sqrt(var+eps)) * [ dxhat - mean(dxhat) - xhat*mean(dxhat*xhat) ]
        """
        # CPU copy
        x_np      = ln_input.cpu().numpy()     # shape(b,d)
        g_out_np  = out_grad.cpu().numpy()     # shape(b,d)
        gamma_np  = gamma.cpu().numpy()        # shape(d,)
        b, d      = x_np.shape

        start_time = time.time()

        # (A) LN forward row-wise
        sum_x  = x_np.sum(axis=1, keepdims=True)    # (b,1)
        mean_  = sum_x / d
        diff   = x_np - mean_
        sum_sq = (diff**2).sum(axis=1, keepdims=True)  # (b,1)
        var_   = sum_sq / (d-1)
        std_   = np.sqrt(var_ + kt.epsilon)
        xhat   = diff / std_

        # (B) LN backward
        # dBeta, dGamma
        dBeta  = g_out_np.sum(axis=0)  # shape(d,)
        dGamma = (g_out_np * xhat).sum(axis=0)

        # dxhat => out_grad * gamma
        dxhat  = g_out_np * gamma_np.reshape(1,d)

        # partial sums for LN
        sum_dxhat   = dxhat.sum(axis=1, keepdims=True)       # (b,1)
        sum_dxhat_x = (dxhat * xhat).sum(axis=1, keepdims=True)

        # dX => 1/std_ * [ dxhat - (1/d)*sum_dxhat - xhat*(1/d)*sum_dxhat_x ]
        dx = dxhat \
             - (1.0/d)* sum_dxhat \
             - xhat*(1.0/d)* sum_dxhat_x
        dx = dx / std_

        end_time = time.time()

        # Convert back
        d_inp   = torch.tensor(dx,      dtype=torch.float32).cuda()
        d_gamma = torch.tensor(dGamma, dtype=torch.float32).cuda()
        d_beta  = torch.tensor(dBeta,  dtype=torch.float32).cuda()

        return [d_gamma, d_beta, d_inp], end_time - start_time

    return custom, baseline


if __name__ == "__main__":
    kt.init(device='cuda:0', nhead=8)
    kt.run('test_launch_layernorm_bw')


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
# lib = ctypes.CDLL("minitorch/cuda_kernels/layernorm_kernel.so")
# print("lib_layernorm loaded from =", lib._name)

# @kt.case(atol=1e-3, rtol=1e-2, ntest=5)
# def test_launch_layernorm_bw():
#     batch_size, seq_len = kt.bs_sl()
#     bsz_seq = batch_size * seq_len
#     hidden_dim = kt.hidden_dim
#     print(
#         "(batch_token_num, hidden_dim): "
#         f"({bsz_seq}, {hidden_dim})"
#     )

#     ln_input = kt.rand((bsz_seq, hidden_dim))
#     out_grad = kt.rand((bsz_seq, hidden_dim))
#     gamma = kt.rand((hidden_dim))
#     beta = kt.rand((hidden_dim))

#     inp_numpy = ln_input.cpu().numpy()
#     var_mt = minitorch.tensor(inp_numpy.var(axis=1).tolist(), backend=backend)
#     means_mt = minitorch.tensor(inp_numpy.mean(axis=1).tolist(), backend=backend)
#     var = torch.tensor(var_mt._tensor._storage.astype(np.float32)).cuda()
#     mean = torch.tensor(means_mt._tensor._storage.astype(np.float32)).cuda()

#     def custom():
#       inp_mt = minitorch.tensor(ln_input.clone().tolist(), backend=backend, requires_grad=True)
#       gamma_mt = minitorch.tensor(gamma.clone().tolist(), backend=backend, requires_grad=True)
#       beta_mt = minitorch.tensor(beta.clone().tolist(), backend=backend, requires_grad=True)
#       out_mt = inp_mt.layernorm(gamma_mt, beta_mt)
#       out_grad_mt = minitorch.tensor(out_grad.clone().tolist(), backend=backend, requires_grad=True) 

#       start_time = time.time()
#       out_mt.backward(out_grad_mt)
#       end_time = time.time()
      
#       inp_grad = torch.tensor(inp_mt.grad.to_numpy(), dtype=torch.float32).cuda()
#       gamma_grad = torch.tensor(gamma_mt.grad.to_numpy(), dtype=torch.float32).cuda()
#       betta_grad = torch.tensor(beta_mt.grad.to_numpy(), dtype=torch.float32).cuda()
        
#       return [gamma_grad, betta_grad, inp_grad], end_time - start_time

#     def baseline():
#       f_input = minitorch.tensor(ln_input.clone().tolist(), backend=backend, requires_grad=True)
#       f_gamma = minitorch.tensor(gamma.clone().tolist(), backend=backend, requires_grad=True)
#       f_out_grad = minitorch.tensor(out_grad.clone().tolist(), backend=backend, requires_grad=True) 

#       start_time = time.time()

#       f_means = f_input.mean(dim=1)
#       f_vars = f_input.var(dim=1)
#       f_stds = minitorch.tensor(np.sqrt(f_vars.to_numpy()).reshape(-1, 1).tolist(), backend=backend, requires_grad=True)

#       xhat = (f_input - f_means) / f_stds
#       dxhat = f_out_grad * f_gamma
#       f_betta_grad = f_out_grad.sum(dim=0)
#       f_gamma_grad = (f_out_grad * xhat).sum(dim=0)
#       dinp = dxhat.sum(dim=1) + xhat * (dxhat * xhat).sum(dim=1)
#       dinp = dxhat - dinp / hidden_dim
#       dinp = dinp / f_stds

#       end_time = time.time()

#       inp_grad = torch.tensor(dinp.to_numpy(), dtype=torch.float32).cuda()
#       gamma_grad = torch.tensor(f_gamma_grad.to_numpy(), dtype=torch.float32).cuda()
#       betta_grad = torch.tensor(f_betta_grad.to_numpy(), dtype=torch.float32).cuda()

#       return kt.norm_res_list(gamma_grad, betta_grad, inp_grad), end_time - start_time

#     return custom, baseline


# kt.init(device='cuda:0', nhead=8)
# kt.run(
#   'test_launch_layernorm_bw'
# )