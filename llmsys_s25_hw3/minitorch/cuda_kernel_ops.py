from typing import Callable, Optional

from . import operators
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps
from .tensor_functions import tensor_from_numpy

import ctypes
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray  # <-- for allocating/copying on GPU
import torch

# Load the shared library
lib = ctypes.CDLL("minitorch/cuda_kernels/combine.so")
lib_softmax = ctypes.CDLL("minitorch/cuda_kernels/softmax_kernel.so")
lib_layernorm = ctypes.CDLL("minitorch/cuda_kernels/layernorm_kernel.so")

# lib_layernorm = ctypes.CDLL("minitorch/cuda_kernels/layernorm_debug.so")
datatype = np.float32

# function map
fn_map = {
  operators.add: 1,
  operators.mul: 2,
  operators.id: 3,
  operators.neg: 4,
  operators.lt: 5,
  operators.eq: 6,
  operators.sigmoid: 7,
  operators.relu: 8,
  operators.relu_back: 9,
  operators.log: 10,
  operators.log_back: 11,
  operators.exp: 12,
  operators.inv: 13,
  operators.inv_back: 14,
  operators.is_close: 15,
  operators.max: 16,
  operators.pow: 17, 
  operators.tanh: 18
}

THREADS_PER_BLOCK = 32

class CudaKernelOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"
        fn_id = fn_map[fn]

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            lib.tensorMap.argtypes = [
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),    # out_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_strides
                ctypes.c_int,                                                            # out_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),    # in_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_strides
                ctypes.c_int,                                                            # in_size
                ctypes.c_int,                                                            # shape_len
                ctypes.c_int,                                                            # fn_id
            ]

            lib.tensorMap.restype = None
            
            # assert out.size == a.size, f"zip {out.size}, {a.size}"

            lib.tensorMap(
                out._tensor._storage,
                out._tensor._shape.astype(np.int32),
                out._tensor._strides.astype(np.int32),
                out.size,
                a._tensor._storage,
                a._tensor._shape.astype(np.int32),
                a._tensor._strides.astype(np.int32),
                a.size,
                len(a.shape),
                fn_id
            )
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        fn_id = fn_map[fn]

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)

            lib.tensorZip.argtypes = [
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # out_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_strides
                ctypes.c_int,                                                            # out_size
                ctypes.c_int,                                                            # out_shape_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # a_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # a_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # a_strides
                ctypes.c_int,                                                            # a_size
                ctypes.c_int,                                                            # a_shape_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),    # b_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # b_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # b_strides
                ctypes.c_int,                                                            # b_size
                ctypes.c_int,                                                            # b_shape_size
                ctypes.c_int,                                                            # fn_id
            ]

            lib.tensorZip.restype = None

            # assert out.size == a.size, f"zip {out.size}, {a.size}"
            # assert out.size == b.size, f"zip {out.size}, {b.size}"

            lib.tensorZip(
                out._tensor._storage,
                out._tensor._shape.astype(np.int32),
                out._tensor._strides.astype(np.int32),
                out.size,
                len(out.shape),
                a._tensor._storage,
                a._tensor._shape.astype(np.int32),
                a._tensor._strides.astype(np.int32),
                a.size,
                len(a.shape),
                b._tensor._storage,
                b._tensor._shape.astype(np.int32),
                b._tensor._strides.astype(np.int32),
                b.size,
                len(b.shape),
                fn_id
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0) -> Callable[[Tensor, int], Tensor]:
        fn_id = fn_map[fn]

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out = a.zeros(tuple(out_shape))

            lib.tensorReduce.argtypes = [
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # out_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_strides
                ctypes.c_int,                                                            # out_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # in_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_strides
                ctypes.c_int,                                                            # reduce_dim
                ctypes.c_double,                                                         # reduce_value
                ctypes.c_int,                                                            # shape_len
                ctypes.c_int,                                                            # fn_id
            ]

            lib.tensorReduce.restype = None

            lib.tensorReduce(
                out._tensor._storage,
                out._tensor._shape.astype(np.int32),
                out._tensor._strides.astype(np.int32),
                out.size,
                a._tensor._storage,
                a._tensor._shape.astype(np.int32),
                a._tensor._strides.astype(np.int32),
                dim,
                start,
                len(a.shape),
                fn_id
            )

            return out

        return ret

    @staticmethod
    def matrix_multiply_cublas(a: Tensor, b: Tensor) -> Tensor:
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]

        if len(a.shape) > 3:
            a = a.contiguous().view(np.prod(a.shape[:-2]), a.shape[-2],
                                    a.shape[-1])
        if len(b.shape) > 3:
            b = b.contiguous().view(np.prod(b.shape[:-2]), b.shape[-2],
                                    b.shape[-1])
        assert a.shape[0] == b.shape[0]

        bs, m, n, k = a.shape[0], a.shape[1], b.shape[2], a.shape[2]
        A, B = a.to_numpy(), b.to_numpy()

        # Convert A and B to column-major order
        A_fortran = np.transpose(A, (0, 2, 1))
        B_fortran = np.transpose(B, (0, 2, 1))

        # Flatten A and B for sending to GPU
        A_flat = A_fortran.reshape(bs, -1)
        B_flat = B_fortran.reshape(bs, -1)

        # Allocate memory on GPU
        A_gpu = cuda.mem_alloc(A_flat.nbytes)
        B_gpu = cuda.mem_alloc(B_flat.nbytes)
        C_gpu = cuda.mem_alloc(bs * m * n * A.itemsize)

        # Copy data to GPU
        cuda.memcpy_htod(A_gpu, A_flat)
        cuda.memcpy_htod(B_gpu, B_flat)

        # Prepare arrays of pointers
        A_gpu_ptrs = np.array(
            [int(A_gpu) + i * m * k * A.itemsize for i in range(bs)],
            dtype=np.uint64)
        B_gpu_ptrs = np.array(
            [int(B_gpu) + i * k * n * B.itemsize for i in range(bs)],
            dtype=np.uint64)
        C_gpu_ptrs = np.array(
            [int(C_gpu) + i * m * n * A.itemsize for i in range(bs)],
            dtype=np.uint64)

        # Allocate device memory for arrays of pointers
        A_array_gpu = cuda.mem_alloc(A_gpu_ptrs.nbytes)
        B_array_gpu = cuda.mem_alloc(B_gpu_ptrs.nbytes)
        C_array_gpu = cuda.mem_alloc(C_gpu_ptrs.nbytes)

        # Copy arrays of pointers to device memory
        cuda.memcpy_htod(A_array_gpu, A_gpu_ptrs)
        cuda.memcpy_htod(B_array_gpu, B_gpu_ptrs)
        cuda.memcpy_htod(C_array_gpu, C_gpu_ptrs)

        # Set argument types for the kernel function
        lib_mm.batchedMatMulKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int]

        # Launch kernel
        lib_mm.batchedMatMulKernel(
            int(A_array_gpu), int(B_array_gpu), int(C_array_gpu), m, k, n, bs)

        # Synchronize device to ensure computation is complete
        cuda.Context.synchronize()

        # Copy back the result
        C = np.empty((bs, n, m), dtype=A.dtype)
        cuda.memcpy_dtoh(C, C_gpu)
        C = np.transpose(C, (0, 2, 1))

        c = tensor_from_numpy(
            np.ascontiguousarray(C),
            backend=a.backend, requires_grad=a.requires_grad()).contiguous()

        # Undo 3d if we added it.
        if both_2d:
            c = c.view(c.shape[1], c.shape[2])
        if len(ls) > 3:
            c = c.view(*ls)
        return c

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # handle cases with more dimensions [64, 4, 32, 128] x [64, 4, 128, 32]
        more_3d = False
        if len(out.shape) > 3:
            # print(f"Debug in matmul: output shape {ls}")
            more_3d = True
            out = out.view(np.prod(out.shape[:-2]), out.shape[-2], out.shape[-1])
            nshape = out._tensor._shape
            nstrides = out._tensor._strides
            # print(f"Debug in matmul: batched dim [:-2] and get the strides {nshape, nstrides}")
        if len(a.shape) > 3:
            a = a.contiguous().view(np.prod(a.shape[:-2]), a.shape[-2], a.shape[-1])
        if len(b.shape) > 3:
            b = b.contiguous().view(np.prod(b.shape[:-2]), b.shape[-2], b.shape[-1])
        
        assert a.shape[0] == b.shape[0]
        assert a.shape[0] == out.shape[0]

        lib.MatrixMultiply.argtypes = [
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # out_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # out_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # out_strides
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # a_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # a_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # a_strides
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # b_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # b_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # b_strides
            ctypes.c_int,                                                             # batch_size
            ctypes.c_int,                                                             # out_shape[1], m
            ctypes.c_int                                                              # out_shape[2], p
        ]

        lib.MatrixMultiply.restype = None

        assert len(out._tensor._shape) == 3, f"{len(out._tensor._shape)}"
        assert len(out._tensor._strides) == 3, f"{len(out._tensor._strides)}"
        assert len(a._tensor._shape) == 3
        assert len(a._tensor._strides) == 3
        assert len(b._tensor._shape) == 3
        assert len(b._tensor._strides) == 3

        lib.MatrixMultiply(
            out._tensor._storage,
            out._tensor._shape.astype(np.int32),
            out._tensor._strides.astype(np.int32),
            a._tensor._storage,
            a._tensor._shape.astype(np.int32),
            a._tensor._strides.astype(np.int32),
            b._tensor._storage,
            b._tensor._shape.astype(np.int32),
            b._tensor._strides.astype(np.int32),
            a.shape[0],
            a.shape[1],
            b.shape[2]
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        if more_3d:
            out = out.view(*ls)
            # print(f"Debug in matmul: output shape {out.shape}")
        return out

    @staticmethod
    def attn_softmax_fw(inp: Tensor, mask: Tensor):
        """
        Calls our fused CUDA kernel that does in-place softmax on `inp`,
        optionally adding `mask` (e.g. attention mask). Returns `inp`.
        """
        # Example shape = [batch_size, nhead, from_len, to_len]
        batch_size, nhead, from_len, to_len = inp.shape
        
        # The kernel uses a boolean for "is_dec_self_attn" or "mask_future".
        # For simplicity here, we set is_dec_self_attn=False
        is_dec_self_attn = False
        
        # Get the current CUDA stream (pytorch) 
        stream = torch.cuda.current_stream().cuda_stream

        # Set up the argtypes for the function call
        lib_softmax.launch_attn_softmax.argtypes = [
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # inp data
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # mask data
            ctypes.c_int,  # batch_size
            ctypes.c_int,  # nhead
            ctypes.c_int,  # from_len
            ctypes.c_int,  # to_len
            ctypes.c_bool, # is_dec_self_attn
            ctypes.c_void_p  # cuda_stream
        ]
        lib_softmax.launch_attn_softmax.restype = None

        # Call the kernel
        lib_softmax.launch_attn_softmax(
            inp._tensor._storage,
            mask._tensor._storage,
            batch_size,
            nhead,
            from_len,
            to_len,
            is_dec_self_attn,
            stream
        )
        return inp

    @staticmethod
    def attn_softmax_bw(out_grad: Tensor, soft_inp: Tensor):
        """
        Calls our fused CUDA kernel that computes dSoftmax = Softmax * (dOut - sum(...))
        in-place on out_grad.
        
        We assume:
         - out_grad shape = [batch_size, nhead, from_len, to_len]
         - soft_inp shape = same, containing the *forward result* of softmax.
        The kernel overwrites out_grad with the correct d_inp for each row.
        
        Returns out_grad, which is now the gradient wrt the softmax input.
        """
        batch_size, nhead, from_len, to_len = out_grad.shape
        rows = batch_size * nhead * from_len  # number of rows
        stream = torch.cuda.current_stream().cuda_stream

        lib_softmax.launch_attn_softmax_bw.argtypes = [
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'), # out_grad
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'), # soft_inp
            ctypes.c_int,   # rows
            ctypes.c_int,   # softmax_len
            ctypes.c_void_p # cuda stream
        ]
        lib_softmax.launch_attn_softmax_bw.restype = None

        # Call the backward kernel
        lib_softmax.launch_attn_softmax_bw(
            out_grad._tensor._storage,
            soft_inp._tensor._storage,
            rows,
            to_len,
            stream
        )
        return out_grad


    # --------------------------------------------------------------------------
    # rowBlock LN forward
    # --------------------------------------------------------------------------
    @staticmethod
    def layernorm_fw(inp: Tensor, gamma: Tensor, beta: Tensor):
        """
        rowBlock LN forward
        shape: (batch_size, width)
        We'll call the debug LN kernel => launch_layernorm
        returning (ln_out, saved_var, saved_mean).
        """
        import pycuda.gpuarray as gpuarray
        import pycuda.driver as cuda

        batch_size, width = inp.shape

        inp_host   = inp.to_numpy()
        gamma_host = gamma.to_numpy()
        beta_host  = beta.to_numpy()

        # GPU copies
        inp_gpu   = gpuarray.to_gpu(inp_host.astype(np.float32, copy=False))
        gamma_gpu = gpuarray.to_gpu(gamma_host.astype(np.float32, copy=False))
        beta_gpu  = gpuarray.to_gpu(beta_host.astype(np.float32, copy=False))

        out_gpu   = gpuarray.zeros((batch_size, width), dtype=datatype)
        var_gpu   = gpuarray.zeros((batch_size,),       dtype=datatype)
        mean_gpu  = gpuarray.zeros((batch_size,),       dtype=datatype)

        stream_ptr = torch.cuda.current_stream().cuda_stream

        lib_layernorm.launch_layernorm(
            ctypes.c_void_p(int(out_gpu.ptr)),
            ctypes.c_void_p(int(var_gpu.ptr)),
            ctypes.c_void_p(int(mean_gpu.ptr)),
            ctypes.c_void_p(int(inp_gpu.ptr)),
            ctypes.c_void_p(int(gamma_gpu.ptr)),
            ctypes.c_void_p(int(beta_gpu.ptr)),
            ctypes.c_int(batch_size),
            ctypes.c_int(width),
            ctypes.c_void_p(stream_ptr)
        )
        cuda.Context.synchronize()

        out_host  = out_gpu.get()
        var_host  = var_gpu.get()
        mean_host = mean_gpu.get()

        # Return as Tensors
        out_t  = tensor_from_numpy(out_host,  backend=inp.backend,
                                   requires_grad=inp.requires_grad())
        var_t  = tensor_from_numpy(var_host,  backend=inp.backend, requires_grad=False)
        mean_t = tensor_from_numpy(mean_host, backend=inp.backend, requires_grad=False)
        return out_t, var_t, mean_t

    # --------------------------------------------------------------------------
    # rowBlock LN backward
    # --------------------------------------------------------------------------
    @staticmethod
    def layernorm_bw(
        out_grad: Tensor,
        inp: Tensor,
        gamma: Tensor,
        beta: Tensor,
        var: Tensor,
        mean: Tensor
    ):
        """
        rowBlock LN backward. We again call the “launch_layernorm_bw”
        which requires hidden_dim = width//4 if the kernel expects float4.
        BUT here we've changed the kernel to rowBlock style, so we do hidden_dim=width.

        For the sake of final code, let's assume it needs width//4 if it's still float4 internally.
        If your rowBlock code uses “(blockDim=width)”, then pass just (width).
        """
        import pycuda.gpuarray as gpuarray
        import pycuda.driver as cuda

        batch_size, width = out_grad.shape

        # Copy to GPU
        og_host   = out_grad.to_numpy()
        inp_host  = inp.to_numpy()
        gm_host   = gamma.to_numpy()
        bt_host   = beta.to_numpy()
        var_host  = var.to_numpy()
        mean_host = mean.to_numpy()

        og_gpu   = gpuarray.to_gpu(og_host.astype(np.float32, copy=False))
        inp_gpu  = gpuarray.to_gpu(inp_host.astype(np.float32, copy=False))
        gm_gpu   = gpuarray.to_gpu(gm_host.astype(np.float32, copy=False))
        bt_gpu   = gpuarray.to_gpu(bt_host.astype(np.float32, copy=False))
        var_gpu  = gpuarray.to_gpu(var_host.astype(np.float32, copy=False))
        mean_gpu = gpuarray.to_gpu(mean_host.astype(np.float32, copy=False))

        d_inp_gpu   = gpuarray.zeros((batch_size, width), dtype=datatype)
        d_gamma_gpu = gpuarray.zeros((width,), dtype=datatype)
        d_beta_gpu  = gpuarray.zeros((width,), dtype=datatype)

        stream_ptr_1 = torch.cuda.current_stream().cuda_stream
        stream_ptr_2 = torch.cuda.current_stream().cuda_stream

        lib_layernorm.launch_layernorm_bw(
            ctypes.c_void_p(int(d_gamma_gpu.ptr)),
            ctypes.c_void_p(int(d_beta_gpu.ptr)),
            ctypes.c_void_p(int(d_inp_gpu.ptr)),
            ctypes.c_void_p(int(og_gpu.ptr)),
            ctypes.c_void_p(int(inp_gpu.ptr)),
            ctypes.c_void_p(int(gm_gpu.ptr)),
            ctypes.c_void_p(int(bt_gpu.ptr)),
            ctypes.c_void_p(int(var_gpu.ptr)),
            ctypes.c_void_p(int(mean_gpu.ptr)),
            ctypes.c_int(batch_size),
            ctypes.c_int(width),   # rowBlock LN => pass width
            ctypes.c_void_p(stream_ptr_1),
            ctypes.c_void_p(stream_ptr_2)
        )
        cuda.Context.synchronize()

        d_inp_host   = d_inp_gpu.get()
        d_gamma_host = d_gamma_gpu.get()
        d_beta_host  = d_beta_gpu.get()

        # Convert back to minitorch Tensors
        d_inp   = tensor_from_numpy(d_inp_host,   backend=out_grad.backend)
        d_gamma = tensor_from_numpy(d_gamma_host, backend=out_grad.backend)
        d_beta  = tensor_from_numpy(d_beta_host,  backend=out_grad.backend)
        return d_inp, d_gamma, d_beta