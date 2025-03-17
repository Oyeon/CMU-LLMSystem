"""
Implementation of the autodifferentiation Functions for Tensor.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING
import numpy as np
import copy

import minitorch
from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple, Union
    from .tensor import Tensor
    from .tensor_data import (
        UserIndex,
        UserShape,
        Storage,
        OutIndex,
        Index,
        Shape,
        Strides
    )

datatype = np.float32
datasize = 4


def wrap_tuple(x):  # type: ignore
    """Turn a possible value into a tuple."""
    if isinstance(x, tuple):
        return x
    return (x,)


# -----------------------------
# Base class for autodiff
# -----------------------------
class Function:
    """
    Base class for all autodifferentiation Functions.
    Each operation in the computational graph is a subclass of Function.
    """

    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """
        Applies this function to the given input Tensors.

        1. Detach the inputs (no gradients).
        2. Save the need_grad status.
        3. Create a context if gradients are needed.
        4. Call the forward pass.
        5. Wrap the result into a new Tensor that has a proper History (if needed).
        """
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        ctx = Context(not need_grad)
        c = cls._forward(ctx, *raw_vals)

        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


# -----------------------------
# Basic arithmetic operations
# -----------------------------
class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """
        Forward pass for Negation: out = -t1
        """
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """
        Backward pass for Negation: d(-t1)/dt1 = -1
        """
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """
        Forward pass for Inversion: out = 1 / t1
        """
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """
        Backward pass for Inversion:
        d(1/t1)/dt1 = -1 / (t1^2)
        """
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """
        Forward pass for Addition: out = t1 + t2
        """
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Backward pass for Addition: 
        d(t1 + t2)/dt1 = 1, d(t1 + t2)/dt2 = 1
        """
        return grad_output, grad_output


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """
        Forward pass for Multiplication: out = a * b
        """
        ctx.save_for_backward(a, b)
        return a.f.mul_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Backward pass for Multiplication:
        d(a * b)/da = b, d(a * b)/db = a
        """
        a, b = ctx.saved_values
        return (
            grad_output.f.mul_zip(b, grad_output),
            grad_output.f.mul_zip(a, grad_output),
        )


# -----------------------------
# Power Scalar
# -----------------------------
class PowerScalar(Function):
    """
    Computes a^exponent by:
      a^exponent = exp(exponent * log(a))
    """

    @staticmethod
    def forward(ctx: Context, a: Tensor, scalar: Tensor) -> Tensor:
        """
        Forward pass: a^exponent
        """
        exponent = float(scalar.item())
        ctx.save_for_backward(a, exponent)

        # out = e^( exponent * log(a) )
        log_a = a.f.log_map(a)
        exp_val = log_a * exponent   # uses "mul_zip"
        out = exp_val.f.exp_map(exp_val)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """
        Backward pass for PowerScalar: d(a^exponent)/da = exponent * a^(exponent - 1)
        We return (grad_for_input, 0.0) because exponent is a constant scalar with no gradient.
        """
        a, exponent = ctx.saved_values
        e_minus_1 = exponent - 1.0

        # a^(exponent - 1) = e^((exponent - 1)*log(a))
        log_a = a.f.log_map(a)
        mul_val = log_a * e_minus_1
        a_pow_e_minus_1 = mul_val.f.exp_map(mul_val)

        # multiply by exponent
        e_tensor = a.zeros(a.shape) + exponent
        grad_a   = a_pow_e_minus_1.f.mul_zip(a_pow_e_minus_1, e_tensor)

        # chain rule with grad_output
        grad_a = grad_a.f.mul_zip(grad_a, grad_output)

        # No gradient for the scalar => 0.0
        return (grad_a, 0.0)


# -----------------------------
# Activation Functions
# -----------------------------
class Tanh(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor) -> Tensor:
        """
        Forward pass for Tanh: out = tanh(input).
        """
        out = input.f.tanh_map(input)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """
        Backward pass: d(tanh(x))/dx = (1 - tanh^2(x)) * grad_output.
        """
        (out,) = ctx.saved_values

        # 1) Generate a tensor full of 1.0 with the same shape as 'out'
        one = out.zeros(out.shape) + 1.0

        # 2) Compute (1 - out^2)
        #    => out^2 = out.f.mul_zip(out, out)
        out_sq = out.f.mul_zip(out, out)
        diff   = one - out_sq  # Tensor - Tensor OK

        # 3) Multiply by grad_output => grad_output * (1 - out^2)
        return grad_output.f.mul_zip(grad_output, diff)

class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """
        Forward pass for Sigmoid: out = 1/(1 + e^-x)
        """
        out = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """
        Backward pass for Sigmoid:
        d(sigma(x))/dx = sigma(x) * (1 - sigma(x))
        """
        sigma: Tensor = ctx.saved_values[0]
        return sigma * (1.0 - sigma) * grad_output


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """
        Forward pass for ReLU: out = max(0, t1)
        """
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """
        Backward pass for ReLU:
        d(max(0, x))/dx = 1 if x > 0 else 0
        """
        (a,) = ctx.saved_values
        return grad_output.f.relu_back_zip(a, grad_output)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """
        Forward pass for Log: out = log(t1)
        """
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """
        Backward pass for Log:
        d(log(x))/dx = 1/x
        """
        (a,) = ctx.saved_values
        return grad_output.f.log_back_zip(a, grad_output)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """
        Forward pass for Exp: out = e^t1
        """
        out = t1.f.exp_map(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """
        Backward pass for Exp:
        d(e^x)/dx = e^x
        """
        (a,) = ctx.saved_values
        return grad_output.f.mul_zip(a, grad_output)


# -----------------------------
# Reduction and comparison
# -----------------------------
class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """
        Forward pass for summation across a given dimension.
        """
        ctx.save_for_backward(a.shape, dim)
        return a.f.add_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """
        Backward pass for summation.
        """
        a_shape, dim = ctx.saved_values
        # The gradient w.r.t. the original shape is the same grad, broadcasted.
        return grad_output, 0.0


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """
        Forward pass for checking if all elements along dimension are non-zero.
        (Typically used as a logical operation, doesn't carry gradient.)
        """
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            # Flatten first, then do multiply reduction
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


class LT(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """
        Forward pass for 'Less Than': out = (a < b) ? 1.0 : 0.0
        """
        ctx.save_for_backward(a.shape, b.shape)
        return a.f.lt_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Backward pass for 'Less Than'.
        This is a boolean comparison => no gradient contribution.
        """
        a_shape, b_shape = ctx.saved_values
        return zeros(a_shape), zeros(b_shape)


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """
        Forward pass for 'Equal': out = (a == b) ? 1.0 : 0.0
        """
        ctx.save_for_backward(a.shape, b.shape)
        return a.f.eq_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Backward pass for 'Equal'.
        This is a boolean comparison => no gradient contribution.
        """
        a_shape, b_shape = ctx.saved_values
        return zeros(a_shape), zeros(b_shape)


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """
        Forward pass for 'IsClose': out = (abs(a-b) < some_epsilon) ? 1 : 0
        """
        return a.f.is_close_zip(a, b)


# -----------------------------
# Tensor shape manipulation
# -----------------------------
class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        """
        Forward pass for permute (transpose in arbitrary dimensions).
        """
        ctx.save_for_backward(order)
        return a._new(a._tensor.permute(*[int(order[i]) for i in range(order.size)]))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """
        Backward pass for permute. We invert the permutation order.
        """
        order: Tensor = ctx.saved_values[0]
        order2 = [
            item[0]
            for item in sorted(
                enumerate([order[i] for i in range(order.size)]),
                key=lambda x: x[1]
            )
        ]
        return grad_output._new(grad_output._tensor.permute(*order2)), 0.0


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """
        Forward pass for viewing a Tensor as a new shape (reshaping).
        """
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """
        Backward pass for view. We simply reshape the grad back to the original.
        """
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """
        Forward pass that copies a Tensor's values.
        """
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """
        Backward pass for copy is just returning the same gradient.
        """
        return grad_output


# -----------------------------
# Matrix multiply, specialized ops
# -----------------------------
class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """
        Forward pass for matrix multiplication: out = t1 @ t2
        """
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Backward pass for matrix multiplication:
        d/dt1(t1 @ t2) = grad_output @ t2^T
        d/dt2(t1 @ t2) = t1^T @ grad_output
        """
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


class Attn_Softmax(Function):
    @staticmethod
    def forward(ctx: Context, inp: Tensor, mask: Tensor) -> Tensor:
        """
        Forward pass for attention softmax with a mask (commonly used in Transformers).
        """
        out = inp.backend.attn_softmax_fw(inp, mask)
        ctx.save_for_backward(out, mask)
        return out

    @staticmethod
    def backward(ctx: Context, out_grad: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Backward pass for attention softmax. Mask typically has no gradient.
        """
        soft_inp, mask = ctx.saved_values
        dx = soft_inp.backend.attn_softmax_bw(out_grad, soft_inp)
        # Return zero gradient for mask
        mask_grad = mask.zeros(mask.shape)
        return dx, mask_grad


class LayerNorm(Function):
    """
    LayerNorm forward/backward:
      out = (x - mean)/sqrt(var + eps) * gamma + beta
    """
    @staticmethod
    def forward(ctx: Context, inp: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
        ln_out, saved_var, saved_mean = inp.f.layernorm_fw(inp, gamma, beta)
        ctx.save_for_backward(inp, gamma, beta, saved_var, saved_mean)
        return ln_out

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor):
        inp, gamma, beta, var, mean = ctx.saved_values
        d_inp, d_gamma, d_beta = inp.f.layernorm_bw(
            grad_out, inp, gamma, beta, var, mean
        )
        return d_inp, d_gamma, d_beta


# ----------------------------------------
# Helper functions to construct Tensors
# ----------------------------------------
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """
    Produce a zero tensor of size `shape`.
    """
    return minitorch.Tensor.make(
        [0] * int(operators.prod(shape)), shape, backend=backend
    )


def ones(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """
    Produce a ones tensor of size `shape`.
    """
    return minitorch.Tensor.make(
        [1] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """
    Produce a random tensor of size `shape`.
    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """
    Produce a tensor with data ls and shape `shape`.
    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """
    Produce a tensor with data and shape derived automatically from ls (nested list).
    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, list) and len(ls) > 0 and not isinstance(ls[0], list):
            # Already flat
            return ls
        elif isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


def tensor_from_numpy(
    ls: Storage, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """
    Wraps a numpy array into a miniTorch Tensor. 
    NOTE: Should only be used to initialize a tensor.
    """
    if ls.dtype != datatype:
        ls = ls.astype(datatype)

    # Flatten the numpy array into 1D storage
    res = minitorch.Tensor(
        v=minitorch.TensorData(
            ls.flatten(), 
            ls.shape,
            tuple(i // datasize for i in ls.strides)
        ),
        backend=backend
    )
    res.requires_grad_(requires_grad)
    return res


def zeros_tensor_from_numpy(shape, backend: TensorBackend = SimpleBackend):
    """
    Creates a zero-filled tensor with the given shape using numpy, then wraps it.
    NOTE: Should only be used to initialize a tensor.
    """
    zs = np.zeros(shape).astype(datatype)
    return minitorch.Tensor(
        v=minitorch.TensorData(
            zs.flatten(),
            shape,
            tuple(i // datasize for i in zs.strides)
        ),
        backend=backend
    )


def ones_tensor_from_numpy(shape, backend: TensorBackend = SimpleBackend):
    """
    Creates a one-filled tensor with the given shape using numpy, then wraps it.
    NOTE: Should only be used to initialize a tensor.
    """
    zs = np.ones(shape).astype(datatype)
    return minitorch.Tensor(
        v=minitorch.TensorData(
            zs.flatten(),
            shape,
            tuple(i // datasize for i in zs.strides)
        ),
        backend=backend
    )


# -----------------------------
# Gradient checking
# -----------------------------
import torch

def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """
    Numerically compute the gradient at a sample index using central difference.
    """
    x = vals[arg]
    up_np = np.zeros(x.shape, dtype=np.float64)
    up_np[ind] = epsilon

    vals1 = [
        torch.tensor(
            v.to_numpy().astype(np.float64)
        )
        if j != arg 
        else torch.tensor(x.to_numpy().astype(np.float64) + up_np)
        for j, v in enumerate(vals)
    ]
    vals2 = [
        torch.tensor(
            v.to_numpy().astype(np.float64)
        )
        if j != arg 
        else torch.tensor(x.to_numpy().astype(np.float64) - up_np)
        for j, v in enumerate(vals)
    ]
    delta = float(f(*vals1).sum() - f(*vals2).sum().numpy())
    return delta / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor, tol=1e-6) -> None:
    """
    Checks the gradient of the function f at random positions against 
    a numerical approximation using central difference.
    """
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """
    Gradient check error for function %s.

    Input %s

    Received derivative %f for argument %d and index %s,
    but was expecting derivative %f from central difference.
    """
    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
