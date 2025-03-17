# ###############################################################################
# # modules_transformer.py
# ###############################################################################
# # Final example code:
# #  - Embedding, Linear init scale=0.001 (더 낮춤)
# #  - LayerNorm eps=1e-2 (더 높임)
# #  - Flatten all but last dim in Linear => "batched matmul"
# #  - local_max for stable softmax
# #  - NumPy fallback for LN sqrt
# ###############################################################################

# import math
# import numpy as np
# from functools import reduce
# from operator import mul
# from typing import Optional

# from .tensor import Tensor, tensor_from_numpy
# from .module import Module, Parameter
# from .tensor_ops import TensorBackend
# from .nn import softmax, GELU

# def rand(shape, backend=None) -> Tensor:
#     """
#     Produce random Tensor in [0,1), shape must be a tuple
#     """
#     arr = np.random.rand(*shape).astype(np.float32)
#     return tensor_from_numpy(arr, backend=backend)

# def local_max(input: Tensor, dim: int, keepdims: bool = True) -> Tensor:
#     """
#     Fallback: convert input to NumPy, do np.max(..., axis=dim),
#     wrap back in a minitorch Tensor
#     """
#     arr = input.to_numpy()
#     mx_arr = arr.max(axis=dim)
#     if keepdims:
#         mx_arr = np.expand_dims(mx_arr, axis=dim)
#     return tensor_from_numpy(mx_arr, backend=input.backend)


# ###############################################################################
# # Embedding
# ###############################################################################
# class Embedding(Module):
#     """
#     Basic embedding with smaller init scale=0.001
#     """
#     def __init__(self, num_embeddings: int, embedding_dim: int, backend=None):
#         super().__init__()
#         self.num_embeddings = num_embeddings
#         self.embedding_dim  = embedding_dim
#         self.backend        = backend

#         # init scale => 0.001
#         init_w = (rand((num_embeddings, embedding_dim), backend=backend) - 0.5) * 0.001
#         self.weight = Parameter(init_w)

#     def forward(self, x: Tensor) -> Tensor:
#         # x: shape (B, T)
#         w_np = self.weight.value.to_numpy()  # (num_embeddings, embedding_dim)
#         x_np = x.to_numpy().astype(np.int64) # (B, T)

#         B, T = x.shape
#         out_arr = np.zeros((B, T, self.embedding_dim), dtype=np.float32)
#         for i in range(B):
#             for j in range(T):
#                 idx = x_np[i,j]
#                 if idx < 0 or idx >= self.num_embeddings:
#                     idx = 0
#                 out_arr[i,j,:] = w_np[idx,:]

#         return tensor_from_numpy(out_arr, backend=self.backend)


# ###############################################################################
# # Linear: flatten all but last dim => batched matmul
# ###############################################################################
# class Linear(Module):
#     """
#     If input shape is (*batch_dims, in_features),
#     we flatten all but the last => (N, in_features).
#     Then replicate weight => (N, in_features, out_features),
#     do (N,1,in_features) x (N,in_features,out_features),
#     => (N,1,out_features)->(N,out_features)->(*batch_dims, out_features).
#     """
#     def __init__(self, in_features, out_features, bias=True, backend=None):
#         super().__init__()
#         self.in_features  = in_features
#         self.out_features = out_features
#         self.backend      = backend

#         # init scale => 0.001
#         w_init = (rand((in_features, out_features), backend=backend) - 0.5) * 0.001
#         self.weight = Parameter(w_init)
#         if bias:
#             b_init = (rand((1, out_features), backend=backend) - 0.5) * 0.001
#             self.bias = Parameter(b_init)
#         else:
#             self.bias = None

#     def forward(self, x: Tensor) -> Tensor:
#         *batch_dims, in_f = x.shape
#         N = 1
#         for d in batch_dims:
#             N *= d
#         x_2d = x.view(N, in_f)

#         # reshape => (N,1,in_features)
#         x_3d = x_2d.view(N, 1, in_f)

#         # weight => (1, in_f, out_f), replicate => (N,in_f,out_f)
#         w_3d = self.weight.value.view(1, self.in_features, self.out_features)
#         w_np = w_3d.to_numpy()
#         w_np_batched = np.repeat(w_np, N, axis=0)
#         w_3d_batched = tensor_from_numpy(w_np_batched, backend=x.backend)

#         # matmul => (N,1,out_f)
#         out_3d = x.f.matrix_multiply(x_3d, w_3d_batched)
#         out_2d = out_3d.view(N, self.out_features)

#         # bias
#         if self.bias is not None:
#             out_2d = out_2d + self.bias.value

#         # reshape => (*batch_dims, out_features)
#         out = out_2d.view(*batch_dims, self.out_features)
#         return out


# ###############################################################################
# # LayerNorm1d
# ###############################################################################
# class LayerNorm1d(Module):
#     """
#     LN with eps=1e-2 (increased from 1e-3)
#     + fallback sqrt using NumPy
#     shape: [B, T, D], LN over last dimension D
#     """
#     def __init__(self, dim, eps=1e-2, backend=None):
#         super().__init__()
#         self.dim  = dim
#         self.eps  = eps
#         self.backend = backend

#         gamma_arr = np.ones((dim,), dtype=np.float32)
#         beta_arr  = np.zeros((dim,), dtype=np.float32)
#         gamma_t   = tensor_from_numpy(gamma_arr, backend=backend)
#         beta_t    = tensor_from_numpy(beta_arr,  backend=backend)

#         self.gamma = Parameter(gamma_t)
#         self.beta  = Parameter(beta_t)

#     def forward(self, x: Tensor) -> Tensor:
#         B, T, D = x.shape
#         mean = x.mean(dim=2).view(B,T,1)
#         var  = x.var(dim=2).view(B,T,1)

#         # fallback sqrt => NumPy
#         var_plus_eps_np = (var + self.eps).to_numpy()
#         var_sqrt_np     = np.sqrt(var_plus_eps_np)
#         var_sqrt        = tensor_from_numpy(var_sqrt_np, backend=self.backend)

#         x_norm = (x - mean) / var_sqrt
#         g_3d = self.gamma.value.view(1,1,D)
#         b_3d = self.beta.value.view(1,1,D)
#         return x_norm*g_3d + b_3d


# ###############################################################################
# # Dropout
# ###############################################################################
# class Dropout(Module):
#     def __init__(self, p=0.0):
#         super().__init__()
#         self.p = p
#     def forward(self, x: Tensor) -> Tensor:
#         if self.p <= 0.0:
#             return x
#         # else real dropout
#         return x

# ###############################################################################
# # MultiHeadAttention
# ###############################################################################
# class MultiHeadAttention(Module):
#     """
#     Softmax = stable (local_max), scale => sqrt(dim)
#     """
#     def __init__(
#         self,
#         n_embd: int,
#         n_head: int,
#         causal: bool = True,
#         p_dropout: float = 0.0,
#         bias: bool = True,
#         backend: Optional[TensorBackend] = None
#     ):
#         super().__init__()
#         self.backend = backend
#         self.n_embd  = n_embd
#         self.n_head  = n_head
#         self.causal  = causal
#         self.attn_hidden_dim = n_embd // n_head

#         self.q_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
#         self.k_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
#         self.v_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
#         self.out_projection= Linear(n_embd, n_embd, bias=bias, backend=backend)

#         self.dropout = Dropout(p_dropout)

#     def create_causal_mask(self, seq_len: int):
#         mask_value = -np.finfo(np.float32).max
#         tri = np.triu(np.ones((seq_len, seq_len),dtype=np.float32), k=1)
#         tri *= mask_value
#         tri = tri.reshape(1,1,seq_len,seq_len)
#         return tensor_from_numpy(tri, backend=self.backend)

#     def project_to_query_key_value(self, x: Tensor):
#         B,T,_ = x.shape
#         q_full = self.q_projection(x)
#         k_full = self.k_projection(x)
#         v_full = self.v_projection(x)

#         # => (B,n_head,T,dim)
#         q_4d = q_full.view(B,T,self.n_head,self.attn_hidden_dim).permute(0,2,1,3).contiguous()
#         k_4d = k_full.view(B,T,self.n_head,self.attn_hidden_dim).permute(0,2,3,1).contiguous()
#         v_4d = v_full.view(B,T,self.n_head,self.attn_hidden_dim).permute(0,2,1,3).contiguous()
#         return q_4d, k_4d, v_4d

#     def self_attention(self, q,kT,v):
#         B,nH,T,dim = q.shape
#         BN = B*nH

#         q2  = q.view(BN,T,dim)
#         kT2 = kT.view(BN,dim,T)
#         v2  = v.view(BN,T,dim)

#         attn_scores = q2 @ kT2
#         attn_scores /= float(dim)**0.5

#         mx = local_max(attn_scores, dim=2, keepdims=True)
#         attn_scores = attn_scores - mx

#         if self.causal:
#             mask = self.create_causal_mask(T)
#             mask_2=mask.view(1,T,T)
#             attn_scores = attn_scores + mask_2

#         attn_weights= softmax(attn_scores, dim=2)
#         attn_weights= self.dropout(attn_weights)
#         out_2d = attn_weights @ v2
#         out_4d = out_2d.view(B,nH,T,dim).permute(0,2,1,3).contiguous()
#         return out_4d.view(B,T,nH*dim)

#     def forward(self, x: Tensor):
#         q,kT,v = self.project_to_query_key_value(x)
#         attn_out= self.self_attention(q,kT,v)
#         out = self.out_projection(attn_out)
#         return out.contiguous()


# ###############################################################################
# # TransformerLayer
# ###############################################################################
# class TransformerLayer(Module):
#     def __init__(
#         self,
#         n_embd,
#         n_head,
#         p_dropout=0.0,
#         ln_eps=1e-2,   # matching LN above
#         bias=True,
#         backend=None
#     ):
#         super().__init__()
#         self.ln_1 = LayerNorm1d(dim=n_embd, eps=ln_eps, backend=backend)
#         self.ln_2 = LayerNorm1d(dim=n_embd, eps=ln_eps, backend=backend)

#         self.attention = MultiHeadAttention(
#             n_embd, n_head, causal=True, p_dropout=p_dropout,
#             bias=bias, backend=backend
#         )
#         self.ff = FeedForward(
#             n_embd, middle_dim=256, p_dropout=p_dropout,
#             bias=bias, backend=backend
#         )

#     def forward(self, x: Tensor) -> Tensor:
#         normed_1 = self.ln_1(x)
#         attn_out = self.attention(normed_1)
#         x = x + attn_out
#         x = x.contiguous()

#         normed_2 = self.ln_2(x)
#         ff_out   = self.ff(normed_2)
#         x = x + ff_out
#         x = x.contiguous()
#         return x


# ###############################################################################
# # FeedForward
# ###############################################################################
# class FeedForward(Module):
#     def __init__(
#         self,
#         n_embd: int,
#         middle_dim: int = 256,
#         p_dropout: float = 0.0,
#         bias: bool = True,
#         backend: TensorBackend = None
#     ):
#         super().__init__()
#         self.linear_in  = Linear(n_embd, middle_dim, bias=bias, backend=backend)
#         self.linear_out = Linear(middle_dim, n_embd, bias=bias, backend=backend)
#         self.dropout    = Dropout(p_dropout)

#     def forward(self, x: Tensor) -> Tensor:
#         B,T,D = x.shape
#         hidden= x.contiguous().view(B*T, D)
#         hidden= self.linear_in(hidden)
#         hidden= GELU(hidden)
#         hidden= self.linear_out(hidden)
#         out   = hidden.view(B, T, D)
#         out   = self.dropout(out)
#         return out


# ###############################################################################
# # DecoderLM
# ###############################################################################
# class DecoderLM(Module):
#     def __init__(
#         self,
#         n_vocab,
#         n_embd,
#         n_head,
#         n_positions,
#         p_dropout=0.0,
#         ln_eps=1e-2,        # also matching new LN
#         bias=True,
#         backend=None
#     ):
#         super().__init__()
#         self.backend = backend
#         self.n_embd = n_embd
#         self.n_vocab= n_vocab

#         self.token_embeddings = Embedding(n_vocab, n_embd, backend=backend)
#         self.position_embeddings=Embedding(n_positions,n_embd,backend=backend)
#         self.dropout = Dropout(p_dropout)

#         self.t_layer_1=TransformerLayer(n_embd,n_head,p_dropout,ln_eps,bias,backend)
#         self.t_layer_2=TransformerLayer(n_embd,n_head,p_dropout,ln_eps,bias,backend)
#         self.t_layer_3=TransformerLayer(n_embd,n_head,p_dropout,ln_eps,bias,backend)
#         self.t_layer_4=TransformerLayer(n_embd,n_head,p_dropout,ln_eps,bias,backend)

#         self.ln = LayerNorm1d(dim=n_embd, eps=ln_eps, backend=backend)
#         self.lm_head = Linear(n_embd, n_vocab, bias=bias, backend=backend)

#     def forward(self, idx: Tensor) -> Tensor:
#         B,T = idx.shape
#         tok_emb = self.token_embeddings(idx)

#         pos_ids_np = np.arange(T,dtype=np.int32).reshape(1,T)
#         pos_ids = tensor_from_numpy(pos_ids_np, backend=self.backend)
#         pos_emb = self.position_embeddings(pos_ids)

#         x = tok_emb + pos_emb
#         x = self.dropout(x)

#         x = self.t_layer_1(x)
#         x = self.t_layer_2(x)
#         x = self.t_layer_3(x)
#         x = self.t_layer_4(x)

#         x = self.ln(x)
#         logits= self.lm_head(x)
#         return logits


###############################################################################
# modules_transformer.py
###############################################################################
# This file defines the core components of a decoder-only Transformer model
# in a GPT-2 style, including multi-head masked self-attention, feedforward,
# a single Transformer block (pre-LN style), and a full DecoderLM module.

import numpy as np
from typing import Optional

from .tensor import tensor, tensor_from_numpy
from .module import Module
from .modules_basic import (
    Embedding,
    Dropout,
    LayerNorm1d,
    Linear
)
from .tensor_ops import TensorBackend
from .nn import (
    softmax,
    dropout,
    GELU,
)
from . import operators  # for numeric constants or helper ops if needed


datatype = np.float32


###############################################################################
# MultiHeadAttention
###############################################################################
class MultiHeadAttention(Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        causal: bool = True,
        p_dropout: float = 0.1,
        bias: bool = True,
        backend: Optional[TensorBackend] = None
    ):
        super().__init__()

        self.backend   = backend
        self.n_embd    = n_embd
        self.n_head    = n_head
        self.causal    = causal

        # dimension per head
        self.attn_hidden_dim = n_embd // n_head

        # Q, K, V linear layers
        self.q_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.k_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.v_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)

        # Output projection
        self.out_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)

        # Dropout on attention weights or final projection
        self.dropout = Dropout(p_dropout)


    def create_causal_mask(self, seq_len: int):
        """
        Returns a shape [1, 1, seq_len, seq_len] upper-triangular mask of -inf above diag.
        This is used to prevent attention to future tokens in an autoregressive LM.
        """
        mask_value = -np.finfo(datatype).max  # effectively -∞
        # Create an upper-triangular matrix (above diag)
        tri = np.triu(np.ones((seq_len, seq_len), dtype=datatype), k=1)
        tri = tri * mask_value
        # reshape to [1,1,T,T] for broadcasting: [B, n_head, T, T]
        tri = tri.reshape(1, 1, seq_len, seq_len)
        return tensor_from_numpy(tri, backend=self.backend)

    def project_to_query_key_value(self, x):
        B, T, _ = x.shape

        # Q
        q_full = self.q_projection(x) 
        q_4d = q_full.view(B, T, self.n_head, self.attn_hidden_dim)
        q_4d = q_4d.permute(0, 2, 1, 3).contiguous()  # make Q physically contiguous

        # K
        k_full = self.k_projection(x)
        k_4d = k_full.view(B, T, self.n_head, self.attn_hidden_dim)
        k_4d = k_4d.permute(0, 2, 3, 1).contiguous()  # make K physically contiguous

        # V
        v_full = self.v_projection(x)
        v_4d = v_full.view(B, T, self.n_head, self.attn_hidden_dim)
        v_4d = v_4d.permute(0, 2, 1, 3).contiguous()  # make V physically contiguous

        return q_4d, k_4d, v_4d

    def self_attention(self, q, kT, v):
        """
        q  = [B, n_head, T, dim]
        kT = [B, n_head, dim, T]   # "keys transposed"
        v  = [B, n_head, T, dim]

        We'll flatten (B,n_head) => BN as the "batch dim"
        so miniTorch's matmul sees [BN, M, K] x [BN, K, N].
        """
        B, nH, T, dim = q.shape

        # Flatten the first two dims into BN
        BN = B * nH

        # (1) Reshape q to [BN, T, dim], kT to [BN, dim, T], v to [BN, T, dim]
        q2  = q.view(BN, T, dim)
        kT2 = kT.view(BN, dim, T)
        v2  = v.view(BN, T, dim)

        # (2) Now do the batched matmul
        attn_scores = q2 @ kT2  # [BN, T, T]
        scale = float(dim) ** 0.5
        attn_scores /= scale

        if self.causal:
            mask = self.create_causal_mask(T)   # shape [1,1,T,T]
            # reshape to (1, T, T)
            mask_2 = mask.view(1, T, T)
            # rely on broadcast => shape (BN,T,T)
            attn_scores = attn_scores + mask_2

        # (4) softmax => shape [BN, T, T]
        attn_weights = softmax(attn_scores, dim=2)
        attn_weights = self.dropout(attn_weights)

        # (5) multiply by v => shape [BN, T, dim]
        out_2d = attn_weights @ v2  # shape [BN, T, dim]

        # (6) Reshape back to [B, n_head, T, dim]
        out_4d = out_2d.view(B, nH, T, dim)

        # (7) Permute => [B, T, n_head, dim] and flatten to [B, T, n_embd]
        out_4d = out_4d.permute(0, 2, 1, 3).contiguous()  # shape [B, T, nH, dim]
        result = out_4d.view(B, T, nH * dim)              # shape [B, T, n_embd]
        return result


    def forward(self, x):
        """
        x => [B, T, n_embd], returns => same shape
        Steps:
          1) project to Q,K,V
          2) do self-attention
          3) final out-projection
        """
        # 1) get Q, K^T, V
        q, kT, v = self.project_to_query_key_value(x)

        # 2) do self-attention => [B, T, n_embd]
        attn_out = self.self_attention(q, kT, v)

        # 3) final out projection => [B, T, n_embd]
        out = self.out_projection(attn_out)
        return out.contiguous()


###############################################################################
# FeedForward
###############################################################################
class FeedForward(Module):
    def __init__(self, n_embd: int, middle_dim: int = 256, p_dropout: float = 0.1,
                 bias: bool = True, backend: TensorBackend = None):
        super().__init__()
        self.linear_in  = Linear(n_embd, middle_dim, bias=bias, backend=backend)
        self.linear_out = Linear(middle_dim, n_embd, bias=bias, backend=backend)
        self.dropout    = Dropout(p_dropout)

    def forward(self, x):
        B, T, D = x.shape
        # => ensure contiguous before .view
        hidden = x.contiguous().view(B * T, D)
        hidden = self.linear_in(hidden)
        hidden = GELU(hidden)
        hidden = self.linear_out(hidden)
        out = hidden.view(B, T, D)
        out = self.dropout(out)
        return out



###############################################################################
# TransformerLayer
###############################################################################
class TransformerLayer(Module):
    """
    Pre-LN Transformer block:
      LN -> MHA -> residual
      LN -> FeedForward -> residual

    Attributes:
        ln_1, ln_2: layernorm
        attention  : a MultiHeadAttention module
        ff         : the feedforward block
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        p_dropout: float = 0.1,
        ln_eps: float = 1e-5,
        bias: bool = True,
        backend: TensorBackend = None
    ):
        super().__init__()
        self.ln_1 = LayerNorm1d(dim=n_embd, eps=ln_eps, backend=backend)
        self.ln_2 = LayerNorm1d(dim=n_embd, eps=ln_eps, backend=backend)

        self.attention = MultiHeadAttention(
            n_embd=n_embd, 
            n_head=n_head, 
            causal=True,  # GPT is causal
            p_dropout=p_dropout,
            bias=bias,
            backend=backend
        )
        # feedforward with 4x expansion
        self.ff = FeedForward(
            n_embd=n_embd,
            middle_dim= 256, #4*n_embd,
            p_dropout=p_dropout,
            bias=bias,
            backend=backend
        )

    def forward(self, x):
        """
        x => [B, T, n_embd], returns => same shape
        Pre-LN approach:
          1) LN(x) -> MHA -> add to x
          2) LN(x) -> FF  -> add to x
        """
        # LN -> MHA -> residual
        normed_1 = self.ln_1(x)
        attn_out = self.attention(normed_1)
        x = x + attn_out  # residual
        x = x.contiguous()               # <-- important

        # LN -> FF -> residual
        normed_2 = self.ln_2(x)
        ff_out   = self.ff(normed_2)
        x        = x + ff_out
        x = x.contiguous()               # <-- also ensure contiguous
        return x


###############################################################################
# DecoderLM
###############################################################################
class DecoderLM(Module):
    """
    GPT-like decoder:
      - token + position embeddings
      - a certain # of TransformerLayers
      - final LN
      - linear projection to vocab
    """

    def __init__(
        self,
        n_vocab: int,
        n_embd: int,
        n_head: int,
        n_positions: int,
        p_dropout: float = 0.1,
        ln_eps: float = 1e-5,
        bias: bool = True,
        backend: TensorBackend = None
    ):
        super().__init__()
        self.backend = backend
        self.n_embd = n_embd
        self.n_vocab = n_vocab

        # Token embeddings => [n_vocab, n_embd]
        self.token_embeddings = Embedding(
            num_embeddings=n_vocab, 
            embedding_dim=n_embd, 
            backend=backend
        )

        # Position embeddings => [n_positions, n_embd]
        self.position_embeddings = Embedding(
            num_embeddings=n_positions, 
            embedding_dim=n_embd, 
            backend=backend
        )

        self.dropout = Dropout(p_dropout)

        # We'll do 4 Transformer layers for demonstration
        self.t_layer_1 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)
        self.t_layer_2 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)
        self.t_layer_3 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)
        self.t_layer_4 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)

        # final LN
        self.ln = LayerNorm1d(dim=n_embd, eps=ln_eps, backend=backend)

        # LM head => project from [n_embd] to [n_vocab]
        self.lm_head = Linear(n_embd, n_vocab, bias=bias, backend=backend)

    def forward(self, idx):
        """
        idx: [B, T]
        => returns => [B, T, n_vocab]  # final logits

        Steps:
        1) token + position embeddings => shape [B, T, n_embd]
        2) dropout
        3) pass through 4 transformer layers
        4) final LN
        5) project to vocab => [B, T, n_vocab]
        """
        B, T = idx.shape

        # 1) token + position embeddings
        # token => [B, T, n_embd]
        tok_emb = self.token_embeddings(idx)

        # position => shape => [1, T, n_embd]
        pos_ids_np = np.arange(T, dtype=np.int32).reshape(1, T)  # shape (1,T)
        pos_ids = tensor_from_numpy(pos_ids_np, backend=self.backend)
        pos_emb = self.position_embeddings(pos_ids)  # => [1, T, n_embd]

        # sum => broadcast => [B, T, n_embd]
        x = tok_emb + pos_emb

        # 2) dropout
        x = self.dropout(x)

        # 3) pass through each Transformer layer
        x = self.t_layer_1(x)
        x = self.t_layer_2(x)
        x = self.t_layer_3(x)
        x = self.t_layer_4(x)

        # 4) final LN
        x = self.ln(x)

        # 5) project to vocab => [B, T, n_vocab]
        logits = self.lm_head(x)
        return logits
