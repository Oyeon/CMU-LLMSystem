#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/cub.cuh>

// LightSeq headers
#include "includes/block_reduce.h"  // warpReduce, blockReduce
#include "includes/kernels.h"       // some definitions, WARP_SIZE, etc

namespace cg = cooperative_groups;

/****************************************************************************************
 * Provide a local definition of lightseq::cuda::check_gpu_error(...) 
 * to fix the missing symbol: _ZN8lightseq4cuda15check_gpu_errorI9cudaErrorEEvT_PKcS5_i
 *
 * Exactly matches the template signature that the macro uses in LightSeq code.
 ***************************************************************************************/

namespace lightseq {
namespace cuda {

// The template that might be used by macros: check_gpu_error(result, "func", __FILE__, __LINE__)
template <>
void check_gpu_error<cudaError_t>(
    cudaError_t result, 
    const char *func, 
    const char *file, 
    int line)
{
  if (result != cudaSuccess) {
    fprintf(stderr,
            "lightseq::cuda::check_gpu_error\n"
            "  --> CUDA error at %s:%d code=%d(%s) in %s\n",
            file, line, int(result), cudaGetErrorString(result), func);
    // You could choose a different error-handling strategy:
    exit(EXIT_FAILURE);
  }
}

} // namespace cuda
} // namespace lightseq

/****************************************************************************************
 * Additional constants / helpers from your original code
 ***************************************************************************************/
static constexpr float EPSILON = 1e-8f;
static constexpr float REDUCE_FLOAT_INF_NEG = -1e30f;

/****************************************************************************************
 * Flatten a 3D index [i0, i1, i2] => single index, used in your forward kernels
 ***************************************************************************************/
static __device__ __forceinline__ int flat_3dim(
    int i0, int i1, int i2, 
    int shape1, int shape2)
{
  // position = i0*(shape1*shape2) + i1*(shape2) + i2
  return i0 * (shape1 * shape2) + i1 * shape2 + i2;
}

/****************************************************************************************
 * Kernel: ker_attn_softmax_lt32
 *   - For small to_len <= 32
 *   - warp-level reduction only
 ***************************************************************************************/
template <typename T, int block_dim, int ele_per_thread>
__global__ void ker_attn_softmax_lt32(
    T *inp,           // [batch_size, nhead, from_len, to_len]
    const T *attn_mask,  // [batch_size, to_len] or null
    int from_len,
    int to_len,
    bool mask_future)
{
  int batch_id = blockIdx.y;
  int head_id  = blockIdx.z;
  const int nhead = gridDim.z;

  const int token_per_reduce = 1;

  // We'll do a CUB block load/store
  typedef cub::BlockLoad<T, block_dim, ele_per_thread, cub::BLOCK_LOAD_VECTORIZE>   BlockLoad;
  __shared__ typename BlockLoad::TempStorage ts_load;

  typedef cub::BlockStore<T, block_dim, ele_per_thread, cub::BLOCK_STORE_VECTORIZE> BlockStore;
  __shared__ typename BlockStore::TempStorage ts_store;

  // Optionally load mask
  T mval[ele_per_thread];
  if (attn_mask) {
    attn_mask += (batch_id * to_len);
    BlockLoad(ts_load).Load(attn_mask, mval, to_len, REDUCE_FLOAT_INF_NEG);
  }

  // offset into inp => shape [batch_size, nhead, from_len, to_len]
  inp += flat_3dim(batch_id, head_id, 0, nhead, from_len * to_len);

  // for each token
  for (int token_id = blockIdx.x * token_per_reduce; 
       token_id < from_len; 
       token_id += gridDim.x * token_per_reduce)
  {
    // load row
    T inp_val[token_per_reduce][ele_per_thread];
    for (int i = 0; i < token_per_reduce && (token_id + i) < from_len; i++) {
      BlockLoad(ts_load).Load(inp + (token_id + i)*to_len, 
                              inp_val[i], 
                              to_len, 
                              REDUCE_FLOAT_INF_NEG);
    }

    // step 1) local max
    float val[token_per_reduce][ele_per_thread];
    float l_max[token_per_reduce];
    for (int i = 0; i < token_per_reduce; i++) {
      l_max[i] = REDUCE_FLOAT_INF_NEG;
      for (int j = 0; j < ele_per_thread; j++) {
        float tmp = float(inp_val[i][j]);
        if (attn_mask) {
          tmp += float(mval[j]);
        }
        int global_col_idx = threadIdx.x * ele_per_thread + j;
        if (mask_future && (global_col_idx > (token_id + i))) {
          tmp = REDUCE_FLOAT_INF_NEG;
        }
        val[i][j] = tmp;
        l_max[i]  = fmaxf(l_max[i], tmp);
      }
    }

    // warp-level reduce => from block_reduce.h
    lightseq::cuda::warpReduce<lightseq::cuda::ReduceType::kMax, token_per_reduce>(l_max);

    // step 2) sum of exp
    float l_sum[token_per_reduce];
    for (int i = 0; i < token_per_reduce; i++) {
      l_sum[i] = 0.f;
      for (int j = 0; j < ele_per_thread; j++) {
        float ex = __expf(val[i][j] - l_max[i]);
        val[i][j] = ex;
        l_sum[i] += ex;
      }
    }
    lightseq::cuda::warpReduce<lightseq::cuda::ReduceType::kSum, token_per_reduce>(l_sum);

    // step 3) store final
    for (int i = 0; i < token_per_reduce && (token_id + i) < from_len; i++) {
      float inv_sum = 1.f / (l_sum[i] + EPSILON);
      for (int j = 0; j < ele_per_thread; j++) {
        inp_val[i][j] = T(val[i][j] * inv_sum);
      }
      BlockStore(ts_store).Store(inp + (token_id + i)*to_len, 
                                 inp_val[i], 
                                 to_len);
    }

  } // token_id
}

/****************************************************************************************
 * Kernel: ker_attn_softmax
 *   - For larger to_len (>32), uses block-level reduction from block_reduce.h
 ***************************************************************************************/
template <typename T, int block_dim, int ele_per_thread>
__global__ void ker_attn_softmax(
    T *inp,
    const T *attn_mask,
    int from_len,
    int to_len,
    bool mask_future)
{
  int batch_id = blockIdx.y;
  int head_id  = blockIdx.z;
  const int nhead = gridDim.z;

  const int token_per_reduce = 1;

  typedef cub::BlockLoad<T,  block_dim, ele_per_thread, cub::BLOCK_LOAD_VECTORIZE>   BlockLoad;
  __shared__ typename BlockLoad::TempStorage ts_load;
  typedef cub::BlockStore<T, block_dim, ele_per_thread, cub::BLOCK_STORE_VECTORIZE>  BlockStore;
  __shared__ typename BlockStore::TempStorage ts_store;

  T mval[ele_per_thread];
  if (attn_mask) {
    attn_mask += (batch_id * to_len);
    BlockLoad(ts_load).Load(attn_mask, mval, to_len, REDUCE_FLOAT_INF_NEG);
  }

  inp += flat_3dim(batch_id, head_id, 0, nhead, from_len*to_len);

  for (int token_id = blockIdx.x * token_per_reduce;
       token_id < from_len;
       token_id += gridDim.x * token_per_reduce)
  {
    // load row
    T inp_val[token_per_reduce][ele_per_thread];
    for (int i = 0; i < token_per_reduce && (token_id + i) < from_len; i++) {
      BlockLoad(ts_load).Load(inp + (token_id + i)*to_len, 
                              inp_val[i], 
                              to_len,
                              REDUCE_FLOAT_INF_NEG);
    }

    // step 1) local max
    float val[token_per_reduce][ele_per_thread];
    float l_max[token_per_reduce];
    for (int i=0; i< token_per_reduce; i++){
      l_max[i] = REDUCE_FLOAT_INF_NEG;
      for (int j=0; j< ele_per_thread; j++){
        float tmp = float(inp_val[i][j]);
        if (attn_mask) {
          tmp += float(mval[j]);
        }
        int global_col = threadIdx.x * ele_per_thread + j;
        if (mask_future && (global_col > (token_id + i))) {
          tmp = REDUCE_FLOAT_INF_NEG;
        }
        val[i][j] = tmp;
        l_max[i]  = fmaxf(l_max[i], tmp);
      }
    }

    // block-level reduce => from block_reduce.h
    lightseq::cuda::blockReduce<lightseq::cuda::ReduceType::kMax, token_per_reduce>(l_max);

    __shared__ float s_max[token_per_reduce];
    if (threadIdx.x == 0) {
      for (int i=0; i<token_per_reduce; i++){
        s_max[i] = l_max[i];
      }
    }
    __syncthreads();

    // step 2) sum of exp
    float l_sum[token_per_reduce];
    for (int i=0; i< token_per_reduce; i++){
      float mx = s_max[i];
      l_sum[i] = 0.f;
      for (int j=0; j< ele_per_thread; j++){
        float ex = __expf(val[i][j] - mx);
        val[i][j] = ex;
        l_sum[i] += ex;
      }
    }
    lightseq::cuda::blockReduce<lightseq::cuda::ReduceType::kSum, token_per_reduce>(l_sum);

    __shared__ float s_sum[token_per_reduce];
    if (threadIdx.x == 0){
      for (int i=0; i< token_per_reduce; i++){
        s_sum[i] = 1.f / (l_sum[i] + EPSILON);
      }
    }
    __syncthreads();

    // step 3) store final
    for (int i=0; i< token_per_reduce && (token_id + i)<from_len; i++){
      float inv_sum = s_sum[i];
      for (int j=0; j< ele_per_thread; j++){
        inp_val[i][j] = T(val[i][j] * inv_sum);
      }
      BlockStore(ts_store).Store(inp + (token_id + i)*to_len, inp_val[i], to_len);
    }

  } // token_id
}

/****************************************************************************************
 * Host function: launch_attn_softmax
 *   shape: [batch_size, nhead, from_len, to_len]
 *   optional attn_mask: [batch_size, to_len], or null
 ***************************************************************************************/
extern "C"
void launch_attn_softmax(
    float *inp,
    const float *attn_mask,
    int batch_size,
    int nhead,
    int from_len,
    int to_len,
    bool mask_future,
    cudaStream_t stream)
{
  // allocate
  size_t float_size = sizeof(float);
  size_t inp_size   = (size_t)batch_size * nhead * from_len * to_len * float_size;

  float *d_inp=nullptr, *d_mask=nullptr;
  lightseq::cuda::check_gpu_error<cudaError_t>(
      cudaMalloc((void**)&d_inp, inp_size), 
      "cudaMalloc(d_inp)", __FILE__, __LINE__);
  lightseq::cuda::check_gpu_error<cudaError_t>(
      cudaMemcpy(d_inp, inp, inp_size, cudaMemcpyHostToDevice),
      "cudaMemcpy inp->d_inp", __FILE__, __LINE__);

  if (attn_mask) {
    size_t mask_size = (size_t)batch_size * to_len * float_size;
    lightseq::cuda::check_gpu_error<cudaError_t>(
        cudaMalloc((void**)&d_mask, mask_size),
        "cudaMalloc(d_mask)", __FILE__, __LINE__);
    lightseq::cuda::check_gpu_error<cudaError_t>(
        cudaMemcpy(d_mask, attn_mask, mask_size, cudaMemcpyHostToDevice),
        "cudaMemcpy attn_mask->d_mask", __FILE__, __LINE__);
  }

  // pick kernel config
  dim3 grid_dim(1, batch_size, nhead);

  if      (to_len <= 32) {
    ker_attn_softmax_lt32<float, 32, 1><<<grid_dim, 32, 0, stream>>>(
        d_inp, d_mask, from_len, to_len, mask_future);
  }
  else if (to_len <= 64) {
    ker_attn_softmax_lt32<float, 32, 2><<<grid_dim, 32, 0, stream>>>(
        d_inp, d_mask, from_len, to_len, mask_future);
  }
  else {
    // bigger range
    if      (to_len <= 128) { grid_dim.x=16;   ker_attn_softmax<float,64,   2><<<grid_dim, 64,   0,stream>>>(d_inp, d_mask, from_len, to_len, mask_future); }
    else if (to_len <= 256) { grid_dim.x=32;   ker_attn_softmax<float,128,  2><<<grid_dim,128,  0,stream>>>(d_inp, d_mask, from_len, to_len, mask_future); }
    else if (to_len <= 512) { grid_dim.x=64;   ker_attn_softmax<float,256,  2><<<grid_dim,256,  0,stream>>>(d_inp, d_mask, from_len, to_len, mask_future); }
    else if (to_len <=1024) { grid_dim.x=128;  ker_attn_softmax<float,512,  2><<<grid_dim,512,  0,stream>>>(d_inp, d_mask, from_len, to_len, mask_future); }
    else if (to_len <=2048) { grid_dim.x=256;  ker_attn_softmax<float,1024, 2><<<grid_dim,1024, 0,stream>>>(d_inp, d_mask, from_len, to_len, mask_future); }
    else {
      fprintf(stderr, "launch_attn_softmax: to_len=%d not supported.\n", to_len);
      if (d_mask) cudaFree(d_mask);
      cudaFree(d_inp);
      return;
    }
  }

  // copy back
  lightseq::cuda::check_gpu_error<cudaError_t>(
      cudaMemcpy(inp, d_inp, inp_size, cudaMemcpyDeviceToHost),
      "cudaMemcpy d_inp->inp", __FILE__, __LINE__);

  // free
  if (d_mask) cudaFree(d_mask);
  cudaFree(d_inp);
  cudaDeviceSynchronize();
}

/****************************************************************************************
 * Kernel: ker_attn_softmax_bw
 *   - Backward pass: dSoftmax = Softmax(x)*(dOut - sum(dOut*Softmax(x)))
 ***************************************************************************************/
template <typename T, int ITERATIONS>
__global__ void ker_attn_softmax_bw(
    T *grad,       // [rows*to_len]
    const T *inp,  // same shape
    int softmax_len)
{
  int warps_per_block = blockDim.y; // e.g. 4
  int row_idx = blockIdx.x * warps_per_block + threadIdx.y;
  int offset  = row_idx*softmax_len + threadIdx.x;

  grad += offset;
  inp  += offset;

  T grad_reg[ITERATIONS];
  T inp_reg[ITERATIONS];
  float sum = 0.f;

#pragma unroll
  for (int i=0; i< ITERATIONS; i++){
    int col = threadIdx.x + i*32; // WARP_SIZE=32
    if (col < softmax_len) {
      grad_reg[i] = grad[i*32];
      inp_reg[i]  = inp [i*32];
      sum += float(grad_reg[i]) * float(inp_reg[i]);
    }
  }

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

  // warp shuffle reduce
  for (int mask = 16; mask > 0; mask >>=1) {
    float other = g.shfl_xor(sum, mask);
    sum += other;
  }

#pragma unroll
  for (int i=0; i< ITERATIONS; i++){
    int col = threadIdx.x + i*32;
    if (col < softmax_len) {
      float g_  = float(grad_reg[i]);
      float in_ = float(inp_reg[i]);
      grad[i*32] = T(in_ * (g_ - sum));
    }
  }
}

/****************************************************************************************
 * Host function: launch_attn_softmax_bw
 *
 *   out_grad => shape [batch_size, nhead, from_len, to_len]
 *   soft_inp => shape [batch_size, nhead, from_len, to_len], forward result
 *   rows = batch_size*nhead*from_len
 ***************************************************************************************/
extern "C"
void launch_attn_softmax_bw(
    float *out_grad,
    const float *soft_inp,
    int rows,        // batch_size*nhead*from_len
    int softmax_len, // to_len
    cudaStream_t stream)
{
  size_t total_size = size_t(rows)* softmax_len * sizeof(float);

  float *d_grad=nullptr, *d_inp=nullptr;
  lightseq::cuda::check_gpu_error<cudaError_t>(
      cudaMalloc((void**)&d_grad, total_size),
      "cudaMalloc(d_grad)", __FILE__, __LINE__);
  lightseq::cuda::check_gpu_error<cudaError_t>(
      cudaMemcpy(d_grad, out_grad, total_size, cudaMemcpyHostToDevice),
      "Memcpy out_grad->d_grad", __FILE__, __LINE__);

  lightseq::cuda::check_gpu_error<cudaError_t>(
      cudaMalloc((void**)&d_inp, total_size),
      "cudaMalloc(d_inp)", __FILE__, __LINE__);
  lightseq::cuda::check_gpu_error<cudaError_t>(
      cudaMemcpy(d_inp, soft_inp, total_size, cudaMemcpyHostToDevice),
      "Memcpy soft_inp->d_inp", __FILE__, __LINE__);

  // We define blockDim=(32,4) => 4 warps => gridDim.x=(rows+3)//4
  const int warps_per_block=4;
  dim3 block_dim(32, warps_per_block);
  dim3 grid_dim((rows + warps_per_block-1)/warps_per_block);

  // pick ITERATIONS so that 32*ITERATIONS >= softmax_len
  if      (softmax_len <=   32) { ker_attn_softmax_bw<float,1>  <<<grid_dim, block_dim, 0, stream>>>(d_grad, d_inp, softmax_len); }
  else if (softmax_len <=   64) { ker_attn_softmax_bw<float,2>  <<<grid_dim, block_dim, 0, stream>>>(d_grad, d_inp, softmax_len); }
  else if (softmax_len <=  128) { ker_attn_softmax_bw<float,4>  <<<grid_dim, block_dim, 0, stream>>>(d_grad, d_inp, softmax_len); }
  else if (softmax_len <=  256) { ker_attn_softmax_bw<float,8>  <<<grid_dim, block_dim, 0, stream>>>(d_grad, d_inp, softmax_len); }
  else if (softmax_len <=  384) { ker_attn_softmax_bw<float,12> <<<grid_dim, block_dim, 0, stream>>>(d_grad, d_inp, softmax_len); }
  else if (softmax_len <=  512) { ker_attn_softmax_bw<float,16> <<<grid_dim, block_dim, 0, stream>>>(d_grad, d_inp, softmax_len); }
  else if (softmax_len <=  768) { ker_attn_softmax_bw<float,24> <<<grid_dim, block_dim, 0, stream>>>(d_grad, d_inp, softmax_len); }
  else if (softmax_len <= 1024) { ker_attn_softmax_bw<float,32> <<<grid_dim, block_dim, 0, stream>>>(d_grad, d_inp, softmax_len); }
  else if (softmax_len <= 2048) { ker_attn_softmax_bw<float,64> <<<grid_dim, block_dim, 0, stream>>>(d_grad, d_inp, softmax_len); }
  else {
    fprintf(stderr, "launch_attn_softmax_bw: softmax_len=%d not supported.\n", softmax_len);
    cudaFree(d_grad);
    cudaFree(d_inp);
    return;
  }

  // copy back
  lightseq::cuda::check_gpu_error<cudaError_t>(
      cudaMemcpy(out_grad, d_grad, total_size, cudaMemcpyDeviceToHost),
      "Memcpy d_grad->out_grad", __FILE__, __LINE__);

  // free
  cudaFree(d_grad);
  cudaFree(d_inp);
  cudaDeviceSynchronize();
}




// /****************************************************************************************
//  * src/softmax_kernel.cu
//  *
//  * Implements fused softmax forward & backward kernels for attention mechanisms.
//  * "ker_attn_softmax_lt32" handles the case where sequence length <= 32
//  * "ker_attn_softmax" handles the general case (>32 up to ~2048)
//  * "ker_attn_softmax_bw" computes backward pass for dSoftmax in self-attention.
//  *
//  * The backward pass uses the identity:
//  *    d(Softmax(x)) = Softmax(x) * (dOut - sum(dOut * Softmax(x)))
//  * for each row along the softmax dimension.
//  *
//  * N.B.: This file also includes host launcher functions:
//  *   launch_attn_softmax(...)     // forward
//  *   launch_attn_softmax_bw(...)  // backward
//  *
//  ***************************************************************************************/

// #include <math.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <cuda_runtime.h>
// #include <cooperative_groups.h>
// #include <cub/block/block_load.cuh>
// #include <cub/block/block_store.cuh>
// #include <cub/cub.cuh>

// // If you want to keep consistent with code from "combine.cu", you might reuse these:
// static constexpr float REDUCE_FLOAT_INF_NEG = -1.0e30f;
// static constexpr float EPSILON = 1e-6f;

// // For warp-level partial reductions
// #define WARP_SIZE 32

// namespace cg = cooperative_groups;

// /**************************************************************************************
//  * Helpers
//  **************************************************************************************/
// inline __device__ int flat_3dim(int i0, int i1, int i2, int d1, int d2) {
//   // Flatten a 3D index [i0, i1, i2] with shape [?, d1, d2].
//   // i.e. position = i0 * (d1*d2) + i1*(d2) + i2.
//   return i0 * (d1*d2) + i1*d2 + i2;
// }

// // For a "mask future" style, we might do something like
// // if (mask_future && (column_idx > row_idx)), val = -inf, etc.

// /**************************************************************************************
//  * Warp Reduction
//  *
//  * We define a warpReduce function that can do either sum or max for an array of length K
//  * (where K <= 4 in this code) across the threads in a warp.
//  *
//  **************************************************************************************/

// enum class ReduceType { kSum, kMax };

// // warp-level partial sum or max across K values
// template <ReduceType RT, int K>
// __device__ __forceinline__ void warpReduce(float vals[K]) {
//   // Each lane has an array of length K in 'vals'.
//   // We'll do standard warp shuffle (32-lane) reduction.
// #pragma unroll
//   for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
//     for (int i = 0; i < K; i++) {
//       float other = __shfl_xor_sync(0xffffffff, vals[i], offset);
//       if (RT == ReduceType::kSum) {
//         vals[i] += other;
//       } else {  // RT == kMax
//         vals[i] = fmaxf(vals[i], other);
//       }
//     }
//   }
// }

// /**************************************************************************************
//  * Block Reduction
//  *
//  * Similar idea but uses shared memory for block-level. We first do a thread local
//  * computation, then we do a warp-level partial reduction, and then we do a final
//  * warp-reduction across the 1 warp that has the partial results from each warp.
//  **************************************************************************************/

// // We'll define a function blockReduce that can do sum or max over an array of length K
// // across an entire blockDim.x threads. For simplicity we assume blockDim.x <= 1024
// // and we unify the results in the first warp.

// template <ReduceType RT, int K>
// __device__ __forceinline__ void blockReduce(float vals[K]) {
//   // Step 1: Warp-level reduce inside each warp
//   warpReduce<RT,K>(vals);

//   // Step 2: The "lane 0" of each warp writes to shared memory
//   static __shared__ float smem[WARP_SIZE * K]; // up to 32*K
//   int laneId = threadIdx.x & (WARP_SIZE - 1);
//   int warpId = threadIdx.x / WARP_SIZE;        // which warp in block

// #pragma unroll
//   for (int i = 0; i < K; i++) {
//     if (laneId == 0) {
//       smem[warpId * K + i] = vals[i];
//     }
//   }
//   __syncthreads();

//   // The number of warps in this block:
//   int nwarp = blockDim.x / WARP_SIZE; // assume blockDim.x is multiple of 32 for simplicity

//   // Step 3: The first warp does a warp-level reduce over those partial sums
//   if (warpId == 0) {
//     float tmp[K];
//     // load from smem
//     if (threadIdx.x < nwarp) {
//       for (int i = 0; i < K; i++) {
//         tmp[i] = smem[threadIdx.x * K + i];
//       }
//     } else {
//       for (int i = 0; i < K; i++) {
//         tmp[i] = (RT == ReduceType::kSum ? 0.f : REDUCE_FLOAT_INF_NEG);
//       }
//     }
//     // warp reduce among these nwarp values
//     warpReduce<RT,K>(tmp);

//     // broadcast the final results
//     for (int i = 0; i < K; i++) {
//       smem[i] = tmp[i]; // store in first warp
//     }
//   }
//   __syncthreads();

//   // Step 4: read back final from smem
//   for (int i = 0; i < K; i++) {
//     vals[i] = smem[i];
//   }
// }

// /**************************************************************************************
//  * Fused kernel for small sequence length (<32)
//  *
//  * This is already implemented for reference. It uses warp-level reduction (no block sync).
//  **************************************************************************************/
// template <typename T, int block_dim, int ele_per_thread>
// __global__ void ker_attn_softmax_lt32(T *inp, const T *attn_mask,
//                                       int from_len, int to_len, bool mask_future) {
//   // gridDim.x = # of tokens per row-chunk
//   // gridDim.y = batch_id
//   // gridDim.z = head_id

//   int batch_id = blockIdx.y;
//   int head_id  = blockIdx.z;
//   const int nhead = gridDim.z;

//   // We'll handle 1 token per warp in the from_len dimension
//   const int token_per_reduce = 1;  // (like a block in bigger kernels, but here is warp-level)

//   // We'll use CUB's BlockLoad/BlockStore in warp-size to read & write
//   typedef cub::BlockLoad<T, block_dim, ele_per_thread, cub::BLOCK_LOAD_VECTORIZE> BlockLoad;
//   __shared__ typename BlockLoad::TempStorage ts_load;
//   typedef cub::BlockStore<T, block_dim, ele_per_thread, cub::BLOCK_STORE_VECTORIZE> BlockStore;
//   __shared__ typename BlockStore::TempStorage ts_store;

//   // If we have an attn_mask, we load it once (the row?), plus each thread gets a chunk
//   T mval[ele_per_thread];
//   if (attn_mask) {
//     attn_mask += batch_id * to_len; // each batch row
//     BlockLoad(ts_load).Load(attn_mask, mval, to_len, REDUCE_FLOAT_INF_NEG);
//   }

//   // Advance "inp" to the correct sub-block for this batch, head
//   // shape: [batch_size, nhead, from_len, to_len]
//   inp += (batch_id * nhead + head_id) * (from_len * to_len);

//   // for each token (token_id in [0.. from_len)), but distributed over gridDim.x
//   for (int token_id = blockIdx.x * token_per_reduce; token_id < from_len; token_id += gridDim.x * token_per_reduce) {
//     // Load this row
//     T inp_val[token_per_reduce][ele_per_thread];
//     for (int i = 0; i < token_per_reduce && (token_id + i) < from_len; i++) {
//       BlockLoad(ts_load).Load(inp + (token_id + i) * to_len, inp_val[i], to_len, REDUCE_FLOAT_INF_NEG);
//     }

//     // (1) compute max
//     float val[token_per_reduce][ele_per_thread];
//     float l_max[token_per_reduce];
//     for (int i = 0; i < token_per_reduce; i++) {
//       l_max[i] = REDUCE_FLOAT_INF_NEG;
//       for (int j = 0; j < ele_per_thread; j++) {
//         float tmp = (float)inp_val[i][j];
//         if (attn_mask) {
//           tmp += (float)mval[j];
//         }
//         // If mask_future is true, we want to mask positions beyond the current token index
//         // i.e. j-th col > token_id + i => set to -inf
//         int col_idx = block_dim * threadIdx.x + j; // or simpler: ele_per_thread*threadIdx.x + j
//         if (mask_future) {
//           // The column index = (blockDim.x=warp=32 * ele_per_thread) * ??? + j
//           // but we can replicate the approach from the sample code:
//           int global_col_idx = ele_per_thread * threadIdx.x + j;
//           if (global_col_idx > (token_id + i)) {
//             tmp = REDUCE_FLOAT_INF_NEG;
//           }
//         }
//         val[i][j] = tmp;
//         l_max[i]  = fmaxf(l_max[i], tmp);
//       }
//     }

//     // warp reduce max
// #pragma unroll
//     for (int offset = 16; offset > 0; offset >>= 1) {
//       for (int i = 0; i < token_per_reduce; i++) {
//         float other = __shfl_xor_sync(0xffffffff, l_max[i], offset);
//         l_max[i] = fmaxf(l_max[i], other);
//       }
//     }

//     // (2) compute sum
//     float l_sum[token_per_reduce];
//     for (int i = 0; i < token_per_reduce; i++) {
//       l_sum[i] = 0.0f;
//       // exponent + sum
//       for (int j = 0; j < ele_per_thread; j++) {
//         val[i][j] = expf(val[i][j] - l_max[i]);
//         l_sum[i] += val[i][j];
//       }
//     }

//     // warp reduce sum
// #pragma unroll
//     for (int offset = 16; offset > 0; offset >>= 1) {
//       for (int i = 0; i < token_per_reduce; i++) {
//         float other = __shfl_xor_sync(0xffffffff, l_sum[i], offset);
//         l_sum[i] += other;
//       }
//     }

//     // (3) final result
//     for (int i = 0; i < token_per_reduce && (token_id + i) < from_len; i++) {
//       float inv_sum = 1.f / (l_sum[i] + EPSILON);
//       for (int j = 0; j < ele_per_thread; j++) {
//         inp_val[i][j] = (T)(val[i][j] * inv_sum);
//       }
//       // store
//       BlockStore(ts_store).Store(inp + (token_id + i) * to_len, inp_val[i], to_len);
//     }
//   }
// }

// /**************************************************************************************
//  * Fused kernel for general sequence length (>=32)
//  *
//  * This implements a block-level approach with blockReduce for max and sum.
//  *
//  * We have placeholders for code in lines marked with "BEGIN ASSIGN3_1" and "END ASSIGN3_1".
//  **************************************************************************************/
// template <typename T, int block_dim, int ele_per_thread>
// __global__ void ker_attn_softmax(T *inp, const T *attn_mask,
//                                  int from_len, int to_len, bool mask_future) {
//   // blockIdx.x => can handle multiple tokens in the from_len dimension
//   // blockIdx.y => batch_id
//   // blockIdx.z => head_id

//   int batch_id = blockIdx.y;
//   int head_id  = blockIdx.z;
//   const int nhead = gridDim.z;
//   // We'll process 1 token per block for "from_len" dimension
//   const int token_per_reduce = 1;

//   // We'll use CUB to load/store in a vectorized manner
//   typedef cub::BlockLoad<T,  block_dim, ele_per_thread, cub::BLOCK_LOAD_VECTORIZE>   BlockLoad;
//   __shared__ typename BlockLoad::TempStorage ts_load;
//   typedef cub::BlockStore<T, block_dim, ele_per_thread, cub::BLOCK_STORE_VECTORIZE>  BlockStore;
//   __shared__ typename BlockStore::TempStorage ts_store;

//   // If attn_mask is not null => shape [batch_size, to_len]
//   // load attn_mask so each thread has partial chunk
//   T mval[ele_per_thread];
//   if (attn_mask) {
//     attn_mask += (batch_id * to_len);
//     BlockLoad(ts_load).Load(attn_mask, mval, to_len, REDUCE_FLOAT_INF_NEG);
//   }

//   // offset pointer to the correct place in "inp"
//   inp += (batch_id * nhead + head_id) * (from_len * to_len);

//   // Loop over the from_len dimension in chunks of token_per_reduce
//   for (int token_id = blockIdx.x * token_per_reduce; token_id < from_len;
//        token_id += gridDim.x * token_per_reduce) {

//     // Load the row [to_len] from global memory
//     T inp_val[token_per_reduce][ele_per_thread];
//     for (int i = 0; i < token_per_reduce && (token_id + i) < from_len; i++) {
//       BlockLoad(ts_load).Load(inp + (token_id + i) * to_len, inp_val[i], to_len, REDUCE_FLOAT_INF_NEG);
//     }

//     // =======================
//     // step 1. compute max
//     // thread local
//     // =======================
//     /************ BEGIN ASSIGN3_1 (step 1) ************/
//     float val[token_per_reduce][ele_per_thread];
//     float l_max[token_per_reduce];
//     #pragma unroll
//     for (int i = 0; i < token_per_reduce; i++) {
//       l_max[i] = REDUCE_FLOAT_INF_NEG;
//       #pragma unroll
//       for (int j = 0; j < ele_per_thread; j++) {
//         float tmp = (float)inp_val[i][j];
//         if (attn_mask) {
//           tmp += (float)mval[j];  // add mask
//         }
//         if (mask_future) {
//           // we want to set positions col_idx > token_id + i => -inf
//           int global_col = threadIdx.x * ele_per_thread + j;  // 0.. to_len-1
//           if (global_col > (token_id + i)) {
//             tmp = REDUCE_FLOAT_INF_NEG;
//           }
//         }
//         val[i][j] = tmp;
//         l_max[i]  = fmaxf(l_max[i], tmp);
//       }
//     }
//     // block reduce for max
//     blockReduce<ReduceType::kMax, token_per_reduce>(l_max);

//     // shared memory to store final block-level max
//     __shared__ float s_max[token_per_reduce];
//     if (threadIdx.x == 0) {
//       for (int i = 0; i < token_per_reduce; i++) {
//         s_max[i] = l_max[i];
//       }
//     }
//     __syncthreads();
//     /************ END ASSIGN3_1 (step 1) ************/

//     // =======================
//     // step 2. compute sum of exp
//     // =======================
//     /************ BEGIN ASSIGN3_1 (step 2) ************/
//     float l_sum[token_per_reduce];
//     #pragma unroll
//     for (int i = 0; i < token_per_reduce; i++) {
//       l_sum[i] = 0.f;
//       float max_val = s_max[i];
//       for (int j = 0; j < ele_per_thread; j++) {
//         // exponent
//         val[i][j] = expf(val[i][j] - max_val);
//         l_sum[i] += val[i][j];
//       }
//     }
//     // block reduce sum
//     blockReduce<ReduceType::kSum, token_per_reduce>(l_sum);
//     // store in shared memory
//     __shared__ float s_sum[token_per_reduce];
//     if (threadIdx.x == 0) {
//       for (int i = 0; i < token_per_reduce; i++) {
//         s_sum[i] = 1.f / (l_sum[i] + EPSILON);
//       }
//     }
//     __syncthreads();
//     /************ END ASSIGN3_1 (step 2) ************/

//     // =======================
//     // step 3. write final
//     // =======================
//     /************ BEGIN ASSIGN3_1 (step 3) ************/
//     for (int i = 0; i < token_per_reduce && (token_id + i) < from_len; i++) {
//       float inv_sum = s_sum[i];  // the reciprocal of sum
//       for (int j = 0; j < ele_per_thread; j++) {
//         inp_val[i][j] = (T)(val[i][j] * inv_sum);
//       }
//       // store back to global
//       BlockStore(ts_store).Store(inp + (token_id + i) * to_len, inp_val[i], to_len);
//     }
//     /************ END ASSIGN3_1 (step 3) ************/

//   }  // for token_id
// }

// /**************************************************************************************
//  * Host launcher for the forward pass
//  *
//  * We choose different kernel configs based on to_len.
//  **************************************************************************************/
// extern "C"
// void launch_attn_softmax(float *inp, const float *attn_mask,
//                          int batch_size, int nhead,
//                          int from_len, int to_len,
//                          bool mask_future,
//                          cudaStream_t stream)
// {
//   // Move data to device
//   // We'll do a simple approach: allocate device buffers, copy, run kernel, copy back, free
//   // For real code, you might keep them on device for repeated calls.

//   size_t float_size = sizeof(float);
//   size_t inp_size   = (size_t)batch_size * nhead * from_len * to_len * float_size;
//   float *d_inp      = nullptr;
//   cudaMalloc(&d_inp, inp_size);
//   cudaMemcpy(d_inp, inp, inp_size, cudaMemcpyHostToDevice);

//   float *d_mask     = nullptr;
//   if (attn_mask) {
//     size_t mask_size  = (size_t)batch_size * to_len * float_size;
//     cudaMalloc(&d_mask, mask_size);
//     cudaMemcpy(d_mask, attn_mask, mask_size, cudaMemcpyHostToDevice);
//   }

//   // Set grid & block dims
//   // Usually from_len can be up to e.g. 1024
//   // We'll pick the kernel style:
//   dim3 grid_dim(1, batch_size, nhead);

//   // If sequence length <= 32, use the "lt32" kernel
//   if (to_len <= 32) {
//     // e.g. block_dim.x=32 => a warp
//     ker_attn_softmax_lt32<float, 32, 1><<<grid_dim, 32, 0, stream>>>(
//         d_inp, d_mask, from_len, to_len, mask_future);
//   }
//   else if (to_len <= 64) {
//     ker_attn_softmax_lt32<float, 32, 2><<<grid_dim, 32, 0, stream>>>(
//         d_inp, d_mask, from_len, to_len, mask_future);
//   }
//   else {
//     // We'll pick an approach for the bigger range
//     // For example, up to 512 we can pick block=256 or something
//     // We'll also do a "grid_dim.x" to process multiple tokens
//     if      (to_len <= 128) { grid_dim.x = 16;  ker_attn_softmax<float, 64,   2><<<grid_dim,  64, 0, stream>>>(d_inp, d_mask, from_len, to_len, mask_future); }
//     else if (to_len <= 256) { grid_dim.x = 32;  ker_attn_softmax<float, 128,  2><<<grid_dim, 128, 0, stream>>>(d_inp, d_mask, from_len, to_len, mask_future); }
//     else if (to_len <= 512) { grid_dim.x = 64;  ker_attn_softmax<float, 256,  2><<<grid_dim, 256, 0, stream>>>(d_inp, d_mask, from_len, to_len, mask_future); }
//     else if (to_len <=1024) { grid_dim.x =128;  ker_attn_softmax<float, 512,  2><<<grid_dim, 512, 0, stream>>>(d_inp, d_mask, from_len, to_len, mask_future); }
//     else if (to_len <=2048) { grid_dim.x =256;  ker_attn_softmax<float,1024,  2><<<grid_dim,1024, 0, stream>>>(d_inp, d_mask, from_len, to_len, mask_future); }
//     else {
//       fprintf(stderr, "launch_attn_softmax: to_len=%d not supported\n", to_len);
//       cudaFree(d_inp);
//       if (d_mask) cudaFree(d_mask);
//       return;
//     }
//   }

//   cudaMemcpy(inp, d_inp, inp_size, cudaMemcpyDeviceToHost);

//   cudaFree(d_inp);
//   if (d_mask) cudaFree(d_mask);

//   cudaDeviceSynchronize();
//   cudaError_t e = cudaGetLastError();
//   if (e != cudaSuccess) {
//     fprintf(stderr, "launch_attn_softmax kernel error: %s\n", cudaGetErrorString(e));
//     return;
//   }
// }

// /**************************************************************************************
//  * Backward pass kernel
//  *
//  * "ker_attn_softmax_bw": dSoftmax = Softmax * (dOut - sum(dOut * Softmax))
//  *
//  * We typically do this with a warp approach or block approach. The user can set
//  * multiple warps per block. We store partial sums in registers, do a warp shuffle sum.
//  **************************************************************************************/

// template <typename T, int ITERATIONS>
// __global__ void ker_attn_softmax_bw(T *grad, const T *inp, int softmax_length) {
//   //
//   // each block handles "warps_per_block" rows of shape? we are told:
//   //   blockDim.x = WARP_SIZE
//   //   blockDim.y = warps_per_block
//   //   gridDim.x  = total_rows / warps_per_block
//   //
//   // The "rows" is presumably batch_size * nhead * from_len
//   // The "softmax_length" is "to_len"
//   // We'll do up to ITERATIONS steps in a for-loop with #pragma unroll
//   //
//   // so threadIdx.y => which warp in this block
//   // batch_idx => blockIdx.x * blockDim.y + threadIdx.y
//   //
//   int batch_idx = blockIdx.x * blockDim.y + threadIdx.y;
//   int offset    = batch_idx * softmax_length + threadIdx.x;  // position in "grad, inp"

//   grad += offset;
//   inp  += offset;

//   T  grad_reg[ITERATIONS];
//   T  inp_reg[ITERATIONS];
//   float sum = 0.f;

// #pragma unroll
//   for (int i = 0; i < ITERATIONS; i++) {
//     int col = threadIdx.x + i*WARP_SIZE;
//     if (col < softmax_length) {
//       grad_reg[i] = grad[i*WARP_SIZE];
//       inp_reg[i]  = inp[i*WARP_SIZE];
//       sum += (float)grad_reg[i] * (float)inp_reg[i];
//     }
//   }

//   // warp reduce sum of "sum"
//   cg::thread_block b      = cg::this_thread_block();
//   cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

// #pragma unroll
//   for (int mask = WARP_SIZE >>1; mask > 0; mask >>=1) {
//     float other = g.shfl_xor(sum, mask);
//     sum += other;
//   }

//   // each thread now does grad[i] = inp[i]*(grad[i]-sum)
// #pragma unroll
//   for (int i = 0; i < ITERATIONS; i++) {
//     int col = threadIdx.x + i*WARP_SIZE;
//     if (col < softmax_length) {
//       float g_  = (float)grad_reg[i];
//       float in_ = (float)inp_reg[i];
//       grad[i*WARP_SIZE] = (T)( in_ * (g_ - sum) );
//     }
//   }
// }


// /**************************************************************************************
//  * Host launcher for backward
//  *
//  * We pick different "ITERATIONS" based on the softmax length. E.g. if to_len<=32 => ITERATIONS=1
//  * else if <=64 => ITERATIONS=2, etc...
//  **************************************************************************************/
// extern "C"
// void launch_attn_softmax_bw(float *out_grad,
//                             const float *soft_inp,
//                             int rows,         // e.g. batch_size*nhead*from_len
//                             int softmax_len,  // e.g. to_len
//                             cudaStream_t stream)
// {
//   // We'll do a typical approach: device allocate, copy, kernel, copy back
//   size_t float_size = sizeof(float);
//   size_t total_size = (size_t)rows * softmax_len * float_size;

//   float *d_grad = nullptr;
//   cudaMalloc(&d_grad, total_size);
//   cudaMemcpy(d_grad, out_grad, total_size, cudaMemcpyHostToDevice);

//   float *d_inp = nullptr;
//   cudaMalloc(&d_inp,  total_size);
//   cudaMemcpy(d_inp, soft_inp, total_size, cudaMemcpyHostToDevice);

//   // we define blockDim.x=32, blockDim.y=warps_per_block=4
//   // gridDim.x=(rows+(warps_per_block-1))/warps_per_block
//   const int warps_per_block = 4;
//   dim3 block_dim(WARP_SIZE, warps_per_block);
//   dim3 grid_dim( (rows + warps_per_block -1)/ warps_per_block );

//   // pick "ITERATIONS" so that WARP_SIZE*ITERATIONS >= softmax_len
//   // The assignment hints we do something like a series of if statements for max seq len up to 2048

//   if (softmax_len <= 32) {
//     ker_attn_softmax_bw<float,1><<<grid_dim, block_dim, 0, stream>>>(
//         d_grad, d_inp, softmax_len);
//   }
//   else if (softmax_len <=64) {
//     ker_attn_softmax_bw<float,2><<<grid_dim, block_dim, 0, stream>>>(
//         d_grad, d_inp, softmax_len);
//   }
//   else if (softmax_len <=128) {
//     ker_attn_softmax_bw<float,4><<<grid_dim, block_dim, 0, stream>>>(
//         d_grad, d_inp, softmax_len);
//   }
//   else if (softmax_len <=256) {
//     ker_attn_softmax_bw<float,8><<<grid_dim, block_dim, 0, stream>>>(
//         d_grad, d_inp, softmax_len);
//   }
//   else if (softmax_len <=384) {
//     // We'll do 12, which is not a power of 2 but can handle up to 384
//     ker_attn_softmax_bw<float,12><<<grid_dim, block_dim, 0, stream>>>(
//         d_grad, d_inp, softmax_len);
//   }
//   else if (softmax_len <=512) {
//     ker_attn_softmax_bw<float,16><<<grid_dim, block_dim, 0, stream>>>(
//         d_grad, d_inp, softmax_len);
//   }
//   else if (softmax_len <=768) {
//     // We'll do 24
//     ker_attn_softmax_bw<float,24><<<grid_dim, block_dim, 0, stream>>>(
//         d_grad, d_inp, softmax_len);
//   }
//   else if (softmax_len <=1024) {
//     ker_attn_softmax_bw<float,32><<<grid_dim, block_dim, 0, stream>>>(
//         d_grad, d_inp, softmax_len);
//   }
//   else if (softmax_len <=2048) {
//     ker_attn_softmax_bw<float,64><<<grid_dim, block_dim, 0, stream>>>(
//         d_grad, d_inp, softmax_len);
//   }
//   else {
//     fprintf(stderr, "launch_attn_softmax_bw: softmax_len=%d not supported.\n", softmax_len);
//     cudaFree(d_grad);
//     cudaFree(d_inp);
//     return;
//   }

//   cudaMemcpy(out_grad, d_grad, total_size, cudaMemcpyDeviceToHost);

//   cudaFree(d_grad);
//   cudaFree(d_inp);

//   cudaDeviceSynchronize();
//   cudaError_t e = cudaGetLastError();
//   if (e != cudaSuccess) {
//     fprintf(stderr, "launch_attn_softmax_bw kernel error: %s\n", cudaGetErrorString(e));
//     return;
//   }
// }

