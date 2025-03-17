/*****************************************************************************
 * layernorm_kernel_debug_rowBlock.cu
 *
 * "rowBlock" style, line-by-line debug kernel.
 * - Forward + Backward, (row,col) => prints out partial debug info
 * - variable names: (ln_res, vars, means, inp, gamma, beta, batch_size, hidden_dim)
 *
 * This kernel does not use float4. The hidden_dim is the actual row width in floats.
 * Each block is one row => blockIdx.x = row, threadIdx.x = col.
 *****************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

static constexpr float LN_EPSILON = 1e-8f;

/******************************************************************************
 * Forward kernel: ker_layer_norm
 * Each block => one row. We do partial sums for mean,var, then do LN transform.
 * ln_res[row,col] = (x - mean)*inv_std*g + b
 ******************************************************************************/
__global__ void ker_layer_norm(
    float *ln_res,    // [batch_size, hidden_dim] LN output
    float *vars,      // [batch_size] sample variance
    float *means,     // [batch_size] row means
    const float *inp, // [batch_size, hidden_dim] input
    const float *gamma,
    const float *beta,
    int batch_size,
    int hidden_dim
)
{
  int row = blockIdx.x;
  if(row >= batch_size) return;

  int col = threadIdx.x;
  if(col >= hidden_dim) return;

  // 1) load x, gamma, beta
  float x = inp[row * hidden_dim + col];
  float g = gamma[col];
  float b_ = beta[col];

  // debug print
  printf("[FWD ker_layer_norm] row=%d col=%d => inp=%.4f gamma=%.4f beta=%.4f\n",
         row, col, x, g, b_);

  // 2) partial sums for mean,var
  float local_sum  = x;
  float local_sum2 = x*x;

  // shared memory: sum_x, sum_x2
  extern __shared__ float sdata[];
  // sdata[0..hidden_dim-1] => sum_x
  // sdata[hidden_dim..2*hidden_dim-1] => sum_x2
  sdata[col] = local_sum;
  sdata[col + hidden_dim] = local_sum2;
  __syncthreads();

  // block-wide reduce
  int half = hidden_dim / 2;
  while(half > 0) {
    if(col < half) {
      sdata[col]             += sdata[col + half];
      sdata[col + hidden_dim] += sdata[col + hidden_dim + half];
    }
    __syncthreads();
    half /= 2;
  }

  // thread 0 => final sums => mean,var
  float mean_val = 0.f, var_val = 0.f;
  if(col == 0) {
    float sumX  = sdata[0];
    float sumX2 = sdata[hidden_dim];
    float n     = float(hidden_dim);

    mean_val = sumX / n;
    float numerator = sumX2 - (sumX*sumX / n);
    float denom = (n > 1.f ? (n - 1.f) : 1.f);  // sample variance
    var_val = numerator / denom;
    if(var_val < 0.f) {
      printf("[FWD ker_layer_norm] row=%d => var<0 => clamp to 0\n", row);
      var_val = 0.f;
    }
    float inv_std = rsqrtf(var_val + LN_EPSILON);

    means[row] = mean_val;
    vars[row]  = var_val;

    printf("[FWD ker_layer_norm] row=%d => sumX=%.4f sumX2=%.4f mean=%.4f var=%.4f invStd=%.4f\n",
           row, sumX, sumX2, mean_val, var_val, inv_std);
  }
  __syncthreads();

  // broadcast mean, var (stored in shared memory)
  __shared__ float sMean, sVar;
  if(col == 0) {
    sMean = mean_val;
    sVar  = var_val;
  }
  __syncthreads();

  float inv_std = rsqrtf(sVar + LN_EPSILON);

  // 3) LN => (x - mean)*inv_std*g + b
  float val = (x - sMean) * inv_std * g + b_;
  ln_res[row * hidden_dim + col] = val;

  printf("[FWD ker_layer_norm] row=%d col=%d => LNout=%.4f\n",
         row, col, val);
}


/******************************************************************************
 * Backward: 2 kernels => (1) dGamma,dBeta, (2) dInp
 ******************************************************************************/

/** ker_ln_bw_dgamma_dbetta
 * Each thread => single (row,col). We do partial atomicAdd for gamma_grad,beta_grad.
 */
__global__ void ker_ln_bw_dgamma_dbetta(
    float *gamma_grad,
    float *beta_grad,
    const float *out_grad,
    const float *inp,
    const float *gamma,
    const float *beta,
    const float *vars,
    const float *means,
    int batch_size,
    int hidden_dim
)
{
  int row = blockIdx.x;
  if(row >= batch_size) return;

  int col = threadIdx.x;
  if(col >= hidden_dim) return;

  int offset = row * hidden_dim + col;

  float var_val  = vars[row];
  float mean_val = means[row];
  float inv_std  = rsqrtf(var_val + LN_EPSILON);

  float go   = out_grad[offset];
  float x    = inp[offset];
  float xhat = (x - mean_val) * inv_std;

  float dBeta  = go;
  float dGamma = go * xhat;

  // debug
  printf("[BWD dGammaBeta] row=%d col=%d => go=%.4f x=%.4f xhat=%.4f => dBeta=%.4f dGamma=%.4f\n",
         row, col, go, x, xhat, dBeta, dGamma);

  // partial atomicAdd => each col accum
  atomicAdd(&beta_grad[col],  dBeta);
  atomicAdd(&gamma_grad[col], dGamma);
}


/** ker_ln_bw_dinp
 * block => row
 * we do row-wide partial sums => sum(dxhat), sum(dxhat*xhat), then final d_in
 */
__global__ void ker_ln_bw_dinp(
    float *inp_grad,
    const float *out_grad,
    const float *inp,
    const float *gamma,
    const float *beta,
    const float *vars,
    const float *means,
    int batch_size,
    int hidden_dim
)
{
  int row = blockIdx.x;
  if(row >= batch_size) return;

  int col = threadIdx.x;
  if(col >= hidden_dim) return;

  int offset = row * hidden_dim + col;

  float var_val  = vars[row];
  float mean_val = means[row];
  float inv_std  = rsqrtf(var_val + LN_EPSILON);

  float n = float(hidden_dim);

  // standard LN => we use 1.f / n in sub-averages
  float scale = 1.f / n;

  // pass1 => sum(dxhat), sum(dxhat*xhat)
  float go = out_grad[offset];
  float gm = gamma[col];
  float dxh= go * gm; // dXhat
  float x   = inp[offset];
  float xhat= (x - mean_val) * inv_std;

  // debug
  printf("[BWD inp pass1] row=%d col=%d => go=%.4f gm=%.4f x=%.4f xhat=%.4f dxh=%.4f\n",
         row, col, go, gm, x, xhat, dxh);

  extern __shared__ float sdata[];
  // sdata[0..hidden_dim-1] => dxhat
  // sdata[hidden_dim..(2*hidden_dim-1)] => dxhat*xhat
  sdata[col]             = dxh;
  sdata[col + hidden_dim]= dxh * xhat;
  __syncthreads();

  // block reduce
  int half = hidden_dim / 2;
  while(half > 0) {
    if(col < half) {
      sdata[col]              += sdata[col + half];
      sdata[col + hidden_dim] += sdata[col + hidden_dim + half];
    }
    __syncthreads();
    half >>= 1;
  }

  float tot_dxhat   = sdata[0];
  float tot_dxhat_x = sdata[hidden_dim];

  if(col == 0) {
    printf("[BWD inp pass1 sum] row=%d => sum_dxhat=%.4f sum_dxhat_x=%.4f\n",
           row, tot_dxhat, tot_dxhat_x);
  }
  __syncthreads();

  // pass2 => final d_in = (dxh - part)* inv_std
  float part = (tot_dxhat + xhat * tot_dxhat_x) * scale;
  float d_in = (dxh - part) * inv_std;
  inp_grad[offset] = d_in;

  printf("[BWD inp pass2] row=%d col=%d => dxh=%.4f xhat=%.4f part=%.4f => d_in=%.4f\n",
         row, col, dxh, xhat, part, d_in);
}


/******************************************************************************
 * Host function for LN forward => launch_layernorm
 ******************************************************************************/
extern "C"
void launch_layernorm(
    float *ln_res,
    float *vars,
    float *means,
    const float *inp,
    const float *gamma,
    const float *beta,
    int batch_size,
    int hidden_dim,
    cudaStream_t stream
)
{
  // Each block => row, blockDim.x => hidden_dim
  dim3 grid(batch_size), block(hidden_dim);
  // shared memory => 2 * hidden_dim floats
  size_t smem = block.x * 2 * sizeof(float);

  ker_layer_norm<<<grid, block, smem, stream>>>(
      ln_res, vars, means, inp, gamma, beta,
      batch_size, hidden_dim
  );

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  if(err != cudaSuccess) {
    fprintf(stderr, "[launch_layernorm] kernel error: %s\n", cudaGetErrorString(err));
  }
}


/******************************************************************************
 * Host function for LN backward => launch_layernorm_bw
 ******************************************************************************/
extern "C"
void launch_layernorm_bw(
    float *gamma_grad,
    float *beta_grad,
    float *inp_grad,
    const float *out_grad,
    const float *inp,
    const float *gamma,
    const float *beta,
    const float *vars,
    const float *means,
    int batch_size,
    int hidden_dim,
    cudaStream_t stream_1,
    cudaStream_t stream_2
)
{
  // 1) zero out gamma_grad, beta_grad
  cudaMemsetAsync(gamma_grad, 0, hidden_dim * sizeof(float), stream_1);
  cudaMemsetAsync(beta_grad,  0, hidden_dim * sizeof(float), stream_1);

  // 2) grad(gamma,beta)
  {
    dim3 grid(batch_size), block(hidden_dim);
    ker_ln_bw_dgamma_dbetta<<<grid, block, 0, stream_1>>>(
        gamma_grad, beta_grad,
        out_grad, inp, gamma, beta,
        vars, means,
        batch_size, hidden_dim
    );
  }

  // 3) grad(inp)
  {
    dim3 grid2(batch_size), block2(hidden_dim);
    size_t smem2 = block2.x * 2 * sizeof(float);
    ker_ln_bw_dinp<<<grid2, block2, smem2, stream_2>>>(
        inp_grad, out_grad, inp, gamma, beta,
        vars, means,
        batch_size, hidden_dim
    );
  }

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  if(err != cudaSuccess) {
    fprintf(stderr, "[launch_layernorm_bw] kernel error: %s\n", cudaGetErrorString(err));
  }
}

