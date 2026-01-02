#include <cuda_fp16.h>
#include <stdint.h>

__device__ inline uint8_t quant_fp4(float x) {
    x = fmaxf(fminf(x, 7.0f), -8.0f); // Clamp to representable range
    int q = __float2int_rn(x);        // Scale and round
    return (uint8_t)(q & 0x0F);       // Keep only 4 bits
}

__global__ void pack_fp4_kernel(const half* in, uint8_t* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * 2;
    if (idx + 1 < n) {
        float x0 = __half2float(in[idx]);
        float x1 = __half2float(in[idx + 1]);
        out[i] = (quant_fp4(x0) << 4) | quant_fp4(x1);
    }
}


void pack_fp4_cuda(const half* in, uint8_t* out, int n) {
    int threads = 256;
    int blocks = (n / 2 + threads - 1) / threads;
    pack_fp4_kernel<<<blocks, threads>>>(in, out, n);
}