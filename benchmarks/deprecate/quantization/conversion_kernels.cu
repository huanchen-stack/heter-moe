#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace conversion {

// ============================================================================
// Helper Functions and Constants
// ============================================================================

constexpr int INT4_MIN = -8;
constexpr int INT4_MAX = 7;
constexpr int WARP_SIZE = 32;
constexpr int THREADS_PER_GROUP = 16;

// INT4 bit packing: 2 values per byte (lower 4 bits: val0, upper 4 bits: val1)
__device__ __forceinline__ uint8_t pack_int4(int8_t val0, int8_t val1) {
  return (uint8_t)((val0 & 0x0F) | ((val1 & 0x0F) << 4));
}

// INT4 unpacking with sign extension
__device__ __forceinline__ void unpack_int4(uint8_t packed, int8_t& val0, int8_t& val1) {
  val0 = (packed & 0x0F);
  val1 = ((packed >> 4) & 0x0F);
  // Sign extension: if bit 3 is set, extend sign to full byte
  if (val0 & 0x08) val0 |= 0xF0;
  if (val1 & 0x08) val1 |= 0xF0;
}

// Warp-level reduction for finding maximum
__device__ __forceinline__ float warp_reduce_max(float val) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
  }
  return val;
}

// Group-level reduction (16 threads per group)
__device__ __forceinline__ float group_reduce_max(float val, int lane_id) {
  unsigned mask = (lane_id >= 16) ? 0xFFFF0000 : 0x0000FFFF;
  val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 4));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 2));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 1));
  return val;
}

// ============================================================================
// FP16 -> FP8 Quantization (Per-Token Group)
// ============================================================================

template <typename T>
__global__ void fp16_to_fp8_kernel(
    const T* __restrict__ input,
    __nv_fp8_e4m3* __restrict__ output,
    float* __restrict__ scales,
    const int group_size,
    const int num_groups,
    const float eps,
    const float fp8_max) {

  const int local_group_id = threadIdx.x / THREADS_PER_GROUP;
  const int lane_id = threadIdx.x % THREADS_PER_GROUP;
  const int global_group_id = blockIdx.x * (blockDim.x / THREADS_PER_GROUP) + local_group_id;

  if (global_group_id >= num_groups) return;

  const int64_t offset = global_group_id * group_size;
  const T* group_input = input + offset;
  __nv_fp8_e4m3* group_output = output + offset;

  // Find group max
  float local_max = eps;
  for (int i = lane_id; i < group_size; i += THREADS_PER_GROUP) {
    float val = fabsf(__half2float(group_input[i]));
    local_max = fmaxf(local_max, val);
  }
  local_max = group_reduce_max(local_max, lane_id);

  // Compute scale
  float scale = local_max / fp8_max;
  if (lane_id == 0) {
    scales[global_group_id] = scale;
  }

  // Quantize
  for (int i = lane_id; i < group_size; i += THREADS_PER_GROUP) {
    float val = __half2float(group_input[i]) / scale;
    val = fminf(fmaxf(val, -fp8_max), fp8_max);
    group_output[i] = __nv_fp8_e4m3(val);
  }
}

// ============================================================================
// FP16 -> INT4 Quantization (Per-Token Group with Bit Packing)
// ============================================================================

template <typename T>
__global__ void fp16_to_int4_kernel(
    const T* __restrict__ input,
    uint8_t* __restrict__ output_packed,
    float* __restrict__ scales,
    const int group_size,
    const int num_groups,
    const float eps) {

  const int local_group_id = threadIdx.x / THREADS_PER_GROUP;
  const int lane_id = threadIdx.x % THREADS_PER_GROUP;
  const int global_group_id = blockIdx.x * (blockDim.x / THREADS_PER_GROUP) + local_group_id;

  if (global_group_id >= num_groups) return;

  const int64_t offset = global_group_id * group_size;
  const T* group_input = input + offset;
  uint8_t* group_output = output_packed + (offset / 2);  // 2 values per byte

  // Find group max
  float local_max = eps;
  for (int i = lane_id; i < group_size; i += THREADS_PER_GROUP) {
    float val = fabsf(__half2float(group_input[i]));
    local_max = fmaxf(local_max, val);
  }
  local_max = group_reduce_max(local_max, lane_id);

  // Compute scale
  float scale = local_max / (float)INT4_MAX;
  if (lane_id == 0) {
    scales[global_group_id] = scale;
  }

  // Quantize and pack (2 values per byte)
  for (int i = lane_id * 2; i < group_size; i += THREADS_PER_GROUP * 2) {
    // Quantize two values
    float val0 = __half2float(group_input[i]) / scale;
    int8_t q0 = (int8_t)fminf(fmaxf(roundf(val0), (float)INT4_MIN), (float)INT4_MAX);

    int8_t q1 = 0;
    if (i + 1 < group_size) {
      float val1 = __half2float(group_input[i + 1]) / scale;
      q1 = (int8_t)fminf(fmaxf(roundf(val1), (float)INT4_MIN), (float)INT4_MAX);
    }

    // Pack into single byte
    group_output[i / 2] = pack_int4(q0, q1);
  }
}

// ============================================================================
// INT4 -> FP8 Dequantization
// ============================================================================

__global__ void int4_to_fp8_kernel(
    const uint8_t* __restrict__ input_packed,
    __nv_fp8_e4m3* __restrict__ output,
    const float* __restrict__ scales,
    const int group_size,
    const int num_groups) {

  const int local_group_id = threadIdx.x / THREADS_PER_GROUP;
  const int lane_id = threadIdx.x % THREADS_PER_GROUP;
  const int global_group_id = blockIdx.x * (blockDim.x / THREADS_PER_GROUP) + local_group_id;

  if (global_group_id >= num_groups) return;

  const int64_t offset = global_group_id * group_size;
  const uint8_t* group_input = input_packed + (offset / 2);
  __nv_fp8_e4m3* group_output = output + offset;
  const float scale = scales[global_group_id];

  // Unpack and dequantize (process 2 values per iteration)
  for (int i = lane_id * 2; i < group_size; i += THREADS_PER_GROUP * 2) {
    int8_t q0, q1;
    unpack_int4(group_input[i / 2], q0, q1);

    // Dequantize to FP8
    float val0 = (float)q0 * scale;
    group_output[i] = __nv_fp8_e4m3(val0);

    if (i + 1 < group_size) {
      float val1 = (float)q1 * scale;
      group_output[i + 1] = __nv_fp8_e4m3(val1);
    }
  }
}

// ============================================================================
// INT4 -> FP16 Dequantization
// ============================================================================

template <typename T>
__global__ void int4_to_fp16_kernel(
    const uint8_t* __restrict__ input_packed,
    T* __restrict__ output,
    const float* __restrict__ scales,
    const int group_size,
    const int num_groups) {

  const int local_group_id = threadIdx.x / THREADS_PER_GROUP;
  const int lane_id = threadIdx.x % THREADS_PER_GROUP;
  const int global_group_id = blockIdx.x * (blockDim.x / THREADS_PER_GROUP) + local_group_id;

  if (global_group_id >= num_groups) return;

  const int64_t offset = global_group_id * group_size;
  const uint8_t* group_input = input_packed + (offset / 2);
  T* group_output = output + offset;
  const float scale = scales[global_group_id];

  // Unpack and dequantize (process 2 values per iteration)
  for (int i = lane_id * 2; i < group_size; i += THREADS_PER_GROUP * 2) {
    int8_t q0, q1;
    unpack_int4(group_input[i / 2], q0, q1);

    // Dequantize to FP16
    float val0 = (float)q0 * scale;
    group_output[i] = __float2half(val0);

    if (i + 1 < group_size) {
      float val1 = (float)q1 * scale;
      group_output[i + 1] = __float2half(val1);
    }
  }
}

// ============================================================================
// FP8 -> FP16 Dequantization
// ============================================================================

template <typename T>
__global__ void fp8_to_fp16_kernel(
    const __nv_fp8_e4m3* __restrict__ input,
    T* __restrict__ output,
    const float* __restrict__ scales,
    const int group_size,
    const int num_groups) {

  const int local_group_id = threadIdx.x / THREADS_PER_GROUP;
  const int lane_id = threadIdx.x % THREADS_PER_GROUP;
  const int global_group_id = blockIdx.x * (blockDim.x / THREADS_PER_GROUP) + local_group_id;

  if (global_group_id >= num_groups) return;

  const int64_t offset = global_group_id * group_size;
  const __nv_fp8_e4m3* group_input = input + offset;
  T* group_output = output + offset;
  const float scale = scales[global_group_id];

  // Dequantize
  for (int i = lane_id; i < group_size; i += THREADS_PER_GROUP) {
    float val = (float)group_input[i] * scale;
    group_output[i] = __float2half(val);
  }
}

}  // namespace conversion

// ============================================================================
// PyTorch Binding Functions
// ============================================================================

void fp16_to_fp8_cuda(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scales,
    int64_t group_size) {

  TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
  TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
  TORCH_CHECK(scales.is_cuda(), "scales must be CUDA tensor");

  const int num_groups = input.numel() / group_size;
  TORCH_CHECK(input.numel() % group_size == 0, "input size must be divisible by group_size");

  const int groups_per_block = 16;
  const int num_blocks = (num_groups + groups_per_block - 1) / groups_per_block;
  const int num_threads = groups_per_block * conversion::THREADS_PER_GROUP;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  conversion::fp16_to_fp8_kernel<half><<<num_blocks, num_threads, 0, stream>>>(
      reinterpret_cast<const half*>(input.data_ptr()),
      reinterpret_cast<__nv_fp8_e4m3*>(output.data_ptr()),
      scales.data_ptr<float>(),
      group_size,
      num_groups,
      1e-10f,
      448.0f  // FP8 E4M3 max
  );
}

void fp16_to_int4_cuda(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scales,
    int64_t group_size) {

  TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
  TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
  TORCH_CHECK(scales.is_cuda(), "scales must be CUDA tensor");

  const int num_groups = input.numel() / group_size;
  TORCH_CHECK(input.numel() % group_size == 0, "input size must be divisible by group_size");
  TORCH_CHECK(group_size % 2 == 0, "group_size must be even for INT4 packing");

  const int groups_per_block = 16;
  const int num_blocks = (num_groups + groups_per_block - 1) / groups_per_block;
  const int num_threads = groups_per_block * conversion::THREADS_PER_GROUP;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  conversion::fp16_to_int4_kernel<half><<<num_blocks, num_threads, 0, stream>>>(
      reinterpret_cast<const half*>(input.data_ptr()),
      output.data_ptr<uint8_t>(),
      scales.data_ptr<float>(),
      group_size,
      num_groups,
      1e-10f
  );
}

void int4_to_fp8_cuda(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scales,
    int64_t group_size) {

  TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
  TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
  TORCH_CHECK(scales.is_cuda(), "scales must be CUDA tensor");

  const int num_groups = output.numel() / group_size;

  const int groups_per_block = 16;
  const int num_blocks = (num_groups + groups_per_block - 1) / groups_per_block;
  const int num_threads = groups_per_block * conversion::THREADS_PER_GROUP;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  conversion::int4_to_fp8_kernel<<<num_blocks, num_threads, 0, stream>>>(
      input.data_ptr<uint8_t>(),
      reinterpret_cast<__nv_fp8_e4m3*>(output.data_ptr()),
      scales.data_ptr<float>(),
      group_size,
      num_groups
  );
}

void int4_to_fp16_cuda(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scales,
    int64_t group_size) {

  TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
  TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
  TORCH_CHECK(scales.is_cuda(), "scales must be CUDA tensor");

  const int num_groups = output.numel() / group_size;

  const int groups_per_block = 16;
  const int num_blocks = (num_groups + groups_per_block - 1) / groups_per_block;
  const int num_threads = groups_per_block * conversion::THREADS_PER_GROUP;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  conversion::int4_to_fp16_kernel<half><<<num_blocks, num_threads, 0, stream>>>(
      input.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(output.data_ptr()),
      scales.data_ptr<float>(),
      group_size,
      num_groups
  );
}

void fp8_to_fp16_cuda(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scales,
    int64_t group_size) {

  TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
  TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
  TORCH_CHECK(scales.is_cuda(), "scales must be CUDA tensor");

  const int num_groups = input.numel() / group_size;

  const int groups_per_block = 16;
  const int num_blocks = (num_groups + groups_per_block - 1) / groups_per_block;
  const int num_threads = groups_per_block * conversion::THREADS_PER_GROUP;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  conversion::fp8_to_fp16_kernel<half><<<num_blocks, num_threads, 0, stream>>>(
      reinterpret_cast<const __nv_fp8_e4m3*>(input.data_ptr()),
      reinterpret_cast<half*>(output.data_ptr()),
      scales.data_ptr<float>(),
      group_size,
      num_groups
  );
}

// ============================================================================
// TORCH_LIBRARY Bindings
// ============================================================================

TORCH_LIBRARY_FRAGMENT(conversion_ops, m) {
  m.def("fp16_to_fp8(Tensor input, Tensor output, Tensor scales, int group_size) -> ()");
  m.def("fp16_to_int4(Tensor input, Tensor output, Tensor scales, int group_size) -> ()");
  m.def("int4_to_fp8(Tensor input, Tensor output, Tensor scales, int group_size) -> ()");
  m.def("int4_to_fp16(Tensor input, Tensor output, Tensor scales, int group_size) -> ()");
  m.def("fp8_to_fp16(Tensor input, Tensor output, Tensor scales, int group_size) -> ()");

  m.impl("fp16_to_fp8", torch::kCUDA, fp16_to_fp8_cuda);
  m.impl("fp16_to_int4", torch::kCUDA, fp16_to_int4_cuda);
  m.impl("int4_to_fp8", torch::kCUDA, int4_to_fp8_cuda);
  m.impl("int4_to_fp16", torch::kCUDA, int4_to_fp16_cuda);
  m.impl("fp8_to_fp16", torch::kCUDA, fp8_to_fp16_cuda);
}
