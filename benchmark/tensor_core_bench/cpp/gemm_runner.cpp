#include "gemm_runner.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cassert>

extern void pack_fp4_cuda(
    const half* in, uint8_t* out, int n
);

#define CHECK_CUDA(x) TORCH_CHECK(x == cudaSuccess, "CUDA error")
#define CHECK_CUBLAS(x) TORCH_CHECK(x == CUBLAS_STATUS_SUCCESS, "cuBLASLt error")

GemmRunner::GemmRunner(
    torch::Tensor A, torch::Tensor B, 
    std::string precision
){
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), 
        "Tensors must be on CUDA device");
    TORCH_CHECK(A.dtype() == torch::kFloat16 && B.dtype() == torch::kFloat16, 
        "Tensors must be of type float16");
    
    M_ = A.size(0);
    K_ = A.size(1);
    N_ = B.size(1);

    TORCH_CHECK(M_ % 16 == 0 && N_ % 16 == 0 && K_ % 16 == 0, 
        "M, N, K must be multiples of 16 (for tensor core?)");

    CHECK_CUDA(cudaStreamCreate(&stream_));

    size_t size_A = M_ * K_ / 2; // fp4: 2 elements per byte
    size_t size_B = K_ * N_ / 2;
    size_t size_C = M_ * N_ * sizeof(half); // C is in fp16

    CHECK_CUDA(cudaMalloc(&A_fp4_, size_A));
    CHECK_CUDA(cudaMalloc(&B_fp4_, size_B));
    CHECK_CUDA(cudaMalloc(&C_fp16_, size_C));
    CHECK_CUDA(cudaMemset(C_fp16_, 0, size_C));

    // Pack A and B to fp4; this only runs once during initialization
    pack_fp4_cuda(
        reinterpret_cast<half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<uint8_t*>(A_fp4_),
        M_ * K_
    );
    pack_fp4_cuda(
        reinterpret_cast<half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<uint8_t*>(B_fp4_),
        K_ * N_
    );

    CHECK_CUBLAS(cublasLtCreate(&handle_));
    CHECK_CUBLAS(cublasLtMatmulDescCreate(
        &desc_, CUBLAS_COMPUTE_16F, CUBLAS_R_16F
    ));

    cublasLtOrder_t row_major = CUBLASLT_ORDER_ROW;
    
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
        &layoutA_, CUDA_R_4F_E2M1, M_, K_, K_
    ));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        layoutA_, CUBLASLT_MATRIX_LAYOUT_ORDER,
        &(row_major), sizeof(row_major)
    ));

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
        &layoutB_, CUDA_R_4F_E2M1, K_, N_, N_
    ));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        layoutB_, CUBLASLT_MATRIX_LAYOUT_ORDER,
        &(row_major), sizeof(row_major)
    ));

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
        &layoutC_, CUDA_R_16F, M_, N_, N_
    ));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        layoutC_, CUBLASLT_MATRIX_LAYOUT_ORDER,
        &(row_major), sizeof(row_major)
    ));
}

void GemmRunner::run_once(){
    half alpha = __float2half(1.0f);
    half beta = __float2half(0.0f);

    CHECK_CUBLAS(cublasLtMatmul(
        handle_,
        desc_,
        &alpha,
        A_fp4_, layoutA_,
        B_fp4_, layoutB_,
        &beta,
        C_fp16_, layoutC_,  // input C
        C_fp16_, layoutC_,  // output destination
        nullptr, nullptr,
        0,
        stream_
    ));
}