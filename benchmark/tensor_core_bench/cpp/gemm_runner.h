#pragma once

#include <torch/extension.h>
#include <cublasLt.h>

class GemmRunner {
public:
    GemmRunner(
        torch::Tensor A, torch::Tensor B, 
        std::string precision
    );
    void run_once();

private:
    int M_, N_, K_;
    std::string precision_;

    void* A_fp4_, * B_fp4_, * C_fp16_;

    cublasLtHandle_t handle_;
    cublasLtMatmulDesc_t desc_;
    cublasLtMatrixLayout_t layoutA_, layoutB_, layoutC_;

    cudaStream_t stream_;
}