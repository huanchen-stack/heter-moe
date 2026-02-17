#include <torch/extension.h>
#include "gemm_runner.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<GemmRunner>(m, "GemmRunner")
        .def(py::init<torch::Tensor, torch::Tensor, std::string>())
        .def("run_once", &GemmRunner::run_once);
}