import torch
from fp4_bench._C import GemmRunner

assert torch.cuda.is_available(), "CUDA is not available"

class GemmBenchmark:
    def __init__(self, M, N, K, precision="fp4"):
        self.M, self.N, self.K = M, N, K
        assert precision in ["fp4", "fp8", "fp16"], "Unsupported precision type"

        # init tensors in fp16
        self.A = torch.randn((M, K), device="cuda", dtype=torch.float16)
        self.B = torch.randn((K, N), device="cuda", dtype=torch.float16)

        self.runner = GemmRunner(self.A, self.B, precision)

        self.graph = None

    def warmup(self, iter=10):
        for _ in range(iter):
            self.runner.run_once()
        torch.cuda.synchronize()
    
    def capture(self, iter=10):
        self.graph = torch.cuda.CUDAGraph()

        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph):
            for _ in range(iter):
                self.runner.run_once()
        torch.cuda.synchronize()
    
    def run(self, capture_iter = 10, iter=3):
        if self.graph is None:
            self.capture(iter=capture_iter)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start.record()
        for _ in range(iter):
            self.graph.replay()
        end.record()
        torch.cuda.synchronize()

        elapsed_time_ms = start.elapsed_time(end)
        avg_time_ms = elapsed_time_ms / iter / capture_iter
        return avg_time_ms