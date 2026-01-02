from fp4_bench.runner import GemmBenchmark

bench = GemmBenchmark(M=8192, N=8192, K=8192, precision="fp4")

bench.warmup(iter=10)
t = bench.run(capture_iter=10, iter=3)

print(f"Average time per GEMM: {t:.6f} ms")