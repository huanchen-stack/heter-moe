# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest

from tensorrt_llm._utils import torch_to_numpy

import _utils

# isort: off
import torch
# isort: on

from parameterized import parameterized
from utils.util import create_session, run_session, unittest_name_func

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm.functional import constant, matmul, silu
from tensorrt_llm.quantization.functional import weight_only_quant_matmul


class TestWeightOnlyQuantMatmul(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def _unconvert_weights(self, weights, scales, dtype, wTypeId):
        assert wTypeId == 1 or wTypeId == 2, f"wTypeId={wTypeId} is not supported"
        torch_dtype = _utils.woq_torch_dtype(dtype)
        # Init operands for multiplication in int32
        mat1 = torch.eye(weights.shape[0], dtype=torch.float32,
                         device="cuda").to(torch_dtype)

        return self._run_matmul(mat1, weights, scales, dtype, wTypeId, True)

    def _run_matmul(self, mat1_l, processed_torch_weights, torch_weight_scales,
                dtype, wTypeId, use_plugin):

        builder = tensorrt_llm.Builder()
        network = builder.create_network()

        if use_plugin:
            network.plugin_config.weight_only_quant_matmul_plugin = dtype

        with tensorrt_llm.net_guard(network):

            # Create separate TRT inputs with unique names
            x_repeat = []
            for i, mat1 in enumerate(mat1_l):
                x_repeat.append(
                    Tensor(
                        name=f'x_{i}',
                        shape=mat1.shape,
                        dtype=tensorrt_llm._utils.str_dtype_to_trt(dtype)
                    )
                )

            # Build weight constants
            weights = constant(torch_to_numpy(processed_torch_weights))
            scale = constant(torch_to_numpy(torch_weight_scales))

            output = None
            for i, x in enumerate(x_repeat):
                if wTypeId == 0:
                    out_i = matmul(x, weights)
                else:
                    out_i = weight_only_quant_matmul(
                        x, weights, scale, wTypeId, dtype=dtype
                    )

                if output is None:
                    output = out_i
                else:
                    output = output + out_i

            output.mark_output('output', dtype)

        # Build engine
        session = create_session(
            builder, network,
            precision=dtype,
            int8=True,
            memory_pool_limit=133554432
        )

        # IMPORTANT: Build correct runtime inputs
        inputs = {}
        for i, mat1 in enumerate(mat1_l):
            inputs[f'x_{i}'] = mat1  # Each TRT input name must match

        # Warmup
        if len(mat1_l) > 3:  # for multiple inputs, do fewer warmup runs
            [ run_session(session, inputs) for _ in range(5) ]
        else:
            [ run_session(session, inputs) for _ in range(10) ]

        # Timing run
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()

        start.record()
        outputs = run_session(session, inputs)
        end.record()
        torch.cuda.synchronize()

        print(f"x shape {mat1_l[0].shape}, weights size {processed_torch_weights.nbytes}, "
            f"wTypeId {wTypeId}, use_plugin {use_plugin}: "
            f"{start.elapsed_time(end):.3f} ms")

        return outputs['output']

    def _woq_matmul(self, m, n, k, dtype, wTypeId, use_plugin=True, repeat=1):
        # Init operands for multiplication in int32
        # mat1 = _utils.woq_gen_weights(m, k, dtype)
        mat1_l = [ _utils.woq_gen_weights(m, k, dtype) for _ in range(repeat) ]
        mat1 = mat1_l[0]
        weight = _utils.woq_gen_weights(k, n, dtype)

        if wTypeId == 0:
            ref_torch_weights = weight
            processed_torch_weights = weight
            torch_weight_scales = torch.ones(n, dtype=_utils.woq_torch_dtype(dtype),
                                             device='cuda')
        else:
            ref_torch_weights, processed_torch_weights, torch_weight_scales = _utils.woq_conversion(
                weight, wTypeId)
        
        if wTypeId == 2 and use_plugin:
            ref_torch_weights = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8(
                ref_torch_weights.cpu())
        if not use_plugin:
            processed_torch_weights = ref_torch_weights

        output = self._run_matmul(mat1_l, processed_torch_weights,
                                  torch_weight_scales, dtype, wTypeId,
                                  use_plugin)

        ref = _utils.woq_gt_matmul(m, mat1, ref_torch_weights.cuda(),
                                   torch_weight_scales.cuda(), dtype)

        # _utils.woq_assert_near_eq(ref, output, wTypeId)
        '''
        ref = ref.cpu().flatten()
        diff = abs(ref - output)

        max_diff = diff.max()
        ref_value_of_max_diff = ref[diff == max_diff]
        out_value_of_max_diff = output[diff == max_diff]
        print("###############\nmax diff is {} form {} vs {}\n###############\n\n".format(max_diff, out_value_of_max_diff, ref_value_of_max_diff))
        '''

    def _run_expert_cudagraph(self, mat1_l, 
                processed_torch_weights_13, torch_weight_scales_13,
                processed_torch_weights_2, torch_weight_scales_2,
                dtype, wTypeId, use_plugin):

        builder = tensorrt_llm.Builder()
        network = builder.create_network()

        if use_plugin:
            network.plugin_config.weight_only_quant_matmul_plugin = dtype

        with tensorrt_llm.net_guard(network):

            # Create separate TRT inputs with unique names
            x_repeat = []
            for i, mat1 in enumerate(mat1_l):
                x_repeat.append(
                    Tensor(
                        name=f'x_{i}',
                        shape=mat1.shape,
                        dtype=tensorrt_llm._utils.str_dtype_to_trt(dtype)
                    )
                )

            # Build weight constants
            weights_13 = constant(torch_to_numpy(processed_torch_weights_13))
            scale_13 = constant(torch_to_numpy(torch_weight_scales_13))
            weights_2 = constant(torch_to_numpy(processed_torch_weights_2))
            scale_2 = constant(torch_to_numpy(torch_weight_scales_2))

            output = None
            for i, x in enumerate(x_repeat):
                if wTypeId == 0:
                    gate = matmul(x, weights_13, transa=False, transb=False)
                    up = matmul(x, weights_13, transa=False, transb=False)
                    act = silu(up) * gate
                    out_i = matmul(act, weights_2, transa=False, transb=False)

                else:
                    gate = weight_only_quant_matmul(
                        x, weights_13, scale_13, wTypeId, dtype=dtype
                    )
                    up = weight_only_quant_matmul(
                        x, weights_13, scale_13, wTypeId, dtype=dtype
                    )
                    act = silu(up) * gate
                    out_i = weight_only_quant_matmul(
                        act, weights_2, scale_2, wTypeId, dtype=dtype
                    )

                if output is None:
                    output = out_i
                else:
                    output = output + out_i

            output.mark_output('output', dtype)

        # Build engine
        session = create_session(
            builder, network,
            precision=dtype,
            int8=True,
        )

        # Create static buffers for CUDA graph
        static_inputs = {}
        for i, mat1 in enumerate(mat1_l):
            static_inputs[f'x_{i}'] = mat1.clone().detach().contiguous().cuda()

        # Dry run to get output shape/structure (outside graph capture)
        tmp_out = run_session(session, static_inputs)
        static_outputs = {
            name: out.clone().detach().contiguous().cuda()
            for name, out in tmp_out.items()
        }

        # Create CUDA graph on the capture stream
        g = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()

        # Warmup run on capture stream (prepares TRT engine)
        with torch.cuda.stream(capture_stream):
            _ = run_session(session, static_inputs)
        
        capture_stream.synchronize()

        # Now capture the graph on the same stream
        with torch.cuda.stream(capture_stream):
            g.capture_begin()
            # Use a version of run_session that doesn't synchronize
            # OR pass the outputs dict to avoid internal allocation
            session.run(static_inputs, static_outputs, stream=capture_stream.cuda_stream)
            g.capture_end()
        
        capture_stream.synchronize()

        # Warmup replays
        warmup_iters = 5 if len(mat1_l) > 3 else 10
        for _ in range(warmup_iters):
            g.replay()
        
        torch.cuda.synchronize()

        # Timing runs
        num_timing_iters = 10
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(num_timing_iters):
            g.replay()
        end.record()
        
        torch.cuda.synchronize()

        avg_time_ms = start.elapsed_time(end) / num_timing_iters / len(mat1_l)

        print(f"x shape {mat1_l[0].shape}, "
            f"weights size {(processed_torch_weights_13.nbytes*2+processed_torch_weights_2.nbytes)/(2**30) :.4f} GB, "
            f"weight type {'fp16' if wTypeId == 0 else 'int8' if wTypeId == 1 else 'int4'}, "
            f"tiling dequantization {use_plugin}: "
            f"{avg_time_ms:.3f} ms")

        return static_outputs['output']

    def _run_expert(self, mat1_l, 
                processed_torch_weights_13, torch_weight_scales_13,
                processed_torch_weights_2, torch_weight_scales_2,
                dtype, wTypeId, use_plugin):

        builder = tensorrt_llm.Builder()
        network = builder.create_network()

        if use_plugin:
            network.plugin_config.weight_only_quant_matmul_plugin = dtype

        with tensorrt_llm.net_guard(network):

            # Create separate TRT inputs with unique names
            x_repeat = []
            for i, mat1 in enumerate(mat1_l):
                x_repeat.append(
                    Tensor(
                        name=f'x_{i}',
                        shape=mat1.shape,
                        dtype=tensorrt_llm._utils.str_dtype_to_trt(dtype)
                    )
                )

            # Build weight constants
            weights_13 = constant(torch_to_numpy(processed_torch_weights_13))
            scale_13 = constant(torch_to_numpy(torch_weight_scales_13))
            weights_2 = constant(torch_to_numpy(processed_torch_weights_2))
            scale_2 = constant(torch_to_numpy(torch_weight_scales_2))

            output = None
            for i, x in enumerate(x_repeat):
                if wTypeId == 0:
                    gate = matmul(x, weights_13, transa=False, transb=False)
                    up = matmul(x, weights_13, transa=False, transb=False)
                    act = silu(up) * gate
                    out_i = matmul(act, weights_2, transa=False, transb=False)

                else:
                    gate = weight_only_quant_matmul(
                        x, weights_13, scale_13, wTypeId, dtype=dtype
                    )
                    up = weight_only_quant_matmul(
                        x, weights_13, scale_13, wTypeId, dtype=dtype
                    )
                    act = silu(up) * gate
                    out_i = weight_only_quant_matmul(
                        act, weights_2, scale_2, wTypeId, dtype=dtype
                    )

                if output is None:
                    output = out_i
                else:
                    output = output + out_i

            output.mark_output('output', dtype)

        # Build engine
        session = create_session(
            builder, network,
            precision=dtype,
            int8=True,
            # memory_pool_limit=133554432
        )

        # IMPORTANT: Build correct runtime inputs
        inputs = {}
        for i, mat1 in enumerate(mat1_l):
            inputs[f'x_{i}'] = mat1  # Each TRT input name must match

        # Warmup
        if len(mat1_l) > 3:  # for multiple inputs, do fewer warmup runs
            [ run_session(session, inputs) for _ in range(5) ]
        else:
            [ run_session(session, inputs) for _ in range(10) ]

        # Timing run
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()

        start.record()
        outputs = run_session(session, inputs)
        end.record()
        torch.cuda.synchronize()

        print(f"x shape {mat1_l[0].shape}, "
            f"weights size {(processed_torch_weights_13.nbytes*2+processed_torch_weights_2.nbytes)/(2**30) :.4f} GB, "
            f"weight type {'fp16' if wTypeId == 0 else 'int8' if wTypeId == 1 else 'int4'}, "
            f"tiling dequantization {use_plugin}: "
            f"{start.elapsed_time(end)/len(mat1_l):.3f} ms")

        return outputs['output']

    def _woq_expert(self, batch, intermediate, hidden, dtype, wTypeId, use_plugin=True, repeat=1):
        mat1_l = [ _utils.woq_gen_weights(batch, hidden, dtype) for _ in range(repeat) ]
        mat1 = mat1_l[0]
        weight_13 = _utils.woq_gen_weights(hidden, intermediate, dtype)
        weight_2 = _utils.woq_gen_weights(intermediate, hidden, dtype)

        if wTypeId == 0:
            processed_torch_weights_13 = weight_13
            torch_weight_scales_13 = torch.ones(intermediate, dtype=_utils.woq_torch_dtype(dtype),
                                             device='cuda')
            processed_torch_weights_2 = weight_2
            torch_weight_scales_2 = torch.ones(hidden, dtype=_utils.woq_torch_dtype(dtype),
                                             device='cuda')
        else:
            ref_torch_weights_13, processed_torch_weights_13, torch_weight_scales_13 = _utils.woq_conversion(
                weight_13, wTypeId)
            ref_torch_weights_2, processed_torch_weights_2, torch_weight_scales_2 = _utils.woq_conversion(
                weight_2, wTypeId)
        
            if not use_plugin:
                processed_torch_weights_13 = ref_torch_weights_13
                processed_torch_weights_2 = ref_torch_weights_2

        output = self._run_expert_cudagraph(mat1_l, 
                                  processed_torch_weights_13, torch_weight_scales_13, 
                                  processed_torch_weights_2, torch_weight_scales_2,
                                  dtype, wTypeId,
                                  use_plugin)

    # hidden=5120  # qwen2.5-32B
    # intermediate=27648
    # hidden=8192  # llama3-70B
    # intermediate=28672
    # hidden=4096  # mixtral-8x7B
    # intermediate=14336
    hidden=6144  # mixtral-8x22B
    intermediate=22528

    repeat=1
    @parameterized.expand(
        [
            (1, intermediate, hidden, 0, False, repeat),
            (1, intermediate, hidden, 1, False, repeat),
            (1, intermediate, hidden, 1, True, repeat),
            (1, intermediate, hidden, 2, True, repeat),
            (8, intermediate, hidden, 0, False, repeat),
            (8, intermediate, hidden, 1, False, repeat),
            (8, intermediate, hidden, 1, True, repeat),
            (8, intermediate, hidden, 2, True, repeat),
            (16, intermediate, hidden, 0, False, repeat),
            (16, intermediate, hidden, 1, False, repeat),
            (16, intermediate, hidden, 1, True, repeat),
            (16, intermediate, hidden, 2, True, repeat),
            (32, intermediate, hidden, 0, False, repeat),
            (32, intermediate, hidden, 1, False, repeat),
            (32, intermediate, hidden, 1, True, repeat),
            (32, intermediate, hidden, 2, True, repeat),
            (64, intermediate, hidden, 0, False, repeat),
            (64, intermediate, hidden, 1, False, repeat),
            (64, intermediate, hidden, 1, True, repeat),
            (64, intermediate, hidden, 2, True, repeat),
            (128, intermediate, hidden, 0, False, repeat),
            (128, intermediate, hidden, 1, False, repeat),
            (128, intermediate, hidden, 1, True, repeat),
            (128, intermediate, hidden, 2, True, repeat),
            (256, intermediate, hidden, 0, False, repeat),
            (256, intermediate, hidden, 1, False, repeat),
            (256, intermediate, hidden, 1, True, repeat),
            (256, intermediate, hidden, 2, True, repeat),
            (512, intermediate, hidden, 0, False, repeat),
            (512, intermediate, hidden, 1, False, repeat),
            (512, intermediate, hidden, 1, True, repeat),
            (512, intermediate, hidden, 2, True, repeat),
            (1024, intermediate, hidden, 0, False, repeat),
            (1024, intermediate, hidden, 1, False, repeat),
            (1024, intermediate, hidden, 1, True, repeat),
            (1024, intermediate, hidden, 2, True, repeat),
        ],
        name_func=unittest_name_func)
    def test_expert_fp16_act(self, m, n, k, wTypeId, use_plugin, repeat):
        self._woq_expert(m, n, k, 'float16', wTypeId, use_plugin, repeat)
    
    # @parameterized.expand(
    #     [
    #         (1, intermediate, hidden, 0, False, repeat),
    #         (1, intermediate, hidden, 1, False, repeat),
    #         (1, intermediate, hidden, 1, True, repeat),
    #         (1, intermediate, hidden, 2, True, repeat),
    #     ],
    #     name_func=unittest_name_func)
    # def test_matmul_fp16_act(self, m, n, k, wTypeId, use_plugin, repeat):
    #     self._woq_matmul(m, n, k, 'float16', wTypeId, use_plugin, repeat)

    # @parameterized.expand(
    #     [
    #         (1, 1024, 4096, 1, True),
    #         (1, 1024, 4096, 1, False),
    #         (12, 1024, 512, 1, True),
    #         (64, 6144, 12288, 1, True),  # BF16 * INT8
    #         (1, 1024, 4096, 2, True),
    #         (32, 1024, 256, 2, True),
    #         (256, 6144, 12288, 2, True),  # BF16 * INT4
    #     ],
    #     name_func=unittest_name_func)
    # def test_matmul_bf16_act(self, m, n, k, wTypeId, use_plugin):
    #     self._woq_matmul(m, n, k, 'bfloat16', wTypeId, use_plugin)

    def _conversion_helper(self, n, k, dtype, wTypeId):
        weight_ref = _utils.woq_gen_weights(n, k, dtype)
        ref_int, perm_int, scale = _utils.woq_conversion(weight_ref, wTypeId)
        weight_act = self._unconvert_weights(perm_int, scale, dtype, wTypeId)

        _utils.woq_assert_near_eq(weight_ref, weight_act, wTypeId)

    # @parameterized.expand([(1024, 4096, 1), (4096, 512, 1), (1024, 4096, 2),
    #                        (4096, 512, 2)],
    #                       name_func=unittest_name_func)
    # def test_fp16_conversion(self, n, k, wTypeId):
    #     self._conversion_helper(n, k, 'float16', wTypeId)

    # @parameterized.expand([(1024, 4096, 1), (4096, 512, 1), (1024, 4096, 2),
    #                        (4096, 512, 2)],
    #                       name_func=unittest_name_func)
    # def test_bf16_conversion(self, n, k, wTypeId):
    #     self._conversion_helper(n, k, 'bfloat16', wTypeId)


if __name__ == '__main__':
    unittest.main()
