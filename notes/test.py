# from tensorrt_llm import LLM, SamplingParams
# # from tensorrt_llm._tensorrt_engine import LLM


# def main():

#     prompts = [
#         "Hello, my name is",
#         "The president of the United States is",
#         "The capital of France is",
#         "The future of AI is",
#     ]
#     sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

#     MODEL = "nvidia/Qwen3-30B-A3B-FP4"
#     MODEL = "Qwen/Qwen3-30B-A3B"
#     MODEL = "Qwen/Qwen3-30B-A3B-FP8"

#     llm = LLM(
#         model=MODEL,
#         tensor_parallel_size=1,
#         backend="tensorrt",
#     )

#     outputs = llm.generate(prompts, sampling_params)

#     # Print the outputs.
#     for output in outputs:
#         prompt = output.prompt
#         generated_text = output.outputs[0].text
#         print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


# # The entry point of the program needs to be protected for spawning processes.
# if __name__ == '__main__':
#     main()


from tensorrt_llm import BuildConfig, SamplingParams
from tensorrt_llm._tensorrt_engine import LLM
def main():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    MODEL = "nvidia/Qwen3-30B-A3B-FP4"
    build_config = BuildConfig()
    build_config.max_batch_size = 256
    build_config.max_num_tokens = 1024
    llm = LLM(
        model=MODEL,
        build_config=build_config,
        tensor_parallel_size=1,
    )
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
if __name__ == '__main__':
    main()