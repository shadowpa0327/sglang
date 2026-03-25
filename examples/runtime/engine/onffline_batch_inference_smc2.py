import os

os.environ["SGLANG_ENABLE_SPEC_V2"] = "True"

import sglang as sgl


def main():
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create a sampling params object.
    sampling_params = {"temperature": 0, "max_new_tokens": 50}

    # Create an LLM with SMC speculative decoding.
    # Using the same model for both draft and target for testing.
    llm = sgl.Engine(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        speculative_algorithm="SMC",
        speculative_draft_model_path="Qwen/Qwen2.5-0.5B-Instruct",
        speculative_num_steps=3,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=4,
        cuda_graph_max_bs=4,
        mem_fraction_static=0.3,
        max_running_requests=8,
    )

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    main()
