# launch the offline engine
import asyncio
import io
import os

from PIL import Image
import requests
import sglang as sgl

from sglang.srt.conversation import chat_templates
from sglang.test.test_utils import is_in_ci
from sglang.utils import async_stream_and_merge, stream_and_merge

if is_in_ci():
    import patch
else:
    import nest_asyncio

    nest_asyncio.apply()

if __name__ == '__main__':
    # Set random seed for reproducibility
    import random
    import numpy as np
    import torch
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    llm = sgl.Engine(
        model_path="meta-llama/Meta-Llama-3.1-8B-Instruct",  
        watchdog_timeout=1000000,
        #speculative_algorithm="SSPEC",
        speculative_algorithm="EAGLE3",
        speculative_draft_model_path="jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B",  
        speculative_num_steps=5,  
        speculative_eagle_topk=1,  
        speculative_num_draft_tokens=5,  
        mem_fraction_static=0.6,  # Note: use mem_fraction_static instead of mem_fraction  
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
        attention_backend="flashinfer",
        #cuda_graph_max_bs=2,  
        dtype="float16",
        disable_radix_cache=True,
        # enable_double_sparsity=True,  
        # ds_channel_config_path="./test/srt/double-sparsity-config-Llama-3.1-8B-Instruct.json",  
        # ds_heavy_channel_num=32,  
        # ds_heavy_channel_type="k",  
        # ds_heavy_token_num=512,  
        # ds_sparse_decode_threshold=0
    )
    prompts = [
        "The president of the United States is?",
        "The president of the Taiwan is?",
    ]

    sampling_params = {"temperature": 0.8, "top_p": 0.95}

    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")