# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""A file containing model definitions."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Literal
from typing import Optional
from typing import Union

from swift.llm import ModelType

MODEL_REGISTER = {
    "nuextract1_5": {
        "model_type": ModelType.phi3_5_mini_instruct,
        "hf_id_or_path": "numind/NuExtract-v1.5",
        "prompt_template_type": "nuextract",
        "gpu_count": 1,
        "run_type": "vllm",
    },
    "phi3_mini_4k_graph": {
        "model_type": ModelType.phi3_4b_4k_instruct,
        "hf_id_or_path": "EmergentMethods/Phi-3-mini-4k-instruct-graph",
        "prompt_template_type": "phi3_graph",
        "gpu_count": 1,
        "run_type": "vllm",
        "hf_auth_required": True,
        "max_input_tokens": 4096,
    },
    "phi3_mini_128k_graph": {
        "model_type": ModelType.phi3_4b_128k_instruct,
        "hf_id_or_path": "EmergentMethods/Phi-3-mini-128k-instruct-graph",
        "prompt_template_type": "phi3_graph",
        "gpu_count": 1,
        "run_type": "vllm",
        "hf_auth_required": True,
        "max_input_tokens": 4096,
    },
    "phi3_medium_128k_graph": {
        "model_type": ModelType.phi3_medium_128k_instruct,
        "hf_id_or_path": "EmergentMethods/Phi-3-medium-128k-instruct-graph",
        "prompt_template_type": "phi3_graph",
        "gpu_count": 4,
        "run_type": "vllm",
        "hf_auth_required": True,
    },
    "scilitllm1_5_7b": {
        "model_type": ModelType.qwen2_5_7b_instruct,
        "hf_id_or_path": "Uni-SMART/SciLitLLM1.5-7B",
        "prompt_template_type": "scilitllm",
        "gpu_count": 1,
        "run_type": "vllm",
        "max_input_tokens": 8192,
    },
    "scilitllm1_5_14b": {
        "model_type": ModelType.qwen2_5_14b_instruct,
        "hf_id_or_path": "Uni-SMART/SciLitLLM1.5-14B",
        "prompt_template_type": "scilitllm",
        "gpu_count": 4,
        "run_type": "vllm",
        "max_input_tokens": 32768,
    },
    "triplex": {
        "model_type": ModelType.phi3_4b_128k_instruct,
        "hf_id_or_path": "SciPhi/Triplex",
        "prompt_template_type": "triplex",
        "gpu_count": 1,
        "run_type": "vllm",
        "max_input_tokens": 4096,
    },
    "qwen2_5_7b": {
        "model_type": ModelType.qwen2_5_7b_instruct,
        "hf_id_or_path": "Qwen/Qwen2.5-7B-Instruct",
        "prompt_template_type": "default",
        "gpu_count": 1,
        "run_type": "vllm",
        "max_input_tokens": 32768,
    },
    "qwen2_5_14b": {
        "model_type": ModelType.qwen2_5_14b_instruct,
        "hf_id_or_path": "Qwen/Qwen2.5-14B-Instruct",
        "prompt_template_type": "default",
        "gpu_count": 4,
        "run_type": "vllm",
        "max_input_tokens": 32768,
    },
    "qwen2_5_32b": {
        "model_type": ModelType.qwen2_5_32b_instruct,
        "hf_id_or_path": "Qwen/Qwen2.5-32B-Instruct",
        "prompt_template_type": "default",
        "gpu_count": 4,
        "run_type": "vllm",
        "max_input_tokens": 32768,
    },
    "deepseek_r1_distill_qwen_7b": {
        "model_type": ModelType.qwen2_5_7b_instruct,
        "hf_id_or_path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "prompt_template_type": "deepseek",
        "gpu_count": 1,
        "run_type": "vllm",
        "max_input_tokens": 32768,
    },
    "deepseek_r1_distill_qwen_14b": {
        "model_type": ModelType.qwen2_5_14b_instruct,
        "hf_id_or_path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "prompt_template_type": "deepseek",
        "gpu_count": 4,
        "run_type": "vllm",
        "max_input_tokens": 32768,
    },
    "deepseek_r1_distill_qwen_32b": {
        "model_type": ModelType.qwen2_5_32b_instruct,
        "hf_id_or_path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "prompt_template_type": "deepseek",
        "gpu_count": 4,
        "run_type": "vllm",
        "max_input_tokens": 32768,
    },
    "gemini_1_5_pro": {"run_type": "api", "prompt_template_type": "default"},
    "meta_llama3_1_8b_instruct_v1_0": {
        "run_type": "api",
        "prompt_template_type": "default",
    },
    "meta_llama3_1_405b_instruct_v1_0": {
        "run_type": "api",
        "prompt_template_type": "default",
    },
    "openai_gpt4_turbo_128k": {
        "run_type": "api",
        "prompt_template_type": "default",
    },
    "deepseek_r1": {"run_type": "api", "prompt_template_type": "deepseek"},
    "zeroshotbioner": {
        "hf_id_or_path": "ProdicusII/ZeroShotBioNER",
        "run_type": "bert",
        "max_input_tokens": 512,
    },
    "gliner_medium": {
        "hf_id_or_path": "gliner-community/gliner_medium-v2.5",
        "run_type": "gliner",
        "max_input_tokens": 384,
    },
    "gliner_large": {
        "hf_id_or_path": "gliner-community/gliner_large-v2.5",
        "run_type": "gliner",
        "max_input_tokens": 768,
    },
    "gliner_multitask": {
        "hf_id_or_path": "knowledgator/gliner-multitask-v1.0",
        "run_type": "gliner",
        "max_input_tokens": 512,
    },
    "gliner_multitask_large": {
        "hf_id_or_path": "knowledgator/gliner-multitask-large-v0.5",
        "run_type": "gliner",
        "max_input_tokens": 512,
    },
    "gliner_large_bio": {
        "hf_id_or_path": "urchade/gliner_large_bio-v0.1",
        "run_type": "gliner",
        "max_input_tokens": 512,
    },
    "gliner_nuner_zero_4k": {
        "hf_id_or_path": "numind/NuNER_Zero-4k",
        "run_type": "gliner",
        "max_input_tokens": 2048,
    },
}

DEFAULT_MAX_INPUT_TOKENS = 128000
DEFAULT_MAX_OUTPUT_TOKENS = 4096

# @dataclass
# class DataInfo:
#     id: str

#     def __init(self, _id):
#         self.id = _id


@dataclass
class ModelInfo:
    id: str
    hf_id_or_path: str
    model_type: ModelType | None
    run_type: Literal["api", "bert", "gliner", "vllm"]
    prompt_template_type: str
    gpu_count: int
    max_input_tokens: int
    max_output_tokens: int
    hf_auth_required: bool
    temperature: float

    def __init__(self, model_id, prompt_template_type=None):
        self.id = model_id

        if model_id not in MODEL_REGISTER:
            print("Model type not implemented:", model_id)
            sys.exit(1)

        model_info = MODEL_REGISTER[model_id]

        self.hf_id_or_path = model_info.get("hf_id_or_path", "")
        self.model_type = model_info.get("model_type")
        self.run_type = model_info.get("run_type")
        if (
            "prompt_template_type" not in model_info
            and prompt_template_type is None
        ):
            print(
                "WARNING: No prompt template specified in model register and none was given at runtime. Using default."
            )
            prompt_template_type = "default"
        self.prompt_template_type = model_info.get(
            "prompt_template_type", prompt_template_type
        )
        self.gpu_count = model_info.get("gpu_count", 0)
        self.max_input_tokens = model_info.get(
            "max_input_tokens", DEFAULT_MAX_INPUT_TOKENS
        )
        self.max_output_tokens = model_info.get(
            "max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS
        )
        self.hf_auth_required = model_info.get("hf_auth_required", False)
        self.temperature = 0
