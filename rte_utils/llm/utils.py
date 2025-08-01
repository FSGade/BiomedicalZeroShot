from __future__ import annotations

import os

import torch
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain.schema.language_model import LanguageModelInput
from langchain_openai import ChatOpenAI
from swift.llm import get_default_template_type
from swift.llm import get_template
from swift.llm import get_vllm_engine
from swift.llm import inference_vllm
from swift.utils import seed_everything


def run_remote_lm(
    model_name: str,
    prompts: list[LanguageModelInput],
    max_input_tokens: int | None = None,
    max_output_tokens: int | None = 2048,
    temperature: int | None = None,
    batch_size: int | None = None,
):
    """_summary_

    Args:
        model_name (str): _description_
        prompts (list[LanguageModelInput]): _description_
        max_input_tokens (int | None, optional): _description_. Defaults to None.
        max_output_tokens (int | None, optional): _description_. Defaults to 2048.
        temperature (int | None, optional): _description_. Defaults to None.
        batch_size (int | None, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

    print(f"[INFO]: Init LLM ({model_name})")
    llm = ChatOpenAI(
        model_name=model_name,
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_API_BASE,
        streaming=False,
        timeout=300,
        max_retries=3,
        temperature=temperature,
        # seed=42,
    )

    # TODO: Implement caching
    output = [
        llm.invoke(prompt, config={"callbacks": [ConsoleCallbackHandler()]})
        for prompt in prompts
    ]

    return output


def run_local_lm(
    model_type: str,
    prompts: list[LanguageModelInput],
    model_id_or_path: str,
    max_input_tokens: int | None = None,
    max_output_tokens: int | None = 2048,
    batch_size: int | None = None,
    temperature: int | None = None,
    gpu_count: int = 1,
):
    """Run local llm

    Args:
        model_type (str): _description_
        prompts (list[LanguageModelInput]): _description_
        model_id_or_path (str): _description_
        max_input_tokens (int | None, optional): _description_. Defaults to None.
        max_output_tokens (int | None, optional): _description_. Defaults to 2048.
        batch_size (int | None, optional): _description_. Defaults to None.
        temperature (int | None, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    print(f"Using tensor_parallel_size {gpu_count}")
    if torch.cuda.is_available():
        print(f"Device count: {torch.cuda.device_count()}")

    vllm_engine = get_vllm_engine(
        model_type,
        torch.bfloat16,
        model_id_or_path=model_id_or_path,
        max_model_len=max_input_tokens,
        max_num_seqs=16,
        tensor_parallel_size=gpu_count,
    )

    tokenizer = vllm_engine.hf_tokenizer

    vllm_engine.generation_config.max_new_tokens = max_output_tokens
    vllm_engine.generation_config.temperature = temperature

    template_type = get_default_template_type(model_type)
    print(f"model template_type: {template_type}")
    template = get_template(template_type, tokenizer)

    generation_info = {}  # type: dict[str, Any]
    request_list = [{"query": query} for query in prompts]

    seed_everything(42)
    response_list = inference_vllm(
        vllm_engine,
        template,
        request_list,
        generation_info=generation_info,
        max_batch_size=batch_size,
        use_tqdm=True,
    )
    # print(f'query: {query}')
    # print(f'response: {response_list[0]["response"]}')
    print(generation_info)

    return response_list
