#!/usr/bin/env python3
"""Benchmark LLMs for KGC"""
from __future__ import annotations

import html
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Literal

import fire
import rte_utils.bert.gliner_bioner as gliner_bioner
import rte_utils.bert.zeroshotbioner as zeroshotbioner
import rte_utils.prompts as p
from datasets import load_dataset
from datasets import load_from_disk
from huggingface_hub import login
from rte_utils import ModelInfo
from rte_utils.parsing import parse_response
from rte_utils.parsing.utils import DEFAULT_TUPLE_DELIMITER, DEFAULT_RECORD_DELIMITER, DEFAULT_COMPLETION_DELIMITER
from rte_utils.benchmarking import calculate_metrics
from rte_utils.llm import run_remote_lm, run_local_lm

# from swift.llm import ModelType

# vllm>=0.5.4

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Remember to set environ var USE_HF=1
# or use os.environ["USE_HF"] = "True"
# before importing swift

def parse_examples_to_text(examples, has_re):
    icl_parts = ["","Examples:"]

    examples_parsed = parse_response(examples, "json")

    for i, (text, response)in enumerate(zip(examples['query'], examples_parsed)):
        icl_parts.append(f"Example {i+1}:")
        icl_parts.append(text)
        icl_parts.append("")
        icl_parts.append("Entities:")

        for entity in response['entities']:
            if entity is not None:
                entity_name, entity_type = entity
                icl_parts.append(f"{entity_name} ({entity_type})")
        icl_parts.append("")

        if has_re:
            icl_parts.append("Relationships:")
            for relation in response['relations']:
                if relation is not None:
                    (((head_name, head_type), rel_type, (tail_name, tail_type)),_) = relation
                    icl_parts.append(f"{head_name} ({head_type}) --{rel_type}-- {tail_name} ({tail_type})")
            icl_parts.append("")

    return "\n".join(icl_parts)

def parse_examples_to_text_scilit(examples, has_re):
    icl_parts = ["","Examples:"]

    examples_parsed = parse_response(examples, "json")

    for i, (text, response)in enumerate(zip(examples['query'], examples_parsed)):
        icl_parts.append(f"Example {i+1}:")
        icl_parts.append(text)
        icl_parts.append("")

        example_parts = []

        for entity in response['entities']:
            if entity is not None:
                entity_name, entity_type = entity
                example_parts.append(f"({entity_name}, {entity_type})")

        if has_re:
            icl_parts.append("Relationships:")
            for relation in response['relations']:
                if relation is not None:
                    (((head_name, _), rel_type, (tail_name, _)),_) = relation
                    example_parts.append(f"({head_name}, {rel_type}, {tail_name})")

        icl_parts.append(", ".join(example_parts))
        icl_parts.append("")

    return "\n".join(icl_parts)

def dataset_encode(example, prompt=""):
    q, output = example.get("query", ""), example["response"]
    if output is None:
        return {}
    return {
        "system": "",
        "query": prompt.format(query=q),
    }  # ['input_ids', 'dataset_name', 'instruction', 'query', 'response']

def create_task_prompt_template(
    ner_types: list[str],
    re_types: list[str],
    model_id: str,
    prompt_addons: dict[Any, Any],
) -> str:
    """Create (default) task prompt template

    Args:
        ner_types (list[str]): list of entity types
        re_types (list[str]): list of relationship types
        model_id (str): model identifier

    Raises:
        ValueError: Either NER only or NER & RE, else error

    Returns:
        str: prompt
    """

    extract_prompt, end_prompt = "", ""

    if model_id.startswith("scilit"):
        entity_desc = "(entity1_name, entity1_type), (entity2_name, entity2_type), "
        relation_desc = "(entity1_name, RELATION, entity2_name), (entity3_name, RELATION, entity4_name), ..."
    else:
        entity_desc = p.DEFAULT_PROMPT_END_ENTITY_FORMAT
        relation_desc = p.DEFAULT_PROMPT_END_RELATION_FORMAT

    if ner_types and re_types:
        extract_prompt = "Please extract a list of entities, and subsequently a list of relations between these entities."
    elif ner_types:
        extract_prompt = "Please extract a list of entities."
        relation_desc = ""
    elif re_types:
        raise ValueError(
            "create_task_prompt_template was supplied with an empty ner_types, but a non-empty re_types"
        )
    else:
        raise ValueError(
            "create_task_prompt_template was supplied with both an empty ner_types and re_types"
        )

    end_prompt = p.DEFAULT_PROMPT_END_TEMPLATE.format(
        icl=prompt_addons.get("icl",""),
        entity_desc=entity_desc,
        relation_desc=relation_desc,
    )

    prompt_parts = [
        p.DEFAULT_PROMPT_BASE_TEMPLATE.format(what_to_extract=extract_prompt)
    ]

    if ner_types:
        prompt_parts.append(
            p.DEFAULT_PROMPT_ENTITY_TEMPLATE.format(
                entity_types=", ".join(ner_types)
            )
        )

    if re_types:
        prompt_parts.append(
            p.DEFAULT_PROMPT_RELATION_TEMPLATE.format(
                relation_types=", ".join(re_types)
            )
        )

    prompt_parts.append(end_prompt)

    prompt = "\n".join(prompt_parts)
    return prompt


def create_task_prompt_template_from_identifier(
    ner_types: list[str],
    re_types: list[str],
    dataset_name: str,
    model: ModelInfo,
    examples: list[dict[str, list]] | None = None,
    prompting_techniques: list[str] | None = None
) -> str | None:
    """Create task prompt template from (model) identifier

    Args:
        ner_types (list[str]): list of entity types
        re_types (list[str]): list of relationship types
        model_id (str): model identifier

    Returns:
        Optional[str]: prompt
    """

    if model.run_type in ("bert", "gliner"):
        return "{query}"

    prompt_addons = {}

    if prompting_techniques:
        for prompting_technique in prompting_techniques:
            if prompting_technique == "icl" and examples is not None:
                if model.prompt_template_type == "scilitllm":
                    prompt_addons["icl"] = parse_examples_to_text_scilit(examples, bool(re_types))
                else:
                    prompt_addons["icl"] = parse_examples_to_text(examples, bool(re_types))
            elif prompting_technique == "cot":
                pass
            else:
                print(
                    f"Prompting technique {prompting_technique} not recognised. Skipping"
                )

    # Generate template from template type
    if model.prompt_template_type in ("default", "deepseek", "scilitllm"):
        try:
            prompt_template = create_task_prompt_template(
                ner_types=ner_types,
                re_types=re_types,
                model_id=model.id,
                prompt_addons=prompt_addons,
            )
        except ValueError as e:
            print(
                f"It was not possible to generate prompt template for {model.id}"
            )
            print(f"with NER types {ner_types} and RE types {re_types}")
            print("Error:", e)
            return None
    elif model.prompt_template_type == "triplex":
        ner_types = [x.upper() for x in ner_types]
        prompt_template = p.TRIPLEX_PROMPT.format(
            entity_types=str(json.dumps({"entity_types": ner_types}))
            .replace("{", "{{")
            .replace("}", "}}"),
            predicates=str(json.dumps({"predicates": re_types}))
            .replace("{", "{{")
            .replace("}", "}}"),
        )
    elif model.prompt_template_type == "phi3_graph":
        if re_types:
            re_description = " and relationship must be one of the following types: " + ", ".join(re_types)
        else:
            re_description = ""

        prompt_template = p.PHI_3_INSTRUCT_GRAPH_PROMPT.format(
            entity_types=", ".join(ner_types),
            re_description=re_description
        )

    elif model.prompt_template_type == "graphrag":
        if "icl" in prompt_addons:
            graphrag_prompt_template = p.BIOGRAPH_EXTRACTION_PROMPT
        else:
            graphrag_prompt_template = p.BIOGRAPH_EXTRACTION_PROMPT_NO_ICL

        prompt_template = graphrag_prompt_template.format(
            tuple_delimiter=DEFAULT_TUPLE_DELIMITER,
            record_delimiter=DEFAULT_RECORD_DELIMITER,
            completion_delimiter=DEFAULT_COMPLETION_DELIMITER,
            entity_types=",".join(ner_types),
            relationship_types=",".join(re_types),
        )

    else:
        print(f"Prompt template type {model.prompt_template_type} not implemented")
        return None

    return prompt_template

def fetch_lm_response(inputs,
                      model: ModelInfo,
                      ner_types,
                      re_types,
                      batch_size):
    if model.run_type == "zeroshotbioner":
        response = zeroshotbioner.batch_predict_entities(
            texts=inputs,
            ner_types=ner_types,
            max_length=model.max_input_tokens,
            model_name=model.hf_id_or_path
        )
    elif model.run_type == "gliner":
        response = gliner_bioner.batch_predict_entities(
            texts=inputs,
            ner_types=ner_types,
            re_types=re_types,
            max_length=model.max_input_tokens,
            model_name=model.hf_id_or_path,
        )
    elif model.run_type == "vllm":
        response = run_local_lm(
            model_type=model.model_type,
            prompts=inputs,
            model_id_or_path=model.hf_id_or_path,
            max_input_tokens=model.max_input_tokens,
            max_output_tokens=model.max_output_tokens,
            batch_size=batch_size,
            temperature=model.temperature,
            gpu_count=model.gpu_count,
        )
    else:
        response = run_remote_lm(
            model_name=model.id,
            prompts=inputs,
            max_input_tokens=model.max_input_tokens,
            max_output_tokens=model.max_output_tokens,
            batch_size=batch_size,
            temperature=model.temperature,
        )
    return response

def fetch_icl_examples(dataset, icl_count):
    if len(dataset['train']) >= icl_count:
        examples = (dataset['train']
                    .filter(lambda x: (x['response'] != '{"entities": [], "relations": []}'))
                    .select(range(icl_count))
                    )
        #print(examples)
        #print(examples['response'])
        return examples
    else:
        raise ValueError

def benchmark_llm(
    model_id: str,
    dataset_index: int,
    max_input_tokens: int = 128000,
    max_output_tokens: int = 4096,
    batch_size: int | None = 16,
    temperature: float | None = 0,
    rerun_llm: bool = False,
    prompting_techniques: list[str] | None = None,
    prompt_type: str | None = None,
    dataset_path: Path = Path.cwd() / Path("gs_merged_all"),
    type_definitions_path: Path = Path.cwd() / Path("dataset_type_definitions.json"),
    limit_samples: bool = True,
    limit_to_number: int = 512,
    icl_count: int = 3
) -> tuple[tuple]:
    CORPORA_DATA = os.environ.get('CORPORA_DATA', os.getcwd())
    dataset_path = Path(CORPORA_DATA) / Path("gs_merged_all")
    type_definitions_path = Path(CORPORA_DATA) / Path("dataset_type_definitions.json")
    
    # LOAD MODEL INFO FROM MODEL REGISTER
    model = ModelInfo(model_id, prompt_template_type=prompt_type)

    if model.hf_auth_required:
        login(token=os.environ["HF_AUTH"])

    max_input_tokens = min(model.max_input_tokens, max_input_tokens)

    # FIX ARRAY INDEXING (SLURM)
    dataset_index = dataset_index - 1

    # FETCH DATA
    full_dataset = load_from_disk(dataset_path)
    dataset_type_definitions = load_dataset(
        "json", data_files=str(type_definitions_path)
    )
    dataset_type_definition = dataset_type_definitions["train"][dataset_index]

    # FILTER DATA
    config_name = dataset_type_definition["dataset_name"]
    selected_dataset = full_dataset.filter(
        lambda example: example["dataset_name"] == config_name
    )

    # FETCH ICL EXAMPLES
    if prompting_techniques is not None and "icl" in prompting_techniques:
        try:
            examples = fetch_icl_examples(selected_dataset, icl_count)
        except ValueError:
            print(f"Attempted to perform a {icl_count}-shot inference for {config_name}, but not enough samples exist in training set. Exiting...")
            sys.exit(1)
    else:
        examples = None

    # FETCH ENTITY AND RELATION TYPES
    ner_types = dataset_type_definition["ner_types"]
    re_types = dataset_type_definition["re_types"]

    # CREATE PROMPT FOR DATASET
    prompt = create_task_prompt_template_from_identifier(
        ner_types=ner_types,
        re_types=re_types,
        dataset_name=dataset_type_definition["dataset_name"],
        model=model,
        examples=examples,
        prompting_techniques=prompting_techniques
    )
    if prompt is None:
        print("No prompt generated. Implement prompt type before proceeding. Exiting...")
        sys.exit(1)

    # SELECT SPLIT
    used_set = ""
    if selected_dataset["test"]:
        dataset = selected_dataset["test"]
        used_set = "test"
    elif selected_dataset["validation"]:
        dataset = selected_dataset["validation"]
        used_set = "validation"
    else:
        print("No validation or test set")
        sys.exit(1)

    # APPLY PROMPT TO DATASET
    dataset = dataset.map(dataset_encode, fn_kwargs={"prompt": prompt})

    # LIMIT SAMPLES USED
    if limit_samples:
        n_samples = min(len(dataset), limit_to_number)
        dataset = dataset.select(range(n_samples))
    else:
        n_samples = len(dataset)

    print(f"Using {n_samples} samples from {config_name} ({used_set})")

    # OUTPUT DIRECTORY AND FILE NAMES
    out_dir = (
        f"Benchmarking/outputs/{model_id}/{config_name}_{used_set}"
    )
    if prompt_type:
        out_dir = f"{out_dir}/{prompt_type}"
    if prompting_techniques:
        prompting_techniques = set(prompting_techniques)
        if "icl" in prompting_techniques:
            prompting_techniques.remove("icl")
            prompting_techniques.add(f"icl{icl_count}")

        prompting_techniques = sorted(list(prompting_techniques))

        out_dir = f"{out_dir}/{'_'.join(prompting_techniques)}"

    llm_response_file = f"{out_dir}/llm_response.pkl"
    gs_response_file = f"{out_dir}/true_response.pkl"
    pred_output_file = f"{out_dir}/pred_output.pkl"
    true_output_file = f"{out_dir}/true_output.pkl"

    # PULL INPUT AND TRUE OUTPUT
    inputs = dataset["query"]
    gs_response = dataset["response"]

    # GENERATE LLM OUTPUT
    if os.path.isdir(out_dir) and not rerun_llm:
        print("Out directory exists and rerun_llm was not set to True.")
        print("Using results in folder.")

        with open(llm_response_file, "rb") as f_in:
            response = pickle.load(f_in)
    else:
        os.makedirs(out_dir, exist_ok=True)

        response = fetch_lm_response(inputs, model,
                                     ner_types, re_types, batch_size)

        with open(llm_response_file, "wb") as f_out:
            pickle.dump(response, f_out)

        with open(gs_response_file, "wb") as f_out:
            pickle.dump(gs_response, f_out)

    # PARSING
    print("Parsing LLM response")
    if model.run_type in ("zeroshotbioner", "gliner"):
        pred_output = response
    else:
        pred_output = parse_response(response, model.prompt_template_type)

    print("Parsing true response")
    true_output = parse_response(gs_response, "json")

    # SAVE PARSED OUTPUT
    with open(pred_output_file, "wb") as f_out:
        pickle.dump(pred_output, f_out)

    with open(true_output_file, "wb") as f_out:
        pickle.dump(true_output, f_out)

    # BENCHMARKING
    benchmarking_metrics = calculate_metrics(
        pred_output,
        true_output,
        ignore_case=True,
        restrict_types=True,
        ignore_directionality=False,
        ner_types=ner_types,
        re_types=re_types,
        match_criteria="partial_strict"
    )

    # SAVE BENCHMARKING METRICS
    with open(f"{out_dir}/benchmarking_metrics.pkl", "wb") as f_out:
        pickle.dump(benchmarking_metrics, f_out)

    return benchmarking_metrics


if __name__ == "__main__":
    fire.Fire(benchmark_llm)
