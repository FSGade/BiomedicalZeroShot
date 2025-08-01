#!/usr/bin/env python3
from __future__ import annotations

import os
import pickle
import re
import sys
import time
import warnings
from itertools import chain
from itertools import islice
from pathlib import Path
from typing import List
from typing import Set

import deepspeed
import numpy as np
import torch
from datasets import load_dataset
from datasets import load_from_disk
from deepspeed.inference.engine import InferenceEngine
from deepspeed.runtime.utils import see_memory_usage
from scipy.special import softmax
from transformers import AutoConfig
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import BertForTokenClassification
from transformers import pipeline
from transformers.models.t5.modeling_t5 import T5Block
from transformers.tokenization_utils_base import BatchEncoding


def _remove_redundant_space(s):
    # '   a  b  \t  c  \n' --> 'a b c'
    #'  kjc,  jns , ((  : ()  )  ( . )( ln  kc  a,,  ' --> 'kjc,jns,((:())(.)(ln kc a,,'
    s = " ".join(s.split())  # 多个空白字符变为单个空格
    s = re.sub(
        r"\s*(,|:|\(|\)|\.|_|;|'|-)\s*", r"\1", s
    )  # 去除特殊符号旁的空白字符
    return s


def _format(s):
    "集大成的格式规范化，集中解决各种格式的疑难杂症"
    s = _remove_redundant_space(s)
    s = s.lower()
    s = s.replace("{", "").replace("}", "")
    s = re.sub(",+", ",", s)
    s = re.sub(r"\.+", ".", s)
    s = re.sub(";+", ";", s)
    s = s.replace("’", "'")
    s = s.replace("location", "located")
    return s


def batch_texts(seq, chunksize):
    # Padding to multiple of chunksize
    seq = seq + min(
        chunksize - (len(seq) % chunksize),
        (len(seq) % chunksize != 0) * chunksize,
    ) * [seq[-1]]
    it = iter(seq)
    while True:
        try:
            yield list(chain([next(it)], islice(it, chunksize - 1)))
        except StopIteration:
            return


def parse_instructuie_entities(pred_output):
    entities, relations = pred_output

    if relations is None:
        relations = len(entities) * [[]]

    parsed = []

    for abstract_entities, abstract_relations in zip(entities, relations):
        parsed.append(
            {
                "entities": abstract_entities,
                "relations": abstract_relations,
            }
        )

    return parsed


def main(
    model_name: str = "ZWK/InstructUIE",
    dataset_path: Path = Path.cwd() / Path("gs_merged_all"),
    type_definitions_path: Path = Path.cwd() / Path("dataset_type_definitions.json"),
    debug: bool = False,
    limit_to_number=512,
):
    CORPORA_DATA = os.environ.get('CORPORA_DATA', os.getcwd())
    dataset_path = Path(CORPORA_DATA) / Path("gs_merged_all")
    type_definitions_path = Path(CORPORA_DATA) / Path("dataset_type_definitions.json")

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_identifier = model_name.replace("/", "_")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)
    model.resize_token_embeddings(len(tokenizer))

    if debug:
        see_memory_usage("After load model", force=True)
    # Initialize the DeepSpeed-Inference engine
    model = deepspeed.init_inference(
        model,
        tensor_parallel={"tp_size": 4},
        dtype=torch.float,
        injection_policy={
            T5Block: (
                "SelfAttention.o",
                "EncDecAttention.o",
                "DenseReluDense.wo",
            )
        },
        zero={
            "stage": 2,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
    )

    # Fetch data
    raw_dataset = load_from_disk(dataset_path)
    dataset_type_definitions = load_dataset(
        "json", data_files=str(type_definitions_path)
    )
    # Fix array indexing (SLURM)
    with open(
        "rte_utils/bert/index.txt"
    ) as f:
        dataset_index = int(f.read().strip())

    dataset_index = dataset_index - 1

    dataset_type_definition = dataset_type_definitions["train"][dataset_index]

    ner_types = dataset_type_definition["ner_types"]
    re_types = dataset_type_definition["re_types"]

    config_name = dataset_type_definition["dataset_name"]

    dataset = raw_dataset.filter(
        lambda example: example["dataset_name"] == config_name
    )

    # Select split
    used_set = ""
    if dataset["test"]:
        dataset = dataset["test"]
        used_set = "test"
    elif dataset["validation"]:
        dataset = dataset["validation"]
        used_set = "validation"
    else:
        print("ERROR: No validation or test set for", config_name)
        sys.exit(1)

    n_samples = min(len(dataset), limit_to_number)
    dataset = dataset.select(range(n_samples))

    print(f"Using {n_samples} samples from {config_name} ({used_set})")

    # Name of output directory
    out_dir = Path.cwd() / f"Benchmarking/outputs/{model_identifier}/{config_name}_{used_set}"

    # Pull x and y
    texts = dataset["query"]

    pred_output = batch_predict_entities(
        texts=texts,
        ner_types=ner_types,
        re_types=re_types,
        tokenizer=tokenizer,
        model=model,
        local_rank=local_rank,
    )

    if local_rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        parsed_output = parse_instructuie_entities(pred_output)
        with open(f"{out_dir}/pred_output.pkl", "wb") as f_out:
            pickle.dump(parsed_output, f_out)

    # ner_instruction = "Please find all the entity words associated with the category in the given text.Output format is \"type1: word1; type2: word2\". \n"
    # ner_instruction = "Please tell me all the entity words in the text that belong to a given category.Output format is \"type1: word1; type2: word2\". \n"

    # texts = 10 * ["""Naloxone reverses the antihypertensive effect of clonidine.In unanesthetized, spontaneously hypertensive rats the decrease in blood pressure and heart rate produced by intravenous clonidine, 5 to 20 micrograms/kg, was inhibited or reversed by nalozone, 0.2 to 2 mg/kg. The hypotensive effect of 100 mg/kg alpha-methyldopa was also partially reversed by naloxone. Naloxone alone did not affect either blood pressure or heart rate. In brain membranes from spontaneously hypertensive rats clonidine, 10(-8) to 10(-5) M, did not influence stereoselective binding of [3H]-naloxone (8 nM), and naloxone, 10(-8) to 10(-4) M, did not influence clonidine-suppressible binding of [3H]-dihydroergocryptine (1 nM). These findings indicate that in spontaneously hypertensive rats the effects of central alpha-adrenoceptor stimulation involve activation of opiate receptors. As naloxone and clonidine do not appear to interact with the same receptor site, the observed functional antagonism suggests the release of an endogenous opiate by clonidine or alpha-methyldopa and the possible role of the opiate in the central control of sympathetic tone."""]


def batch_predict_entities(
    texts: list[str],
    ner_types: list[str],
    re_types: list[str],
    tokenizer: AutoTokenizer,
    model: InferenceEngine,
    local_rank: int,
    batch_size: int = 4,
    max_target_length: int = 512,
):

    ner_instruction = 'Please list all entity words in the text that fit the category.Output format is "type1: word1; type2: word2". \n'

    ner_instructions = [
        f"{ner_instruction}Option: {', '.join(ner_types)} \nText: {text} \nAnswer:"
        for text in texts
    ]

    # if debug:
    #     see_memory_usage("After DS-inference init", force=True)
    #     torch.cuda.synchronize()
    #     start = time.time()

    if local_rank == 0:
        all_ner_outputs = []
        all_re_outputs = []

    for ner_batch in batch_texts(ner_instructions, batch_size):
        input_ids = tokenizer(
            ner_batch,
            return_tensors="pt",
            is_split_into_words=False,
            padding=True,
            truncation=True,
        ).input_ids.to("cuda")

        output = model.generate(input_ids, max_new_tokens=max_target_length)

        if local_rank == 0:
            for i in range(batch_size):
                all_ner_outputs.append(
                    tokenizer.decode(output[i], skip_special_tokens=True)
                )

    if local_rank == 0:
        all_ner_outputs = all_ner_outputs[: len(ner_instructions)]
        entity_dicts = []
        for example in all_ner_outputs:
            example = example.strip().strip(";")
            example_list = example.split("; ")
            example_entity_dict = dict()
            for entity_str in example_list:
                entity = entity_str.split(": ", 1)
                if len(entity) == 2:
                    example_entity_dict[entity[1]] = entity[0]
            entity_dicts.append(example_entity_dict)

        if not re_types:
            return [
                list(entities_dict.items()) for entities_dict in entity_dicts
            ], None

    re_instruction = 'Given a phrase that describes the relationship between two words, extract the words and the lexical relationship betweenthem. The output format should be "relation1: word1, word2; relation2: word3, word4". \n'
    re_instructions = [
        f"{re_instruction}Option: {', '.join(re_types)} \nText: {text} \nAnswer:"
        for text in texts
    ]

    for re_batch in batch_texts(re_instructions, batch_size):
        input_ids = tokenizer(
            re_batch,
            return_tensors="pt",
            is_split_into_words=False,
            padding=True,
            truncation=True,
        ).input_ids.to("cuda")

        output = model.generate(input_ids, max_new_tokens=max_target_length)

        if local_rank == 0:
            for i in range(batch_size):
                all_re_outputs.append(
                    tokenizer.decode(output[i], skip_special_tokens=True)
                )

    if local_rank == 0:
        all_re_outputs = all_re_outputs[: len(re_instructions)]
        relationships = []
        for example, entities_dict in zip(all_re_outputs, entity_dicts):
            example = example.strip().strip(";")
            example_list = example.split("; ")
            example_relationships = []
            for entity_str in example_list:
                entity = entity_str.split(": ", 1)
                if len(entity) == 2:
                    rel, head_and_tail = entity

                    head_and_tail = head_and_tail.split(", ", 1)
                    if len(head_and_tail) == 2:
                        head, tail = head_and_tail
                        example_relationships.append(
                            (
                                (
                                    (head, entities_dict.get(head, "unknown")),
                                    rel,
                                    (tail, entities_dict.get(tail, "unknown")),
                                ),
                                None,
                            )
                        )

            relationships.append(example_relationships)

        entities = [
            list(entities_dict.items()) for entities_dict in entity_dicts
        ]

        return entities, relationships


if __name__ == "__main__":
    main()


# output = pipe(instruction)

# print(output)

# ZeRO 2
# Estimated memory needed for params, optim states and gradients for a:
# HW: Setup with 1 node, 4 GPUs per node.
# SW: Model with 11003M total params.
#   per CPU  |  per GPU |   Options
#   245.95GB |  20.50GB | offload_optimizer=cpu
#   245.95GB |  66.61GB | offload_optimizer=none

# ZeRO 3
# Estimated memory needed for params, optim states and gradients for a:
# HW: Setup with 1 node, 4 GPUs per node.
# SW: Model with 11003M total params, 131M largest layer params.
#   per CPU  |  per GPU |   Options
#   276.69GB |   0.49GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
#   276.69GB |   0.49GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
#   245.95GB |   5.61GB | offload_param=none, offload_optimizer=cpu , zero_init=1
#   245.95GB |   5.61GB | offload_param=none, offload_optimizer=cpu , zero_init=0
#     2.94GB |  46.61GB | offload_param=none, offload_optimizer=none, zero_init=1
#   245.95GB |  46.61GB | offload_param=none, offload_optimizer=none, zero_init=0

# deepspeed --num_gpus 4 instructuie.py

# max_source_length = 512
# label_pad_token_id = -100
# pad_to_multiple_of = 8

# tokenized_source = tokenizer(instruction)["input_ids"]
# label = " None"

# if len(tokenized_source) <= max_source_length:
#     source = instruction
# else:
#     source = tokenizer.decode(tokenized_source[:max_source_length], skip_special_tokens=True)


# model_inputs = tokenizer(instruction, padding="longest", truncation=True,
#         pad_to_multiple_of=pad_to_multiple_of, max_length=max_source_length, return_tensors='pt')

# with tokenizer.as_target_tokenizer():
#     labels = tokenizer(
#         label,
#         max_length=max_target_length,
#         padding=True,
#         return_tensors='pt',
#         truncation=True,
#         pad_to_multiple_of=pad_to_multiple_of
#     )

# label_mask = labels["attention_mask"].bool()
# model_inputs["labels"] = labels["input_ids"].masked_fill(~label_mask, label_pad_token_id)

# decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
# model_inputs["decoder_input_ids"] = decoder_input_ids

# a = model(**model_inputs)
