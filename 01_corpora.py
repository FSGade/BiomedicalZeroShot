#!/usr/bin/env python
# coding: utf-8
# pylint: disable-msg=C0103
import glob
import json
import os
import re
from pprint import pprint
from typing import List

import fire
import numpy as np
import pandas as pd
from datasets import Dataset
from datasets import DatasetDict
from datasets import load_dataset
from datasets import load_from_disk
from rte_utils.corpora import create_bigbio_bibliography
from rte_utils.corpora import fetch_type_names_dict
from rte_utils.corpora import fetch_type_set
from rte_utils.corpora import load_bigbio_datasets, parse_bigbio_to_llm
from rte_utils.corpora import standardise_type_name
from rte_utils.corpora import umls_dicts

CORPORA_DATA = os.environ.get('CORPORA_DATA', os.getcwd())
os.environ["HF_DATASETS_CACHE"] = CORPORA_DATA

@fire.Fire
def main(create_bibliography=False):
    UMLS_DICT, UMLS_CUI_DICT = umls_dicts(CORPORA_DATA)
    
    bb_kb_datasets, bb_kb_helpers = load_bigbio_datasets()
    
    if create_bibliography:
        create_bigbio_bibliography(bb_kb_helpers)
    
    split_names = dict()
    for dataset_name, bb_kb_dataset in bb_kb_datasets.items():
        for dataset_key in bb_kb_dataset.keys():
            if dataset_key not in split_names:
                split_names[dataset_key] = [dataset_name]
            else:
                split_names[dataset_key].append(dataset_name)

    ###########################
    ### ID ENTITY/REL TYPES ###
    ###########################
    
    translation_dicts = dict()

    for helper in bb_kb_helpers:
        print(helper.config.name)

        if helper.config.name not in bb_kb_datasets:
            print("Data not loaded for", helper.config.name)
            continue

        if helper.config.name in translation_dicts:
            print("Already generated translation_dict for", helper.config.name)
            continue

        entity_dict = fetch_type_names_dict(helper.config.name, "entities", bb_kb_datasets)
        relations_dict = fetch_type_names_dict(helper.config.name, "relations", bb_kb_datasets)

        translation_dicts[helper.config.name] = (
            entity_dict,
            relations_dict,
        )

        print("Entities:", entity_dict)
        print("Relations:", relations_dict)
        print()

    ####
    #### Loop through and parse to wanted format
    ####

    llm_structured_input = {"train": dict(), "validation": dict(), "test": dict()}

    for helper in bb_kb_helpers:
        print(helper.config.name)

        if helper.config.name not in bb_kb_datasets:
            print("Data not loaded for", helper.config.name)
            continue
        # TODO: Augment with subsets of entity/relationship types
        for split in ("train", "validation", "test"):
            if not helper.config.name in llm_structured_input[split]:
                temp_out = parse_bigbio_to_llm(helper, split)
                if temp_out is not None:
                    llm_structured_input[split][helper.config.name] = temp_out

    del llm_structured_input["test"][
        "bionlp_st_2019_bb_bigbio_kb"
    ]  # NO ANNOTATIONS - USE VALIDATION

    def create_task_prompt_template_from_helper(helper):
        config_name = helper.config.name

        (
            ENTITY_TRANSLATION_DICT,
            RELATION_TRANSLATION_DICT,
        ) = translation_dicts[config_name]

        allowed_entity_types = sorted(
            list(
                set(
                    [
                        entity_type
                        for entity_type in ENTITY_TRANSLATION_DICT.values()
                        if entity_type != ""
                    ]
                )
            )
        )
        allowed_relation_types = sorted(
            list(
                set(
                    [
                        relation_type
                        for relation_type in RELATION_TRANSLATION_DICT.values()
                        if relation_type != ""
                    ]
                )
            )
        )
        if config_name == "verspoor_2013_bigbio_kb":
            allowed_relation_types = [
                "HAS_MUTATION",
                "VARIANT_ASSOCIATED_WITH",
                "HAS_MUTATION_FREQUENCY",
                "HAS_DISEASE_CHARACTERISTIC",
                "HAS_GENETIC_ASSOCIATION",
                "DISEASE_OCCURS_IN",
                "HAS_AGE",
                "HAS_GENDER",
                "HAS_ETHNICITY",
                "HAS_DISEASE",
                "HAS_CHARACTERISTIC",
                "HAS_COHORT_SIZE",
            ]

        re_types = allowed_relation_types

        return allowed_entity_types, re_types

    # TO FILES
    REL_TYPES = {"RELATION_EXTRACTION"}  # , "EVENT_EXTRACTION"}


    split_datasets = []
    for split in ("train", "validation", "test"):
        instructions = []
        dataset_name = []
        text_ids = []
        queries = []
        responses = []
        for helper in bb_kb_helpers:
            config_name = helper.config.name

            if (
                config_name in llm_structured_input[split]
                and llm_structured_input[split][config_name] is not None
            ):
                llm_structured_data, text_types = llm_structured_input[split][
                    config_name
                ]
            else:
                print("Skipping", config_name, "as no data was found.")
                continue

            print(
                f"Outputting {len(llm_structured_data)} texts for {config_name}."
            )

            dataset_name.extend([config_name] * len(llm_structured_data))
            
            for text_id, (
                article_text,
                entities,
                relations,
                events,
            ) in llm_structured_data.items():
                queries.append(article_text)
                text_ids.append(f"{config_name}_{text_id}")

                data_str = json.dumps(
                    {"entities": entities, "relations": relations}
                )

                responses.append(data_str)

        split_datasets.append(
            Dataset.from_dict(
                {
                    "input_ids": text_ids,
                    "dataset_name": dataset_name,
                    # "instruction": instructions,
                    "query": queries,
                    "response": responses,
                }
            )
        )


    merged_all = DatasetDict(
        {
            "train": split_datasets[0],
            "validation": split_datasets[1],
            "test": split_datasets[2],
        }
    )

    merged_all.save_to_disk(f"{CORPORA_DATA}/gs_merged_all")

    # dataset_type_definitions
    ners, res, config_names = [], [], []

    for helper in bb_kb_helpers:
        NER, RE = create_task_prompt_template_from_helper(helper)
        config_name = helper.config.name

        ners.append(NER)
        res.append(RE)
        config_names.append(config_name)

    dataset_type_definitions = Dataset.from_dict(
        {"dataset_name": config_names, "ner_types": ners, "re_types": res}
    )

    dataset_type_definitions.to_json(
        f"{CORPORA_DATA}/dataset_type_definitions.json"
    )

    for split in ("train", "validation", "test"):
        # all_text_types = dict()
        clean_name_table = []
        total_text_pieces = 0

        for helper in bb_kb_helpers:
            config_name = helper.config.name
            if (
                config_name in llm_structured_input[split]
                and llm_structured_input[split][config_name] is not None
            ):
                llm_structured_data, text_types = llm_structured_input[split][
                    config_name
                ]
                _, RE = create_task_prompt_template_from_helper(helper)
                total_text_pieces += len(llm_structured_data.keys())

                clean_name_table.append(
                    (
                        helper.config.name,
                        helper.display_name,
                        helper.languages,
                        ("NER & RE" if RE else "NER"),
                        len(llm_structured_data.keys()),
                    )
                )

        print(total_text_pieces, "total pieces of text loaded in split", split)

        with open(
            f"{CORPORA_DATA}/bigbio_names_{split}_w_cnt.tsv", "w"
        ) as f:
            print(
                "config_name",
                "display_name",
                "language",
                "task",
                "count",
                sep="\t",
                file=f,
            )
            for clean_name_row in clean_name_table:
                conf_name, disp_name, langs, task, cnt = clean_name_row
                if len(langs) > 1:
                    lang = "English"
                else:
                    lang = langs[0]
                print(conf_name, disp_name, lang, task, cnt, sep="\t", file=f)


    all_text_types = dict()
    clean_name_table = []
    total_text_pieces_trunc = total_text_pieces = 0

    config_names, split_names, text_len, e_cnt, r_cnt = [], [], [], [], []

    for helper in bb_kb_helpers:
        config_name = helper.config.name
        NER, RE = create_task_prompt_template_from_helper(helper)

        if (
            "test" in bb_kb_datasets[config_name]
            and config_name != "bionlp_st_2019_bb_bigbio_kb"
        ):
            split = "test"
        else:
            split = "validation"

        llm_structured_data, text_types = llm_structured_input[split][config_name]

        if len(llm_structured_data.keys()) < 8:
            print(
                "ignoring",
                config_name,
                "less than 8 samples:",
                len(llm_structured_data.keys()),
            )
            continue

        for text, entities, relationships, events in llm_structured_data.values():
            text_len.append(len(text))
            e_cnt.append(len(entities))
            r_cnt.append(len(relationships))

        split_names.extend(len(llm_structured_data.keys()) * [split])
        config_names.extend(len(llm_structured_data.keys()) * [config_name])

        # print(config_name, len(llm_structured_data.keys()), text_types)
        for text_type in text_types[:512]:
            all_text_types[text_type] = all_text_types.get(text_type, 0) + 1
        total_text_pieces += len(llm_structured_data.keys())
        total_text_pieces_trunc += min(len(llm_structured_data.keys()), 512)
        clean_name_table.append(
            (
                helper.config.name,
                helper.display_name,
                helper.languages,
                ("NER & RE" if RE else "NER"),
                len(llm_structured_data.keys()),
                len(NER),
                len(RE) if RE else "NA",
            )
        )

    print(total_text_pieces, "total pieces of text loaded in usedsplit")
    print(
        total_text_pieces_trunc,
        "total pieces of text loaded in usedsplit (if truncated to 512)",
    )

    bigbio_gs_output_cnts = Dataset.from_dict(
        {
            "config_name": config_names,
            "split": split_names,
            "text_length": text_len,
            "entity_count": e_cnt,
            "relationship_count": r_cnt,
        }
    )
    bigbio_gs_output_cnts.to_csv(
        f"{CORPORA_DATA}/bigbio_gs_output_cnts.tsv", sep="\t"
    )

    with open(
        f"{CORPORA_DATA}/bigbio_names_usedsplits_w_cnt.tsv", "w"
    ) as f:
        print(
            "config_name",
            "display_name",
            "language",
            "task",
            "count",
            "entity_type_count",
            "relation_type_count",
            sep="\t",
            file=f,
        )
        for clean_name_row in clean_name_table:
            conf_name, disp_name, langs, task, cnt, e_t_cnt, r_t_cnt = (
                clean_name_row
            )
            if conf_name == "muchmore_de_bigbio_kb":
                lang = "German"
            elif len(langs) > 1:
                lang = "English"
            else:
                lang = langs[0]
            print(
                conf_name,
                disp_name,
                lang,
                task,
                cnt,
                e_t_cnt,
                r_t_cnt,
                sep="\t",
                file=f,
            )
