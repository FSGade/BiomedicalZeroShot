#!/usr/bin/env python3
from __future__ import annotations

import pickle
from pathlib import Path

from datasets import load_dataset
from rte_utils.benchmarking import calculate_metrics

dataset_type_definitions = load_dataset(
    "json",
    data_files="dataset_type_definitions.json",
)

benchmark_output_dir = Path.cwd() / "Benchmarking" / "outputs"

benchmark_summary = []

benchmark_files = list(benchmark_output_dir.glob("*/*/pred_output.pkl"))
benchmark_files_subfolders = list(
    benchmark_output_dir.glob("*/*/*/pred_output.pkl")
)

standard_benchmark = len(benchmark_files) * [True] + len(
    benchmark_files_subfolders
) * [False]

all_benchmark_files = benchmark_files + benchmark_files_subfolders

for i, (is_standard, filename) in enumerate(
    zip(standard_benchmark, all_benchmark_files)
):
    if is_standard:
        config_name, used_set = filename.parent.stem.rsplit("_", 1)
        model_identifier = filename.parent.parent.stem
        subanalysis = "-"
    else:
        config_name, used_set = filename.parent.parent.stem.rsplit("_", 1)
        model_identifier = filename.parent.parent.parent.stem
        subanalysis = filename.parent.stem

    if not config_name.startswith("bc5cdr_bigbio_kb"):
    #if not config_name.startswith("biorelex_bigbio_kb"):
        continue
    if not model_identifier.startswith("gemini"):
        continue
    # if subanalysis != "-":
    #     continue
    
    

    dataset_type_definition = dataset_type_definitions["train"].filter(
        lambda x: x["dataset_name"] == config_name
    )[0]

    ner_types = dataset_type_definition["ner_types"]
    re_types = dataset_type_definition["re_types"]

    print(
        f"{config_name}, {model_identifier}, {subanalysis} ({i} of {len(standard_benchmark)})"
    )

    with open(filename.parent / "pred_output.pkl", "rb") as f_in:
        pred_output = pickle.load(f_in)

    with open(filename.parent / "true_output.pkl", "rb") as f_in:
        true_output = pickle.load(f_in)

    # BENCHMARKING
    # Paper benchmark
    # benchmarking_metrics = calculate_metrics(
    #     pred_output,
    #     true_output,
    #     ignore_case=True,
    #     # restrict_types=False,
    #     restrict_types=True,
    #     # ignore_directionality=True,
    #     ignore_directionality=True,
    #     ner_types=ner_types,
    #     re_types=re_types,
    #     # match_criteria="strict"
    #     match_criteria="partial_strict",
    #     # match_criteria="partial"
    # )
    
    
    # Appendix benchmark
    benchmarking_metrics = calculate_metrics(
        pred_output,
        true_output,
        ignore_case=False,
        #ignore_case=True,
        #restrict_types=False,
        restrict_types=True,
        ignore_directionality=False,
        # ignore_directionality=True,
        ner_types=ner_types,
        re_types=re_types,
        match_criteria="strict",
        # match_criteria="partial_strict",
        # match_criteria="partial"
        print_examples=10
    )

    # benchmarking_metrics = calculate_metrics(
    #     pred_output,
    #     true_output,
    #     ignore_case=True,
    #     restrict_types=False,
    #     # ignore_directionality=True,
    #     ignore_directionality=True,
    #     ner_types=ner_types,
    #     re_types=re_types,
    #     # match_criteria="strict"
    #     match_criteria="partial_strict",
    #     # match_criteria="partial"
    # )

    with open(filename.parent / "benchmarking_metrics.pkl", "wb") as f_out:
        pickle.dump(benchmarking_metrics, f_out)
