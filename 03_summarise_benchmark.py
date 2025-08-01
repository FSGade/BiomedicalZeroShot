#!/usr/bin/env python3
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_colwidth", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.expand_frame_repr", False)

# benchmark_dirs = f"Benchmarking/outputs/{model_identifier}/{config_name}_{used_set}"
benchmark_output_dir = Path.cwd() / "Benchmarking" / "outputs"

benchmark_summary = []
benchmark_detailed_summary = []

benchmark_files = list(
    benchmark_output_dir.glob("*/*/benchmarking_metrics.pkl")
)
benchmark_files_subfolders = list(
    benchmark_output_dir.glob("*/*/*/benchmarking_metrics.pkl")
)

standard_benchmark = len(benchmark_files) * [True] + len(
    benchmark_files_subfolders
) * [False]

all_benchmark_files = benchmark_files + benchmark_files_subfolders

f1_type = "micro"

for is_standard, filename in zip(standard_benchmark, all_benchmark_files):
    with filename.open(mode="rb") as f:
        benchmarking_metrics = pickle.load(f)

    ner_metrics, re_metrics = benchmarking_metrics
    ner_f1, ner_precision, ner_recall, _ = ner_metrics[f1_type]
    re_f1, re_precision, re_recall, _ = re_metrics[f1_type]

    if is_standard:
        config_name, used_set = filename.parent.stem.rsplit("_", 1)
        model_identifier = filename.parent.parent.stem
        subanalysis = "-"
    else:
        config_name, used_set = filename.parent.parent.stem.rsplit("_", 1)
        model_identifier = filename.parent.parent.parent.stem
        subanalysis = filename.parent.stem
    
    if config_name in ("bc5cdr_bigbio_kb", "jnlpba_bigbio_kb", "chemdner_bigbio_kb", "biored_bigbio_kb", "cellfinder_bigbio_kb", "pubtator_central_sample_bigbio_kb"):
        continue

    benchmark_summary.append(
        [
            model_identifier,
            config_name,
            used_set,
            ner_f1,
            ner_precision,
            ner_recall,
            re_f1,
            re_precision,
            re_recall,
            subanalysis,
        ]
    )

    ner_metrics_by_type = ner_metrics["by_type"]
    re_metrics_by_type = re_metrics["by_type"]

    for ner_type, (
        f1,
        precision,
        recall,
        support,
    ) in ner_metrics_by_type.items():
        benchmark_detailed_summary.append(
            [
                model_identifier,
                config_name,
                used_set,
                subanalysis,
                "ner",
                ner_type,
                f1,
                precision,
                recall,
                support,
            ]
        )
    for re_type, (
        f1,
        precision,
        recall,
        support,
    ) in re_metrics_by_type.items():
        benchmark_detailed_summary.append(
            [
                model_identifier,
                config_name,
                used_set,
                subanalysis,
                "re",
                re_type,
                f1,
                precision,
                recall,
                support,
            ]
        )

benchmark_detailed_summary_df = pd.DataFrame(
    benchmark_detailed_summary,
    columns=[
        "model_identifier",
        "config_name",
        "used_set",
        "subanalysis",
        "task",
        "type",
        "f1",
        "precision",
        "recall",
        "support",
    ],
)

benchmark_detailed_summary_df.to_csv(
    "benchmark_detailed_summary.tsv", sep="\t", index=False
)


benchmark_summary_df = pd.DataFrame(
    benchmark_summary,
    columns=[
        "model_identifier",
        "config_name",
        "used_set",
        "ner_f1",
        "ner_precision",
        "ner_recall",
        "re_f1",
        "re_precision",
        "re_recall",
        "subanalysis",
    ],
)

benchmark_summary_df.to_csv("benchmark_summary.tsv", sep="\t", index=False)

import numpy as np

benchmark_summary_df["re_f1"] = benchmark_summary_df["re_f1"].replace(
    0, np.nan
)
benchmark_summary_df["re_precision"] = benchmark_summary_df[
    "re_precision"
].replace(0, np.nan)
benchmark_summary_df["re_recall"] = benchmark_summary_df["re_recall"].replace(
    0, np.nan
)

wm = lambda x: np.average(
    x, weights=benchmark_detailed_summary_df.loc[x.index, "support"]
)

summary_by_type = (
    benchmark_detailed_summary_df[
        (benchmark_detailed_summary_df["support"] != 0)
        & (benchmark_detailed_summary_df["subanalysis"] == "-")
    ]
    .groupby(["type", "model_identifier", "task"])
    .agg(support=("support", "sum"), f1=("f1", wm))
    .reset_index()
    .sort_values(["f1", "type"], ascending=False)
    .groupby(["type"])
    .head(20)
)

summary_by_type.to_csv("benchmark_summary_by_type.tsv", sep="\t", index=True)

print("Average performance across all datasets benchmarked:")
total_summary = (
    benchmark_summary_df[benchmark_summary_df["subanalysis"] == "-"]
    .groupby(["model_identifier", "subanalysis"])
    .filter(lambda x: len(x) > 10)
)

print(len(total_summary["model_identifier"].unique()))
total_summary = (
    total_summary.groupby(["config_name"])
    .filter(
        lambda x: len(x) == len(total_summary["model_identifier"].unique())
    )
    .groupby(["model_identifier"])
    .mean(numeric_only=True)
    .round(3)
)
print(total_summary)
print()

print("Number of models benchmarked per dataset (zero-shot analysis):")
print(
    benchmark_summary_df[(benchmark_summary_df["subanalysis"] == "-")]
    .groupby(["config_name"])
    .count()
)
print()

print("Number of datasets benchmarked:")
print(
    benchmark_summary_df.groupby(["model_identifier", "subanalysis"]).count()
)
print()

print("BC5CDR performance:")
bc5cdr_benchmark = (
    benchmark_summary_df[
        benchmark_summary_df["config_name"] == "bc5cdr_bigbio_kb"
    ]
    .groupby(["model_identifier", "subanalysis"])
    .mean(numeric_only=True)
    .round(3)
)
print(bc5cdr_benchmark)
bc5cdr_benchmark.to_csv("bc5cdr_benchmark.csv", index=True)
print()

print("BioRed performance:")
biored_benchmark = (
    benchmark_summary_df[
        benchmark_summary_df["config_name"] == "biored_bigbio_kb"
    ]
    .groupby(["model_identifier", "subanalysis"])
    .mean(numeric_only=True)
    .round(3)
)
print(biored_benchmark)
biored_benchmark.to_csv("biored_benchmark.csv", index=True)
print()


print("JNLPBA performance:")
jnlpba_benchmark = (
    benchmark_summary_df[
        benchmark_summary_df["config_name"] == "jnlpba_bigbio_kb"
    ]
    .groupby(["model_identifier", "subanalysis"])
    .mean(numeric_only=True)
    .round(3)
)
print(jnlpba_benchmark)
print()


print("ChemDNER performance:")
chemdner_benchmark = (
    benchmark_summary_df[
        benchmark_summary_df["config_name"] == "chemdner_bigbio_kb"
    ]
    .groupby(["model_identifier", "subanalysis"])
    .mean(numeric_only=True)
    .round(3)
)
print(chemdner_benchmark)
print()

print("BioInfer performance:")
bioinfer_benchmark = (
    benchmark_summary_df[
        benchmark_summary_df["config_name"] == "bioinfer_bigbio_kb"
    ]
    .groupby(["model_identifier", "subanalysis"])
    .mean(numeric_only=True)
    .round(3)
)
print(bioinfer_benchmark)
bioinfer_benchmark.to_csv("bioinfer_benchmark.csv", index=True)
print()

print("BioRelEx performance:")
biorelex_benchmark = (
    benchmark_summary_df[
        benchmark_summary_df["config_name"] == "biorelex_bigbio_kb"
    ]
    .groupby(["model_identifier", "subanalysis"])
    .mean(numeric_only=True)
    .round(3)
)
print(biorelex_benchmark)
biorelex_benchmark.to_csv("biorelex_benchmark.csv", index=True)
print()

print("GENETAG performance:")
genetag_benchmark = (
    benchmark_summary_df[
        benchmark_summary_df["config_name"] == "genetaggold_bigbio_kb"
    ]
    .groupby(["model_identifier", "subanalysis"])
    .mean(numeric_only=True)
    .round(3)
)
print(genetag_benchmark)
print()

print("AnEM performance:")
anem_benchmark = (
    benchmark_summary_df[
        benchmark_summary_df["config_name"] == "an_em_bigbio_kb"
    ]
    .groupby(["model_identifier", "subanalysis"])
    .mean(numeric_only=True)
    .round(3)
)
print(anem_benchmark)
print()

print("AnatEM performance:")
anatem_benchmark = (
    benchmark_summary_df[
        benchmark_summary_df["config_name"] == "anat_em_bigbio_kb"
    ]
    .groupby(["model_identifier", "subanalysis"])
    .mean(numeric_only=True)
    .round(3)
)
print(anatem_benchmark)
print()

print("CPI performance:")
cpi_benchmark = (
    benchmark_summary_df[
        benchmark_summary_df["config_name"] == "cpi_bigbio_kb"
    ]
    .groupby(["model_identifier", "subanalysis"])
    .mean(numeric_only=True)
    .round(3)
)
print(cpi_benchmark)
print()


benchmark_summary_df = benchmark_summary_df[
    benchmark_summary_df["subanalysis"] == "-"
].dropna(subset=["ner_f1"])

KGC_SFT_benchmark = (
    benchmark_summary_df[
        benchmark_summary_df["config_name"].isin(
            [
                "bc5cdr_bigbio_kb",
                "anem_bigbio_kb",
                "jnlpba_bigbio_kb",
                "biored_bigbio_kb",
                "chemdner_bigbio_kb",
            ]
        )
    ]
    .sort_values(["config_name", "ner_f1"])
    .round(3)
)
print(KGC_SFT_benchmark)
print()

# print(benchmark_summary_df.groupby("config_name").mean(numeric_only=True))
print(benchmark_summary_df.groupby("config_name").max(numeric_only=True))

print(
    benchmark_summary_df.sort_values(["config_name", "ner_f1"])
    .groupby("config_name")
    .tail(1)
)

performance_ranks = benchmark_summary_df.sort_values(
    ["config_name", "ner_f1"]
).groupby("config_name")

benchmark_summary_df["ner_rank"] = performance_ranks["ner_f1"].rank(
    "average", ascending=False
)
print(benchmark_summary_df.groupby("model_identifier")["ner_rank"].median())


print(
    benchmark_summary_df.dropna(subset=["ner_f1"])
    .sort_values(["config_name", "ner_f1"])
    .groupby("config_name")
    .tail(1)
    .value_counts("model_identifier")
)

print(
    benchmark_summary_df[benchmark_summary_df["re_f1"] != 0]
    .dropna(subset=["re_f1"])
    .sort_values(["config_name", "re_f1"])
    .groupby("config_name")
    .tail(1)
)
print(
    benchmark_summary_df[benchmark_summary_df["re_f1"] != 0]
    .dropna(subset=["re_f1"])
    .sort_values(["config_name", "re_f1"])
    .groupby("config_name")
    .tail(1)
    .value_counts("model_identifier")
)

# print("Gemini performance:")
# gemini_benchmark = (
#     benchmark_summary_df[
#        (benchmark_summary_df["model_identifier"] == "gemini_1_5_pro") & (benchmark_summary_df["subanalysis"] == "-")
#     ]
#     .sort_values(["config_name", "re_f1"])
#     .round(3)
# )
# print(gemini_benchmark)

# print("scilitllm1_5_14b performance:")
# scilit_benchmark = (
#     benchmark_summary_df[
#        (benchmark_summary_df["model_identifier"] == "scilitllm1_5_14b") & (benchmark_summary_df["subanalysis"] == "-")
#     ]
#     .sort_values(["config_name", "re_f1"])
#     .round(3)
# )
# print(scilit_benchmark)
