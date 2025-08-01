# Benchmarking zero-shot biomedical relation triplet extraction across language model architectures

This repository contains the code that accompanies the 2025 BioNLP paper "Benchmarking zero-shot biomedical relation triplet extraction across language model architectures" (see citation below).

## Project Structure

The repository is organized as follows:

- `01_corpora.py`: Dataset handling and preprocessing
- `02_benchmark.py`: Main script for running LLM benchmarking experiments
- `03_summarise_benchmark.py`: LLM benchmark summariser
- `04_benchmark_plots.R`: Plot scripts (in R)
- `rte_utils/`: Utility functions for relation triplet extraction
  - `benchmarking/`: Evaluation metrics and utilities
  - `bert/`: BERT model implementations
  - `llm/`: LLM-specific implementations
  - `parsing/`: Response parsing utilities

## Installation
```bash
# Clone the repository
git clone https://github.com/FSGade/BiomedicalZeroShot

# Create conda environment from file
conda env create -f environment.yml

# Activate the environment
conda activate biomed0shot

# Clone BigBIO fork into working directory and install
git clone https://github.com/FSGade/biomedical bigbio
```

## Usage
First, set the environment variables `CORPORA_DATA` (where the corpora data will be stored) and, if using remote models, `OPENAI_API_BASE` and `OPENAI_API_KEY`.

### 1. Data Preparation
First, prepare your input data:
```bash
# Fetch and preprocess corpora
python 01_corpora.py
```

### 2. Running Benchmarks

#### Basic LM Benchmarking
The main benchmarking script `02_benchmark.py` supports all models and configurations (except InstructUIE):

```bash
# Basic usage
python 02_benchmark.py MODEL_ID DATASET_INDEX [flags]

# Example with SciLitLLM 1.5
python 02_benchmark.py scilitllm1_5_7b 1 --temperature 0

# Example with Gemini 1.5 Pro and ICL (5-shot)
python 02_benchmark.py gemini_1_5_pro 1 --icl_count 5
```

Do note that some large models were provided through a custom endpoint, and thus a standard OpenAI API endpoint does not necessarily support all models specified.

#### InstructUIE
Run the setup instructions from the original repo, [BeyonderXX/InstructUIE](https://github.com/BeyonderXX/InstructUIE)

```bash
python rte_utils/bert/instructuie.py [same_flags_as_02_benchmark]
```

### 3. Evaluation and Analysis

```bash
# Summarize benchmark results
python 03_summarise_benchmark.py

# Generate the paper plots
python 04_benchmark_plots.R
```

### Important Parameters

#### Benchmark Configuration
- `--max_input_tokens`: Maximum input context length (default: 128000)
- `--max_output_tokens`: Maximum generated response length (default: 4096)
- `--batch_size`: Number of samples to process in parallel (default: 16)
- `--temperature`: Sampling temperature, lower means more deterministic (default: 0)
- `--icl_count`: Number of in-context learning examples (default: 3)

#### Dataset Control
- `--limit_samples`: Whether to use a subset of data (default: True)
- `--limit_to_number`: Maximum number of samples when limiting (default: 512)

#### Advanced Options
- `--prompt_type`: Specify custom prompt template
- `--prompting_techniques`: List of prompting strategies to evaluate
- `--rerun_llm`: Force re-evaluation instead of using cached results

## License

### Code
MIT License, see LICENSE file.

### Data
Different licences apply to each datasets used in the benchmark, and is described in their original publications or repos released alongside them.

## Citation
If you use this code in your research, please cite:

```bibtex
@inproceedings{Gade2025,
    title={Benchmarking zero-shot biomedical relation triplet extraction across language model architectures},
    author={Gade, Frederik Steensgaard and Lund, Ole and Mendoza, Marie Lisandra Zepeda},
    editor = {Demner-Fushman, Dina and Ananiadou, Sophia and Miwa, Makoto and Tsujii, Junichi},
    booktitle={Proceedings of the 24th Workshop on Biomedical Natural Language Processing},
    month={aug},
    year={2025},
    address = {Vienna, Austria},
    publisher={Association for Computational Linguistics},
    url={https://aclanthology.org/2025.bionlp-1.9/},
    pages = {88--100},
    ISBN = {979-8-89176-275-6},
    abstract = "Many language models (LMs) in the literature claim excellent zero-shot and/or few-shot capabilities for named entity recognition (NER) and relation extraction (RE) tasks and assert their ability to generalize beyond their training datasets. However, these claims have yet to be tested across different model architectures. This paper presents a performance evaluation of zero-shot relation triplet extraction (NER followed by RE of the entities) for both small and large LMs, utilizing 13,867 texts from 61 biomedical corpora and encompassing 151 unique entity types. This comprehensive evaluation offers valuable insights into the practical applicability and performance of LMs within the intricate domain of biomedical relation triplet extraction, highlighting their effectiveness in managing a diverse range of relations and entity types. Gemini 1.5 Pro, the largest LM included in the study, was the top-performing zero-shot model, achieving an average partial match micro F1 of 0.492 for NER, followed closely by SciLitLLM 1.5 14B with a score of 0.475. Fine-tuned models generally outperformed others on the corpora they were trained on, even in a few-shot setting, but struggled to generalize across all datasets with similar entity types. No models achieved an F1 score above 0.5 for the RTE task on any dataset, and their scores fluctuated based on the specific class of entity and the dataset involved. This observation highlights that there is still large room for improvement on the zero-shot utility of LMs in biomedical RTE applications."
}
```

## Acknowledgments

The work of FSG is partly funded by the Innovation Fund Denmark (IFD) under File No. 3129-00056 and co-financed through a Novo Nordisk R&ED novoSTAR Industrial PhD fellowship.

Additionally, thanks to Jesper Ferkinghoff-Borg, Julien Fauqueur, and Robert R. Kitchen for their help and support for this project.

## Contact

For questions or issues, please contact:
- Email: fzsg@novonordisk.com
