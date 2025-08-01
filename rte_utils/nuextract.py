from __future__ import annotations

import json

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


def predict_NuExtract(
    model,
    tokenizer,
    texts,
    template,
    batch_size=1,
    max_length=10_000,
    max_new_tokens=4_000,
):
    template = json.dumps(json.loads(template), indent=4)
    prompts = [
        f"""<|input|>\n### Template:\n{template}\n### Text:\n{text}\n\n<|output|>"""
        for text in texts
    ]
    outputs = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            batch_encodings = tokenizer(
                batch_prompts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            ).to(model.device)
            pred_ids = model.generate(
                **batch_encodings, max_new_tokens=max_new_tokens
            )
            outputs += tokenizer.batch_decode(
                pred_ids, skip_special_tokens=True
            )
    return [output.split("<|output|>")[1] for output in outputs]


model_name = "numind/NuExtract-v1.5"
device = "cpu"
model = (
    AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    .to(device)
    .eval()
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

text = """Probing the salmeterol binding site on the beta 2-adrenergic receptor using a novel photoaffinity ligand, [(125)I]iodoazidosalmeterol.\nSalmeterol is a long-acting beta2-adrenergic receptor (beta 2AR) agonist used clinically to treat asthma. In addition to binding at the active agonist site, it has been proposed that salmeterol also binds with very high affinity at a second site, termed the \"exosite\", and that this exosite contributes to the long duration of action of salmeterol. To determine the position of the phenyl ring of the aralkyloxyalkyl side chain of salmeterol in the beta 2AR binding site, we designed and synthesized the agonist photoaffinity label [(125)I]iodoazidosalmeterol ([125I]IAS). In direct adenylyl cyclase activation, in effects on adenylyl cyclase after pretreatment of intact cells, and in guinea pig tracheal relaxation assays, IAS and the parent drug salmeterol behave essentially the same. Significantly, the photoreactive azide of IAS is positioned on the phenyl ring at the end of the molecule which is thought to be involved in exosite binding. Carrier-free radioiodinated [125I]IAS was used to photolabel epitope-tagged human beta 2AR in membranes prepared from stably transfected HEK 293 cells. Labeling with [(125)I]IAS was blocked by 10 microM (-)-alprenolol and inhibited by addition of GTP gamma S, and [125I]IAS migrated at the same position on an SDS-PAGE gel as the beta 2AR labeled by the antagonist photoaffinity label [125I]iodoazidobenzylpindolol ([125I]IABP). The labeled receptor was purified on a nickel affinity column and cleaved with factor Xa protease at a specific sequence in the large loop between transmembrane segments 5 and 6, yielding two peptides. While the control antagonist photoaffinity label [125I]IABP labeled both the large N-terminal fragment [containing transmembranes (TMs) 1-5] and the smaller C-terminal fragment (containing TMs 6 and 7), essentially all of the [125I]IAS labeling was on the smaller C-terminal peptide containing TMs 6 and 7. This direct biochemical evidence demonstrates that when salmeterol binds to the receptor, its hydrophobic aryloxyalkyl tail is positioned near TM 6 and/or TM 7. A model of IAS binding to the beta 2AR is proposed."""

ner_types = ["Chemical", "GeneOrProtein"]
re_types = [
    "DOWNREGULATES",
    "IS_AGONIST_FOR",
    "IS_ANTAGONIST_FOR",
    "IS_SUBSTRATE_FOR",
    "UPREGULATES",
]

template = """{
    "Chemical": [],
    "GeneOrProtein": []
}"""

template = """{
    "UPREGULATES": [{"Arg1": {"name": ""}, "Arg2": {"name": ""}},
    {"Arg1": {"name": ""}, "Arg2": {"name": ""}},
    {"Arg1": {"name": ""}, "Arg2": {"name": ""}}],
    "DOWNREGULATES": [{"Arg1": {"name": ""}, "Arg2": {"name": ""}},
    {"Arg1": {"name": ""}, "Arg2": {"name": ""}},
    {"Arg1": {"name": ""}, "Arg2": {"name": ""}}]
}"""

template = """{
    "Model": {
        "Name": "",
        "Number of parameters": "",
        "Number of max token": "",
        "Architecture": []
    },
    "Usage": {
        "Use case": [],
        "Licence": ""
    }
}"""


text = """We introduce Mistral 7B and Qwen 7B, 7–billion-parameter language models engineered for
superior performance and efficiency. Mistral 7B outperforms the best open 13B
model (Llama 2) across all evaluated benchmarks, and the best released 34B
model (Llama 1) in reasoning, mathematics, and code generation. Our model
leverages grouped-query attention (GQA) for faster inference, coupled with sliding
window attention (SWA) to effectively handle sequences of arbitrary length with a
reduced inference cost. We also provide a model fine-tuned to follow instructions,
Mistral 7B – Instruct, that surpasses Llama 2 13B – chat model both on human and
automated benchmarks. Our models are released under the Apache 2.0 license.
Code: <https://github.com/mistralai/mistral-src>
Webpage: <https://mistral.ai/news/announcing-mistral-7b/>"""

template = """{
    "Model": {
        "Name": "",
        "Number of parameters": "",
        "Number of max token": "",
        "Architecture": []
    },
    "Usage": {
        "Use case": [],
        "Licence": ""
    }
}"""


prediction = predict_NuExtract(model, tokenizer, [text], template)[0]
print(prediction)
