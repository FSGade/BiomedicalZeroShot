#!/usr/bin/env python3
from __future__ import annotations

import warnings
from typing import List
from typing import Set

import numpy as np
from scipy.special import softmax
from transformers import AutoTokenizer
from transformers import BertForTokenClassification
from transformers.tokenization_utils_base import BatchEncoding


def logits_to_words(
    logits: np.ndarray, original_string: str, encodings: BatchEncoding
) -> set[str]:
    probs = softmax(logits, axis=1)
    pred_token_indices = np.flatnonzero(np.argmax(probs, axis=1))
    pred_token_indices_grouped = np.split(
        pred_token_indices, np.where(np.diff(pred_token_indices) != 1)[0] + 1
    )
    word_probs = dict()
    for group in pred_token_indices_grouped:
        span_group = [encodings.token_to_chars(idx) for idx in group]
        chars = []
        prev_span_end = None
        for span in span_group:
            if prev_span_end is not None and prev_span_end != span[0]:
                chars.append(" ")
            chars.append(original_string[span[0] : span[1]])
            prev_span_end = span[1]
        word = "".join(chars)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            word_prob = np.mean(probs[group, 1])
        if not np.isnan(word_prob):
            word_probs[word] = max(word_prob, word_probs.get(word, 0))
    return word_probs


def parse_zeroshotbioner_entities(entities):
    parsed = []

    for abstract_entities in entities:
        parsed.append(
            {
                "entities": [
                    (entity_name, entity_type)
                    for (
                        entity_name,
                        (prob, entity_type),
                    ) in abstract_entities.items()
                ],
                "relations": [],
            }
        )

    return parsed


def batch_predict_entities(
    texts: list[str],
    ner_types: list[str],
    model_name: str = "ProdicusII/ZeroShotBioNER",
    threshold: float = 0.5,
    max_length: int = 512,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )  ## loading the tokenizer of that model
    model = BertForTokenClassification.from_pretrained(
        model_name, num_labels=2
    )
    entities = []
    for text in texts:
        text_entities_probs = None
        for entity_type in ner_types:
            encodings = tokenizer(
                entity_type,
                text,
                is_split_into_words=False,
                padding=True,
                truncation=True,
                add_special_tokens=True,
                return_offsets_mapping=False,
                max_length=max_length,
                return_tensors="pt",
            )
            token_classifier_output = model(**encodings)
            prediction_logits = token_classifier_output["logits"]
            prediction_logits = prediction_logits[0].detach().numpy()
            pred_words_probs = logits_to_words(
                prediction_logits, text, encodings
            )
            if text_entities_probs is None:
                text_entities_probs = {
                    word: (prob, entity_type)
                    for (word, prob) in pred_words_probs.items()
                }
            else:
                for word, prob in pred_words_probs.items():
                    if (
                        not (
                            word in text_entities_probs
                            and prob < text_entities_probs[word][0]
                        )
                        and prob > threshold
                    ):
                        text_entities_probs[word] = (prob, entity_type)
            # text_entities.extend([(word, entity_type) for word in pred_words])
        # entities.append(text_entities)
        entities.append(text_entities_probs)

    return parse_zeroshotbioner_entities(entities)


# ner_types = ["Chemical", "Disease"]
# string1 = """Naloxone reverses the antihypertensive effect of clonidine.In unanesthetized, spontaneously hypertensive rats the decrease in blood pressure and heart rate produced by intravenous clonidine, 5 to 20 micrograms/kg, was inhibited or reversed by nalozone, 0.2 to 2 mg/kg. The hypotensive effect of 100 mg/kg alpha-methyldopa was also partially reversed by naloxone. Naloxone alone did not affect either blood pressure or heart rate. In brain membranes from spontaneously hypertensive rats clonidine, 10(-8) to 10(-5) M, did not influence stereoselective binding of [3H]-naloxone (8 nM), and naloxone, 10(-8) to 10(-4) M, did not influence clonidine-suppressible binding of [3H]-dihydroergocryptine (1 nM). These findings indicate that in spontaneously hypertensive rats the effects of central alpha-adrenoceptor stimulation involve activation of opiate receptors. As naloxone and clonidine do not appear to interact with the same receptor site, the observed functional antagonism suggests the release of an endogenous opiate by clonidine or alpha-methyldopa and the possible role of the opiate in the central control of sympathetic tone."""
# string2 = """Naloxone reverses the antihypertensive effect of clonidine.In unanesthetized, spontaneously hypertensive rats the decrease in blood pressure and heart rate produced by intravenous clonidine, 5 to 20 micrograms/kg, was inhibited or reversed by nalozone, 0.2 to 2 mg/kg. The hypotensive effect of 100 mg/kg alpha-methyldopa was also partially reversed by naloxone."""

# ner_types = ["DNAFamilyOrGroup", "Gene", "GeneOrProteinOrRNA", "IndividualProtein", "ProteinComplex", "ProteinFamilyOrGroup"]
# string1 = "alpha-catenin inhibits beta-catenin signaling by preventing formation of a beta-catenin*T-cell factor*DNA complex."
# strings = [string1]#, string2]

# res = batch_predict_entities(strings, ner_types=ner_types, threshold = 0.8)


# https://huggingface.co/ProdicusII/ZeroShotBioNER
# training_args = TrainingArguments(
#         output_dir=os.path.join('Results', class_unseen, str(j)+'Shot'),  # folder for results
#         num_train_epochs=10,                                              # number of epochs
#         per_device_train_batch_size=16,                                   # batch size per device during training
#         per_device_eval_batch_size=16,                                    # batch size for evaluation
#         weight_decay=0.01,                                                # strength of weight decay
#         logging_dir=os.path.join('Logs', class_unseen, str(j)+'Shot'),    # folder for logs
#         save_strategy='epoch',
#         evaluation_strategy='epoch',
#         load_best_model_at_end=True,
#     )

# model0 = BertForTokenClassification.from_pretrained(model_path, num_labels=2)
# trainer = Trainer(
#     model=model0,                # pretrained model
#     args=training_args,          # training artguments
#     train_dataset=dataset,       # Object of class torch.utils.data.Dataset for training
#     eval_dataset=dataset_valid   # Object of class torch.utils.data.Dataset for vaLidation
#     )
# start_time = time.time()
# trainer.train()
# total_time = time.time()-start_time
# model0_path = os.path.join('Results', class_unseen, str(j)+'Shot', 'Model')
# os.makedirs(model0_path, exist_ok=True)
# trainer.save_model(model0_path)
