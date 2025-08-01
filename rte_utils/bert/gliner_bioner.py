#!/usr/bin/env python3
from __future__ import annotations

from typing import List
from typing import Optional

from gliner import GLiNER


def parse_gliner_entities(entities, relations=None):
    parsed = []

    if relations is None:
        relations = [[] for _ in range(len(entities))]

    for abstract_entities, abstract_relations in zip(entities, relations):
        parsed.append(
            {
                "entities": [
                    (entity_name, entity_type)
                    for (
                        entity_name,
                        (prob, entity_type),
                    ) in abstract_entities.items()
                ],
                "relations": abstract_relations,
            }
        )

    return parsed


def merge_entities(entities, text):
    if not entities:
        return []
    merged = []
    current = entities[0]
    for next_entity in entities[1:]:
        if next_entity["label"] == current["label"] and (
            next_entity["start"] == current["end"] + 1
            or next_entity["start"] == current["end"]
        ):
            current["text"] = text[
                current["start"] : next_entity["end"]
            ].strip()
            current["end"] = next_entity["end"]
        else:
            merged.append(current)
            current = next_entity
    # Append the last entity
    merged.append(current)
    return merged


def batch_predict_entities(
    texts: list[str],
    ner_types: list[str],
    re_types: list[str] | None = None,
    model_name: str = "urchade/gliner_medium-v2.5",
    max_length: int = 384,
    threshold: float = 0.5,
):
    # Initialize GLiNER with the base model
    # "urchade/gliner_large-v2.1"
    print(max_length)
    model = GLiNER.from_pretrained(model_name, max_len=max_length)
    model.config.max_len = max_length

    if model_name == "urchade/gliner_large_bio-v0.1":
        threshold = 0.3

    if "NuNER" in model_name:
        model.data_processor.transformer_tokenizer.add_prefix_space = True
        ner_types = [l.lower() for l in ner_types]

    all_entities = []
    rel_separator = " <> "

    # Sample text for entity prediction
    for text in texts:
        entity_dict = dict()
        # Perform entity prediction
        entities = model.predict_entities(text, ner_types, threshold=threshold)

        if "NuNER" in model_name:
            entities = merge_entities(entities, text)

        # [{'start': 50, 'end': 59, 'text': 'clonidine', 'label': 'Chemical', 'score': 0.8339420557022095}, ...]

        # Display predicted entities and their labels
        for entity in entities:
            entity_name, entity_type = entity["text"], entity["label"]
            prob = entity["score"]

            if (
                entity_name not in entity_dict
                or entity_dict[entity_name][0] < prob
            ):
                entity_dict[entity_name] = (prob, entity_type)

        all_entities.append(entity_dict)

    if re_types is None or not "multitask" in model_name:
        return parse_gliner_entities(all_entities)

    all_relations = []

    for i, text in enumerate(texts):
        labels = []
        entity_dict = all_entities[i]
        for entity_name in entity_dict.keys():
            for re_type in re_types:
                labels.append(f"{entity_name}{rel_separator}{re_type}")

        if not labels:
            all_relations.append([])
            continue

        relations_pred = model.predict_entities(text, labels)
        relations = []
        for relation in relations_pred:
            head_name, re_type = relation["label"].split(rel_separator)
            tail_name = relation["text"]
            try:
                head_type = entity_dict[head_name][1]
                tail_type = entity_dict[tail_name][1]
            except KeyError:
                continue
            relations.append(
                (
                    ((head_name, head_type), re_type, (tail_name, tail_type)),
                    None,
                )
            )

        all_relations.append(relations)

    return parse_gliner_entities(all_entities, all_relations)
