from __future__ import annotations

import json
import re


def parse_triplex_format(response):
    """_summary_

    Args:
        response (_type_): _description_

    Returns:
        _type_: _description_
    """
    response = response.strip()
    if response.startswith("```json\n"):
        response = response[8:]

    if response.endswith("```"):
        response = response[:-4]

    try:
        response_dict = json.loads(response)
    except json.JSONDecodeError:
        return

    if "entities_and_triples" not in response_dict:
        return

    entity_dict = {}
    save_relations = []
    relations = []

    for entry in response_dict["entities_and_triples"]:
        if re.match(r"^\[\d+\],", entry):  # Entities
            entity_id, entity = entry.split(", ", 1)
            entity_type, entity_name = entity.split(":", 1)
            entity_type = entity_type.strip()
            entity_name = entity_name.strip()

            entity_dict[entity_id] = (entity_name, entity_type)
        else:  # Relations
            try:
                head_id, rel_and_tail_id = entry.split(" ", 1)
                rel, tail_id = rel_and_tail_id.rsplit(" ", 1)
            except ValueError:
                continue
            save_relations.append((head_id, rel, tail_id))

    for head, rel, tail in save_relations:
        head = entity_dict[head_id]
        tail = entity_dict[tail_id]

        relations.append(((head, rel, tail), None))

    entities = list(entity_dict.values())
    return entities, relations
