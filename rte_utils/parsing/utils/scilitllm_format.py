from __future__ import annotations


def parse_scilitllm_format(response):
    response = response.strip()

    response_list = response[1:-1].split("), (")

    entities_type_dict = dict()
    relationships = []

    for example_str in response_list:
        example = example_str.split(", ")
        if len(example) == 2:  # Entity
            entity_name, entity_type = example
            entity_name = entity_name.strip()
            entity_type = entity_type.strip()
            entities_type_dict[entity_name] = entity_type
        elif len(example) == 3:  # Relationship
            head = example[0]
            rel = example[1]
            tail = example[2]

            head_type = entities_type_dict.get(head, "unknown")
            tail_type = entities_type_dict.get(tail, "unknown")

            relationships.append(
                (((head, head_type), rel, (tail, tail_type)), None)
            )

    entities = list(entities_type_dict.items())

    return entities, relationships
