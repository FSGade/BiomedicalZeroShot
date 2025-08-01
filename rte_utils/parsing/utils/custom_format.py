from __future__ import annotations

import re


def parse_entity(entry):
    if entry == "":
        return None

    _entry = entry.strip()[:-1].rsplit(" (", 1)
    if len(_entry) == 1:
        return None

    entity_name, entity_type = _entry

    return entity_name.strip(), entity_type.strip()


def parse_context(context):
    if context is None:
        return

    context_list = context.split("; ")
    # print(context_list)
    context_dict = dict()
    for entry in context_list:
        _entry = entry.split(": ", 1)
        if len(_entry) == 1:
            continue
        context_name, context_info = _entry
        context_dict[context_name] = context_info
    return context_dict


def process_none_entities(entity_name, parsed_entities):
    entity_name = entity_name.strip()

    ent_matches = [
        entity
        for entity in parsed_entities
        if entity is not None and entity[0] == entity_name  # entity[0] = _name
    ]

    if not ent_matches:
        return None
    elif len(ent_matches) > 1:
        print(
            f"Error: found multiple potential matches for {entity_name}: {ent_matches}. Picked none."
        )
        return None

    print(f"Found one match for {entity_name}: {ent_matches}. Picked.")
    return entity_name, ent_matches[0][1]


def parse_relationship(entry, parsed_entities):
    entry = entry.split(" || ", 1)
    if len(entry) == 1:
        relationship_str = entry[0].strip()
        context_str = None
    else:
        # print(entry)
        relationship_str, context_str = entry

    relationship_triple = re.split(r"\s*--([A-Z_]+)--\s*", relationship_str)
    if len(relationship_triple) != 3:
        return None

    head, relationship_type, tail = relationship_triple
    # print(relationship_triple)

    parsed_head = parse_entity(head)
    if parsed_head is None:
        parsed_head = process_none_entities(head, parsed_entities)

    parsed_tail = parse_entity(tail)
    if parsed_tail is None:
        parsed_tail = process_none_entities(tail, parsed_entities)

    if parsed_head is None or parsed_tail is None:
        return None
    relationship_triple = parsed_head, relationship_type, parsed_tail

    return relationship_triple, parse_context(context_str)


def parse_custom_format(text_arr):
    if len(text_arr) < 2:
        print(f"No output! {text_arr}")
        return

    entities = []
    relationships = []
    entities_flag, relationships_flag = False, False

    for line in text_arr:
        line = line.strip()

        if line == "Entities:":
            entities_flag = True
        elif line in ("Relationships:", "Relations:"):
            relationships_flag = True
            entities_flag = False
        elif line != "":
            if entities_flag:
                entities.append(line)
            elif relationships_flag:
                relationships.append(line)

    parsed_entities = []
    for entity_str in entities:
        parsed_entities.append(parse_entity(entity_str))

    parsed_relationships = []
    for relationship_str in relationships:
        rel = parse_relationship(relationship_str, parsed_entities)
        if rel is not None:
            parsed_relationships.append(rel)

    return parsed_entities, parsed_relationships
