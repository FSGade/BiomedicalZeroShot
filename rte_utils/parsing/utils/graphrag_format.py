from __future__ import annotations

import html
import re

DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"


def graphrag_clean_str(s):
    s = html.unescape(s.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    s = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", s)
    # if s and (s[0] == "<" and s[-1] == ">"):
    #    s = s[1:-1]
    s = s.lstrip("<").rstrip(">")
    return s


def graphrag_infer_relationship_entity_types(entities, relations):
    return entities, relations


def parse_graphrag_format(response):
    response = response.strip()

    completion_delimiter = DEFAULT_COMPLETION_DELIMITER
    if not response.endswith(completion_delimiter):
        return None, None

    record_delimiter = DEFAULT_RECORD_DELIMITER
    tuple_delimiter = DEFAULT_TUPLE_DELIMITER

    response = graphrag_clean_str(response.strip(completion_delimiter))
    records = [r.strip() for r in response.split(record_delimiter)]

    entities_dict = dict()
    entities_dict_case_insensitive = dict()

    relationships = []

    for record in records:
        record = re.sub(r"^\(|\)$", "", record.strip())
        record_attributes = record.split(tuple_delimiter)

        if record_attributes[0] == '"entity"' and len(record_attributes) >= 4:
            # add this record as a node in the G
            entity_name = graphrag_clean_str(record_attributes[1])
            entity_type = graphrag_clean_str(record_attributes[2])
            entity_description = graphrag_clean_str(record_attributes[3])

            entities_dict[entity_name] = entity_type
            entities_dict_case_insensitive[entity_name.lower()] = entity_type

        if (
            record_attributes[0] == '"relationship"'
            and len(record_attributes) >= 6
        ):
            # add this record as edge
            source = graphrag_clean_str(record_attributes[1])
            target = graphrag_clean_str(record_attributes[2])
            edge_type = graphrag_clean_str(record_attributes[3])
            edge_description = graphrag_clean_str(record_attributes[4])

            try:
                weight = float(record_attributes[-1])
            except ValueError:
                weight = 1.0

            context = {"description": edge_description, "weight": weight}

            relationships.append(
                (
                    (
                        (source, entities_dict.get(source, "unknown")),
                        edge_type,
                        (target, entities_dict.get(target, "unknown")),
                    ),
                    context,
                )
            )

    entities = list(entities_dict.items())

    # entities, relationships = graphrag_infer_relationship_entity_types(entities, relationships)

    return entities, relationships
