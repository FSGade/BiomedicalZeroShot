from __future__ import annotations

import json


def parse_phi3_graph_format(response):
    response = response.strip()

    try:
        response_dict = json.loads(response)
    except json.decoder.JSONDecodeError:
        return [], []

    entities = [(e["id"], e.get("type", "")) for e in response_dict["nodes"]]
    entities_type_dict = {
        e["id"]: e.get("type", "") for e in response_dict["nodes"]
    }

    relationships = []
    for r in response_dict["edges"]:
        if "label" not in r or "to" not in r or "from" not in r:
            continue
        head = r["from"]
        tail = r["to"]
        rel = r["label"]

        head_type = entities_type_dict.get(head, "unknown")
        tail_type = entities_type_dict.get(head, "unknown")

        relationships.append(
            (((head, head_type), rel, (tail, tail_type)), None)
        )

    return entities, relationships
