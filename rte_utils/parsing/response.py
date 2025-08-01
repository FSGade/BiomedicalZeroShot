from __future__ import annotations

import json
import sys
from typing import Any

from langchain_core.messages.ai import AIMessage

from .utils import parse_custom_format
from .utils import parse_graphrag_format
from .utils import parse_phi3_graph_format
from .utils import parse_scilitllm_format
from .utils import parse_triplex_format


def parse_response(
    response: list[AIMessage | dict[str, Any] | str],
    parse_format: str = "default",
) -> list[dict[str, list]]:
    """_summary_

    Args:
        response (list[AIMessage  |  dict[str, Any]  |  str]): _description_
        parse_format (str, optional): _description_. Defaults to "default".

    Returns:
        list[dict[str, list]]: _description_
    """
    if not response:
        return []

    if isinstance(response[0], AIMessage):
        response_strs = [o.content for o in response]
    elif isinstance(response[0], dict):
        response_strs = [o["response"] for o in response]
    else:
        response_strs = response

    parsed_response = []

    for entry in response_strs:
        if parse_format == "default":
            entry_arr = entry.lstrip("\n").split("\n")
            if entry_arr:
                parsed_entry = parse_custom_format(entry_arr)
            else:
                parsed_entry = None

        elif parse_format == "deepseek":
            entry_think = entry.split("</think>")
            if len(entry_think) != 2:
                parsed_entry = None
            else:
                entry_arr = (
                    entry_think[1]
                    .lstrip("\n")
                    .rstrip("<｜end▁of▁sentence｜>")
                    .split("\n")
                )
                if entry_arr:
                    parsed_entry = parse_custom_format(entry_arr)
                else:
                    parsed_entry = None

        elif parse_format == "triplex":
            parsed_entry = parse_triplex_format(entry)

        elif parse_format == "scilitllm":
            parsed_entry = parse_scilitllm_format(entry)

        elif parse_format == "phi3_graph":
            parsed_entry = parse_phi3_graph_format(entry)

        elif parse_format == "json":
            try:
                entry_dict = json.loads(entry)
            except json.JSONDecodeError:
                print(
                    f"Could not load json entry {entry} from response. Adding empty output"
                )
                parsed_response.append({"entities": [], "relations": []})
                continue
            entry_dict["relations"] = [
                (r, None) for r in entry_dict["relations"]
            ]
            parsed_response.append(entry_dict)
            continue

        elif parse_format == "graphrag":
            parsed_entry = parse_graphrag_format(entry)

        else:
            print(
                f"ERROR: parse_format '{parse_format}' not implemented. Exiting"
            )
            sys.exit(1)

        if parsed_entry is not None:
            parsed_entities, parsed_relationships = parsed_entry
        else:
            parsed_entities, parsed_relationships = [], []

        parsed_response.append(
            {"entities": parsed_entities, "relations": parsed_relationships}
        )

    return parsed_response
