# Benchmarking utils functions
from __future__ import annotations

import sys
from pprint import pprint
from typing import Literal

import numpy as np


def dict_by_type(data, data_type):
    _dict_by_type = {}

    if data_type == "e":
        for entity in data:
            entity_name, entity_type = entity
            if entity_type is None:
                continue
            if entity_type not in _dict_by_type:
                _dict_by_type[entity_type] = []
            _dict_by_type[entity_type].append(entity_name)
    else:
        for relationship in data:
            (
                ((head_name, head_type), rel, (tail_name, tail_type)),
                context,
            ) = relationship
            if head_type is None or tail_type is None:
                continue
            if rel not in _dict_by_type:
                _dict_by_type[rel] = []
            _dict_by_type[rel].append(
                ((head_name, head_type), (tail_name, tail_type))
            )

    return _dict_by_type


def remove_types_from_output(outputs):
    outputs_types_removed = []
    for output in outputs:
        output["entities"] = [
            (e[0], "") for e in output["entities"] if e is not None
        ]
        output["relations"] = [
            (((r[0][0][0], ""), "", (r[0][2][0], "")), r[1])
            for r in output["relations"]
            if r is not None
        ]
        outputs_types_removed.append(output)
    return outputs_types_removed


def sort_relations(outputs):
    outputs_sort_relations = []
    for output in outputs:
        output["relations"] = [
            r if (r[0][0][0] < r[0][2][0]) else (r[0][::-1], r[1])
            for r in output["relations"]
        ]
        outputs_sort_relations.append(output)
    return outputs_sort_relations


def to_tuples(outputs):
    outputs_lower = []
    for output in outputs:
        output["entities"] = [
            (e[0], e[1].lower())
            for e in output["entities"]
            if e is not None
        ]
        output["relations"] = [
            (
                (
                    (r[0][0][0], r[0][0][1].lower()),
                    r[0][1].lower(),
                    (r[0][2][0], r[0][2][1].lower()),
                ),
                r[1],
            )
            for r in output["relations"]
            if r is not None
        ]
        outputs_lower.append(output)
    return outputs_lower

def output_to_lower(outputs):
    outputs_lower = []
    for output in outputs:
        output["entities"] = [
            (e[0].lower(), e[1].lower())
            for e in output["entities"]
            if e is not None
        ]
        output["relations"] = [
            (
                (
                    (r[0][0][0].lower(), r[0][0][1].lower()),
                    r[0][1].lower(),
                    (r[0][2][0].lower(), r[0][2][1].lower()),
                ),
                r[1],
            )
            for r in output["relations"]
            if r is not None
        ]
        outputs_lower.append(output)
    return outputs_lower


def remove_context(outputs):
    outputs_no_context = []
    for output in outputs:
        if output["relations"]:
            output["relations"] = [
                (
                    (
                        (r[0][0][0].lower(), r[0][0][1].lower()),
                        r[0][1],
                        (r[0][2][0].lower(), r[0][2][1].lower()),
                    ),
                    None,
                )
                for r in output["relations"]
                if r is not None
            ]
        else:
            output["relations"] = []

        if output["entities"] is None:
            output["entities"] = []

        outputs_no_context.append(output)
    return outputs_no_context


def restrict_to_schema(outputs, ner_types, re_types):
    ner_types = set(ner_types)
    re_types = set(re_types)

    outputs_restrict = []
    for output in outputs:
        output["entities"] = [
            e for e in output["entities"] if e[1].lower() in ner_types
        ]
        temp_entity_set = set(output["entities"])
        output["relations"] = [
            r
            for r in output["relations"]
            if r[0][0] in temp_entity_set
            and r[0][2] in temp_entity_set
            and r[0][1].lower() in re_types
        ]
        outputs_restrict.append(output)
    return outputs_restrict


def _calculate_metrics_from_conf(confusion_matrix, beta=1.0):
    tp = confusion_matrix["tp"]
    fn = confusion_matrix["fn"]
    fp = confusion_matrix["fp"]

    precision, recall = 0, 0
    support = tp + fn

    if tp + fp:
        precision = tp / (tp + fp)
    if support:
        recall = tp / support

    if (precision) and (recall) and (precision + recall):
        f_score = (
            (1 + beta * beta)
            * precision
            * recall
            / (beta * beta * precision + recall)
        )
    else:
        f_score = 0

    return f_score, precision, recall, support


def _calculate_metrics_from_muc(confusion_matrix, PAR_weight=0.5, beta=1.0):
    COR = confusion_matrix["COR"]
    INC = confusion_matrix["INC"]
    PAR = confusion_matrix["PAR"]
    MIS = confusion_matrix["MIS"]
    SPU = confusion_matrix["SPU"]

    precision, recall = 0, 0  # None, None

    POS = COR + INC + PAR + MIS
    ACT = COR + INC + PAR + SPU

    if ACT:
        precision = (COR + PAR_weight * PAR) / ACT
    if POS:
        recall = (COR + PAR_weight * PAR) / POS

    # if (
    #     (precision is not None)
    #     and (recall is not None)
    #     and (precision + recall)
    # ):
    if (precision) and (recall) and (precision + recall):
        f_score = (
            (1 + beta * beta)
            * precision
            * recall
            / (beta * beta * precision + recall)
        )
    else:
        f_score = 0  # None

    return f_score, precision, recall, POS


def _calculate_metrics(
    confusion_matrices_dict, muc=False, PAR_weight=0.5, beta=1.0
):
    metrics_dict = {
        "macro": None,
        "micro": None,
        "weighted": None,
        "by_type": dict(),
    }

    if muc:
        total_confusion_matrix = {
            "COR": 0,
            "INC": 0,
            "PAR": 0,
            "MIS": 0,
            "SPU": 0,
        }
    else:
        total_confusion_matrix = {"tp": 0, "fn": 0, "fp": 0}

    for entity_type, confusion_matrix in confusion_matrices_dict.items():
        if muc:
            total_confusion_matrix["COR"] += confusion_matrix["COR"]
            total_confusion_matrix["INC"] += confusion_matrix["INC"]
            total_confusion_matrix["PAR"] += confusion_matrix["PAR"]
            total_confusion_matrix["MIS"] += confusion_matrix["MIS"]
            total_confusion_matrix["SPU"] += confusion_matrix["SPU"]

            metrics_dict["by_type"][entity_type] = _calculate_metrics_from_muc(
                confusion_matrix, PAR_weight=PAR_weight, beta=beta
            )
        else:
            total_confusion_matrix["tp"] += confusion_matrix["tp"]
            total_confusion_matrix["fn"] += confusion_matrix["fn"]
            total_confusion_matrix["fp"] += confusion_matrix["fp"]

            metrics_dict["by_type"][entity_type] = (
                _calculate_metrics_from_conf(confusion_matrix, beta=beta)
            )

    if muc:
        metrics_dict["micro"] = _calculate_metrics_from_muc(
            total_confusion_matrix, PAR_weight=PAR_weight, beta=beta
        )
    else:
        metrics_dict["micro"] = _calculate_metrics_from_conf(
            total_confusion_matrix, beta=beta
        )

    avg_f_score, avg_precision, avg_recall, avg_support = [], [], [], []
    for entity_type, (f_score, precision, recall, support) in metrics_dict[
        "by_type"
    ].items():
        avg_f_score.append(f_score)
        avg_precision.append(precision)
        avg_recall.append(recall)
        avg_support.append(support)

    metrics_dict["macro"] = (
        np.average(avg_f_score),
        np.average(avg_precision),
        np.average(avg_recall),
        np.sum(avg_support),
    )

    if np.sum(avg_support):
        metrics_dict["weighted"] = (
            np.average(avg_f_score, weights=avg_support),
            np.average(avg_precision, weights=avg_support),
            np.average(avg_recall, weights=avg_support),
            np.sum(avg_support),
        )
    else:
        metrics_dict["weighted"] = (None, None, None, None)

    pprint(metrics_dict)
    return metrics_dict


# def _calculate_metrics_from_res(confusion_matrices_dict, PAR_weight=0.5, beta=1.0):
#     metrics_dict = {"macro": None,
#                     "micro": None,
#                     "weighted": None,
#                     "by_type": dict()}

#     total_confusion_matrix = {"COR": 0, "INC": 0, "PAR": 0, "MIS": 0, "SPU": 0}

#     for entity_type, confusion_matrix in confusion_matrices_dict.items():
#         total_confusion_matrix["COR"] += confusion_matrix["COR"]
#         total_confusion_matrix["INC"] += confusion_matrix["INC"]
#         total_confusion_matrix["PAR"] += confusion_matrix["PAR"]
#         total_confusion_matrix["MIS"] += confusion_matrix["MIS"]
#         total_confusion_matrix["SPU"] += confusion_matrix["SPU"]

#         metrics_dict["by_type"][entity_type] = _calculate_metrics_from_muc(confusion_matrix, PAR_weight=PAR_weight, beta=beta)


#     metrics_dict["micro"] = _calculate_metrics_from_muc(total_confusion_matrix, PAR_weight=PAR_weight, beta=beta)

#     avg_f_score, avg_precision, avg_recall, avg_support = [], [], [], []
#     for entity_type, (f_score, precision, recall, ACT) in metrics_dict["by_type"].items():
#         avg_f_score.append(f_score)
#         avg_precision.append(precision)
#         avg_recall.append(recall)
#         avg_support.append(ACT)

#     metrics_dict["macro"] = (np.average(avg_f_score), np.average(avg_precision), np.average(avg_recall), np.sum(avg_support))

#     if np.sum(avg_support):
#         metrics_dict["weighted"] = (np.average(avg_f_score, weights=avg_support),
#                                     np.average(avg_precision, weights=avg_support),
#                                     np.average(avg_recall, weights=avg_support),
#                                     np.sum(avg_support))
#     else:
#         metrics_dict["weighted"] = (None, None, None, None)

#     pprint(metrics_dict)
#     return metrics_dict


def calculate_metrics(
    pred_output,
    true_output,
    ignore_case: bool | None = False,
    ignore_directionality: bool | None = False,
    restrict_types: bool | None = False,
    ner_types: list[str] | None = None,
    re_types: list[str] | None = None,
    match_criteria: Literal[
        "strict", "exact", "partial", "partial_strict"
    ] = "strict",
    print_examples=0,
):
    assert len(pred_output) == len(
        true_output
    ), "Could not calculate metric as the outputs do not contain the same number of samples"

    pred_output = remove_context(pred_output)

    # Make sure the entity pairs are tuples
    pred_output = to_tuples(pred_output)
    true_output = to_tuples(true_output)

    ner_types = [t.lower() for t in ner_types]
    re_types = [t.lower() for t in re_types]

    # Ignore case (transform to lower case)
    if ignore_case:
        pred_output = output_to_lower(pred_output)
        true_output = output_to_lower(true_output)

    if restrict_types:
        pred_output = restrict_to_schema(pred_output, ner_types, re_types)
        true_output = restrict_to_schema(true_output, ner_types, re_types)

    if ignore_directionality:
        pred_output = sort_relations(pred_output)
        true_output = sort_relations(true_output)

    # Consider only the entity names (ignore types)
    if match_criteria in ("exact", "partial"):
        pred_output = remove_types_from_output(pred_output)
        true_output = remove_types_from_output(true_output)

    # Count matches
    ner_confusion_matrix_per_type = dict()
    re_confusion_matrix_per_type = dict()

    i = 0

    for pred_vals, true_vals in zip(pred_output, true_output):
        pred_entities, pred_relations = (
            pred_vals["entities"],
            pred_vals["relations"],
        )
        true_entities, true_relations = (
            true_vals["entities"],
            true_vals["relations"],
        )

        pred_entities_unique = set(pred_entities)
        true_entities_unique = set(true_entities)

        pred_entities_unique_dict = {e: [] for e in pred_entities}
        true_entities_unique_dict = {e: [] for e in true_entities}

        if match_criteria in ("strict", "exact"):
            pred_entities = dict_by_type(pred_entities, "e")
            true_entities = dict_by_type(true_entities, "e")

            for entity_type in set(pred_entities.keys()) | set(
                true_entities.keys()
            ):  # union
                pred_entities_unique = set(pred_entities.get(entity_type, ""))
                true_entities_unique = set(true_entities.get(entity_type, ""))

                # print(entity_type)
                # print(pred_entities_unique)
                # print(true_entities_unique)
                if entity_type not in ner_confusion_matrix_per_type:
                    ner_confusion_matrix_per_type[entity_type] = {
                        "tp": 0,
                        "fp": 0,
                        "fn": 0,
                    }

                # print(ner_confusion_matrix_per_type[entity_type])
                # print()

                ner_confusion_matrix_per_type[entity_type]["tp"] += len(
                    pred_entities_unique & true_entities_unique
                )
                ner_confusion_matrix_per_type[entity_type]["fp"] += len(
                    pred_entities_unique - true_entities_unique
                )
                ner_confusion_matrix_per_type[entity_type]["fn"] += len(
                    true_entities_unique - pred_entities_unique
                )
        else:
            # res = {"COR": 0, "INC": 0, "PAR": 0, "MIS": 0, "SPU": 0}
            res = {"COR": [], "INC": [], "PAR": [], "MIS": [], "SPU": []}

            partial_pred_to_gs_dict = dict()

            # if not pred_entities_unique:
            #     res["MIS"] = len(true_entities_unique)
            # elif not true_entities_unique:
            #     res["SPU"] = len(pred_entities_unique)
            if not pred_entities_unique:
                res["MIS"] = [e[1] for e in true_entities_unique]
            elif not true_entities_unique:
                res["SPU"] = [e[1] for e in pred_entities_unique]
            else:
                for e in pred_entities_unique & true_entities_unique:
                    pred_entities_unique_dict[e].append("COR")
                    true_entities_unique_dict[e].append("COR")

                for e in pred_entities_unique_dict.keys():
                    if "COR" not in pred_entities_unique_dict[e]:
                        for gs_e in true_entities_unique:
                            if gs_e[1] != e[1]:
                                continue
                            if (
                                gs_e[0].startswith(e[0])
                                or gs_e[0].endswith(e[0])
                                or e[0].startswith(gs_e[0])
                                or e[0].endswith(gs_e[0])
                            ):
                                pred_entities_unique_dict[e].append("PAR")
                                true_entities_unique_dict[gs_e].append("PAR")
                                partial_pred_to_gs_dict[e] = gs_e
                                continue

                for e, e_res in pred_entities_unique_dict.items():
                    if not e_res:
                        res["SPU"].append(e[1])
                        # res["SPU"] += 1

                for e, e_res in true_entities_unique_dict.items():
                    if not e_res:
                        res["MIS"].append(e[1])
                        # res["MIS"] += 1
                    else:
                        if "COR" in e_res:
                            res["COR"].append(e[1])
                        else:
                            res["PAR"].append(e[1])
                        # for _res in e_res:
                        #     res[_res].append(e[1])
                        # res[_res] += 1

            for e_res, entity_types in res.items():
                for entity_type in entity_types:
                    if entity_type not in ner_confusion_matrix_per_type:
                        ner_confusion_matrix_per_type[entity_type] = {
                            "COR": 0,
                            "INC": 0,
                            "PAR": 0,
                            "MIS": 0,
                            "SPU": 0,
                        }
                    else:
                        ner_confusion_matrix_per_type[entity_type][e_res] += 1

        ## RELATIONS
        pred_relations_unique = set(pred_relations)
        true_relations_unique = set(true_relations)

        if match_criteria in ("strict", "exact"):
            pred_relations = dict_by_type(pred_relations, "r")
            true_relations = dict_by_type(true_relations, "r")

            for relation_type in set(pred_relations.keys()) | set(
                true_relations.keys()
            ):  # union
                pred_relations_unique = set(
                    pred_relations.get(relation_type, "")
                )
                true_relations_unique = set(
                    true_relations.get(relation_type, "")
                )

                if relation_type not in re_confusion_matrix_per_type:
                    re_confusion_matrix_per_type[relation_type] = {
                        "tp": 0,
                        "fn": 0,
                        "fp": 0,
                    }

                re_confusion_matrix_per_type[relation_type]["tp"] += len(
                    pred_relations_unique & true_relations_unique
                )
                re_confusion_matrix_per_type[relation_type]["fp"] += len(
                    pred_relations_unique - true_relations_unique
                )
                re_confusion_matrix_per_type[relation_type]["fn"] += len(
                    true_relations_unique - pred_relations_unique
                )
        else:
            # res = {"COR": 0, "INC": 0, "PAR": 0, "MIS": 0, "SPU": 0}
            res = {"COR": [], "INC": [], "PAR": [], "MIS": [], "SPU": []}

            if not pred_relations_unique:
                res["MIS"] = [r[0][1] for r in true_relations_unique]
                # res["MIS"] = len(true_relations_unique)
            elif not true_relations_unique:
                res["SPU"] = [r[0][1] for r in pred_relations_unique]
                # res["SPU"] = len(pred_relations_unique)
            else:
                pred_relations_unique_dict = {r: [] for r in pred_relations}
                true_relations_unique_dict = {r: [] for r in true_relations}

                overlap = pred_relations_unique & true_relations_unique

                for r in overlap:
                    pred_relations_unique_dict[r].append("COR")
                    true_relations_unique_dict[r].append("COR")

                partial_pred_relations_unique_dict = {
                    r: (
                        (
                            partial_pred_to_gs_dict.get(r[0][0], r[0][0]),
                            r[0][1],
                            partial_pred_to_gs_dict.get(r[0][2], r[0][2]),
                        ),
                        r[1],
                    )
                    for r in pred_relations
                    if r not in overlap
                }

                for r in (
                    set(partial_pred_relations_unique_dict.values())
                    & true_relations_unique
                ):
                    for key, val in partial_pred_relations_unique_dict.items():
                        if r == val:
                            pred_relations_unique_dict[key].append("PAR")
                    true_relations_unique_dict[r].append("PAR")

                for r, r_res in pred_relations_unique_dict.items():
                    if not r_res:
                        res["SPU"].append(r[0][1])
                        # res["SPU"] += 1

                for r, r_res in true_relations_unique_dict.items():
                    if not r_res:
                        res["MIS"].append(r[0][1])
                        # res["MIS"] += 1
                    else:
                        for _res in r_res:
                            res[_res].append(r[0][1])
                            # res[_res] += 1

            for r_res, relation_types in res.items():
                for relation_type in relation_types:
                    if relation_type not in re_confusion_matrix_per_type:
                        re_confusion_matrix_per_type[relation_type] = {
                            "COR": 0,
                            "INC": 0,
                            "PAR": 0,
                            "MIS": 0,
                            "SPU": 0,
                        }
                    else:
                        re_confusion_matrix_per_type[relation_type][r_res] += 1

        if i < print_examples:
            print(f"Example {i + 1}")
            print("Predicted entities")
            pprint(pred_entities_unique_dict)
            print("GS entities:")
            pprint(true_entities_unique_dict)
            if match_criteria in ("strict", "exact"):
                pprint(ner_confusion_matrix_per_type)
            else:
                pprint(ner_confusion_matrix_per_type)
            print()

            print("Predicted relations")
            pprint(pred_relations_unique)
            print("GS relations:")
            pprint(true_relations_unique)
            if match_criteria in ("strict", "exact"):
                pprint(re_confusion_matrix_per_type)
            else:
                pprint(re_confusion_matrix_per_type)
            print()
            i += 1

    # Calculate NER and RE metrics
    if match_criteria in ("strict", "exact"):
        ner_metrics = _calculate_metrics(
            ner_confusion_matrix_per_type, beta=1.0
        )
        re_metrics = _calculate_metrics(re_confusion_matrix_per_type, beta=1.0)
    else:
        ner_metrics = _calculate_metrics(
            ner_confusion_matrix_per_type, muc=True, beta=1.0
        )
        re_metrics = _calculate_metrics(
            re_confusion_matrix_per_type, muc=True, beta=1.0
        )

    return ner_metrics, re_metrics
