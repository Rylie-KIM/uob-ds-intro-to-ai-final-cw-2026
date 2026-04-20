def exact_sentence_match(true_label, pred_label):
    return int(str(true_label).strip().lower() == str(pred_label).strip().lower())


def attribute_overlap(parsed_true, parsed_pred):
    keys = ["size1", "colour1", "object1", "relation", "size2", "colour2", "object2"]
    matches = sum(1 for k in keys if parsed_true.get(k) == parsed_pred.get(k))
    return matches, matches / len(keys)


def relation_match(parsed_true, parsed_pred):
    return int(parsed_true.get("relation") == parsed_pred.get("relation"))


def object_pair_match(parsed_true, parsed_pred):
    return int(
        parsed_true.get("object1") == parsed_pred.get("object1")
        and parsed_true.get("object2") == parsed_pred.get("object2")
    )


def full_structure_match(parsed_true, parsed_pred):
    keys = ["size1", "colour1", "object1", "relation", "size2", "colour2", "object2"]
    return int(all(parsed_true.get(k) == parsed_pred.get(k) for k in keys))