from helpers.bleu import compute_bleu

from collections import Counter


def bleu(gold, prediction):
    return compute_bleu([[gold.split()]], [prediction.split()])[0]


def f1(gold, prediction):
    prediction_tokens = prediction.split()
    ground_truth_tokens = gold.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
