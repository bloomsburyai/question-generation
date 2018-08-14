from helpers.bleu import compute_bleu
from helpers.preprocessing import tokenise

from collections import Counter
from nltk.translate.bleu_score import corpus_bleu

# takes a single untokenised string as input
def bleu(gold, prediction, order=4):
    # return compute_bleu([[gold.split()]], [prediction.split()], smooth=True, max_order=order)[0]
    return compute_bleu([[tokenise(gold,asbytes=False)]], [tokenise(prediction,asbytes=False)], smooth=False, max_order=order)[0]
    # return sentence_bleu([tokenise(gold,asbytes=False)], tokenise(prediction,asbytes=False))

# takes a list of untokenized strings as inputs
def bleu_corpus(golds, preds, order=4):
    weights = [1/order for i in range(order)]
    return corpus_bleu([[tokenise(gold,asbytes=False)] for gold in golds], [tokenise(pred,asbytes=False) for pred in preds], weights=weights)

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
