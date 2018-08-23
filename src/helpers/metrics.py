import string, re

from helpers.bleu import compute_bleu
from helpers.preprocessing import tokenise

from collections import Counter

# takes a single untokenised string as input
def bleu(gold, prediction, order=4):
    return compute_bleu([[tokenise(gold,asbytes=False)]], [tokenise(prediction,asbytes=False)], smooth=False, max_order=order)[0]

# takes a list of untokenized strings as inputs
def bleu_corpus(golds, preds, order=4):
    return compute_bleu([[tokenise(gold,asbytes=False)] for gold in golds], [tokenise(pred,asbytes=False) for pred in preds], smooth=False, max_order=order)[0]

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

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
