from metrics import compute_bleu


def bleu(gold, prediction):
    return compute_bleu(gold, prediction)
