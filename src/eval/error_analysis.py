import sys, json
sys.path.insert(0, "/Users/tom/Dropbox/msc-ml/project/src/")

import helpers.preprocessing as preprocessing
import helpers.metrics as metrics

import matplotlib.pyplot as plt
import numpy as np



with open('./logs'+'/out_eval_MALUUBA.json') as f:
    results = json.load(f)

q_words=["who","when","what","why","how many","which"]
scores = {k:[] for k in q_words}

gold_pred_f1=[]
x=[]
for res in results:
    qpred,qgold,ctxt,answer = res
    gold_pred_f1.append(metrics.bleu(qgold, qpred))
    # x.append(metrics.f1(ctxt, qpred))
    x.append(len(preprocessing.tokenise(qgold, asbytes=False)))

    for q in q_words:
        if q in qgold.lower():
            scores[q].append(metrics.bleu(qgold, qpred))

# plt.scatter(x, gold_pred_f1)
# plt.show()

plt.bar([x for x in range(len(q_words))], [np.mean(scores[q]) for q in q_words], tick_label=q_words)
plt.show()
