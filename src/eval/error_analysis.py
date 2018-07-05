import sys, json
sys.path.insert(0, "/Users/tom/Dropbox/msc-ml/project/src/")

import helpers.preprocessing as preprocessing
import helpers.metrics as metrics

import matplotlib.pyplot as plt



with open('./logs'+'/out_eval_MALUUBA.json') as f:
    results = json.load(f)


gold_pred_f1=[]
x=[]
for res in results:
    qpred,qgold,ctxt,answer = res
    gold_pred_f1.append(metrics.bleu(qgold, qpred))
    # x.append(metrics.f1(ctxt, qpred))
    x.append(len(preprocessing.tokenise(qgold, asbytes=False)))

plt.scatter(x, gold_pred_f1)
plt.show()
