import sys, json
sys.path.insert(0, "/Users/tom/Dropbox/msc-ml/project/src/")

import helpers.preprocessing as preprocessing
import helpers.metrics as metrics

import matplotlib.pyplot as plt



with open('./logs'+'/out_eval_MALUUBA.json') as f:
    results = json.load(f)


gold_pred_f1=[]
ctxt_pred_f1=[]
for res in results:
    qpred,qgold,ctxt,answer = res
    gold_pred_f1.append(metrics.f1(qgold, qpred))
    ctxt_pred_f1.append(metrics.f1(ctxt, qpred))

plt.scatter(ctxt_pred_f1, gold_pred_f1)
plt.show()
