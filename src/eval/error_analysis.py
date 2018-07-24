import sys, json
sys.path.insert(0, "/Users/tom/Dropbox/msc-ml/project/src/")

import helpers.preprocessing as preprocessing
import helpers.metrics as metrics

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools



with open('./logs'+'/out_eval_MALUUBA_RL_QA.json') as f:
    results = json.load(f)

# q_words=["who","when","what","why","how many","which","where","other"]
q_words=["who","when","what","how","which","where","other"]
scores = {k:[] for k in q_words}
counts = {k:0 for k in q_words}
word_gold=["other" for i in range(len(results))]
word_pred=["other" for i in range(len(results))]

gold_pred_f1=[]
x=[]
for i,res in enumerate(results):
    qpred,qgold,ctxt,answer = res
    gold_pred_f1.append(metrics.bleu(qgold, qpred))
    # x.append(metrics.f1(ctxt, qpred))
    x.append(len(preprocessing.tokenise(qgold, asbytes=False)))

    triggered=False
    for q in q_words:
        if q != "other" and q in qpred.lower():
            scores[q].append(metrics.bleu(qgold, qpred))
            counts[q] += 1
            word_pred[i]=q

        if q != "other" and q in qgold.lower():
            scores[q].append(metrics.bleu(qgold, qpred))
            counts[q] += 1
            word_gold[i]=q
            triggered=True
    if not triggered:
        scores["other"].append(metrics.bleu(qgold, qpred))
        counts["other"] += 1
        word_gold[i]="other"
        print(qpred, qgold)

mat = confusion_matrix(word_gold, word_pred)
print(mat)
plt.imshow(mat, cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(q_words))
plt.xticks(tick_marks, q_words, rotation=45)
plt.yticks(tick_marks, q_words)
fmt = 'd'
thresh = mat.max() / 2.
for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
    plt.text(j, i, format(mat[i, j], fmt),
             horizontalalignment="center",
             color="white" if mat[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
print(counts)
plt.show()
exit()


plt.title('Distribution of "wh" words in SQuAD questions')
plt.xlabel('Interrogative')
plt.ylabel('Count')

# plt.bar([x for x in range(len(q_words))], [np.mean(scores[q]) for q in q_words], tick_label=q_words)
plt.bar([x for x in range(len(q_words))], [counts[q] for q in q_words], tick_label=q_words)
plt.savefig("/users/Tom/Dropbox/msc-ml/project-report/figures/squad_wh_count.pdf", format="pdf")

plt.show()
