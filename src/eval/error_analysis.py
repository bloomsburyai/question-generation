import sys, json
sys.path.insert(0, "/Users/tom/Dropbox/msc-ml/project/src/")

import helpers.preprocessing as preprocessing
import helpers.metrics as metrics

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools



with open('./results'+'/unsplit/out_eval_MALUUBA-CROP-SMART-SET_train.json') as f:
    results = json.load(f)['results']

# q_words=["who","when","what","why","how many","which","where","other"]
q_words=["who","when","what","how","which","where","why","other"]
scores = {k:[] for k in q_words}
counts = {k:0 for k in q_words}

word_gold=["other" for i in range(len(results))]
word_pred=["other" for i in range(len(results))]

gold_pred_bleu=[]
gold_pred_f1=[]
nlls=[]
x=[]
for i,res in enumerate(results):
    qpred,qgold,ctxt,answer,a_pos = res['q_pred'], res['q_gold'], res['c'], res['a_text'], res['a_pos']
    gold_pred_bleu.append(metrics.bleu(qgold, qpred))
    nlls.append(res['metrics']['nll'])
    # x.append(metrics.f1(ctxt, qpred))
    x.append(len(preprocessing.tokenise(qgold, asbytes=False)))
    gold_pred_f1.append(metrics.f1(qgold, qpred))

    triggered=False
    for q in q_words:
        if q != "other" and q in qpred.lower():
            # scores[q].append(metrics.bleu(qgold, qpred))
            # scores[q].append(res['metrics']['qa'])
            # counts[q] += 1
            word_pred[i]=q

        if q != "other" and q in qgold.lower():
            counts[q] += 1
            word_gold[i]=q
            triggered=True
    if not triggered:
        # scores["other"].append(metrics.bleu(qgold, qpred))
        # scores['other'].append(res['metrics']['qa'])
        counts["other"] += 1
        word_gold[i]="other"
        # print(qpred, qgold)

# *********** CONFUSION MATRIX
# cm = confusion_matrix(word_gold, word_pred)
# mat = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# print(mat)
# plt.imshow(mat, cmap=plt.cm.Blues)
# plt.colorbar()
# tick_marks = np.arange(len(q_words))
# plt.xticks(tick_marks, q_words, rotation=45)
# plt.yticks(tick_marks, q_words)
# fmt = '.2f'
# thresh = mat.max() / 2.
# for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
#     plt.text(j, i, format(mat[i, j], fmt),
#              horizontalalignment="center",
#              color="white" if mat[i, j] > thresh else "black")
#
# # plt.tight_layout()
# plt.ylabel('Gold interrogative')
# plt.xlabel('Predicted interrogative')
# print(counts)
# plt.savefig("/users/Tom/Dropbox/Apps/Overleaf/Question Generation/figures/confusion_maluuba_crop.pdf", format="pdf")
# plt.show()
# exit()

# ************* Score violin plot
# plt.violinplot(nlls, points=60, widths=0.7, showmeans=True,
#                       showextrema=True, showmedians=False, bw_method=0.5)
# plt.title('Distribution of NLLs')
# plt.ylabel('NLL')
# plt.savefig("/users/Tom/Dropbox/Apps/Overleaf/Question Generation/figures/nll_violin.pdf", format="pdf")
# plt.show()
# exit()


# ************ Dist of interrogatives
plt.title('Distribution of "wh" words in SQuAD questions')
plt.xlabel('Interrogative')
plt.ylabel('Count')

# plt.bar([x for x in range(len(q_words))], [np.mean(scores[q]) for q in q_words], tick_label=q_words)
plt.bar([x for x in range(len(q_words))], [counts[q] for q in q_words], tick_label=q_words)
plt.savefig("/users/Tom/Dropbox/Apps/Overleaf/Question Generation/figures/squad_wh_count.pdf", format="pdf")

plt.show()
