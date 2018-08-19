import sys
from time import time
sys.path.insert(0, "/Users/tom/Dropbox/msc-ml/project/src/")

from collections import Counter

from helpers import loader, preprocessing

# from qa.qanet.prepro import word_tokenize

import string
import matplotlib.pyplot as plt
import numpy as np

squad =  loader.load_squad_triples('./data/',False,v2=False)#[9654:9655]
squad_dev =  loader.load_squad_triples('./data/',True,v2=False)#[9654:9655]


# glove_vocab = set(loader.get_glove_vocab('./data/', size=1e12, d=200).keys())
# glove_short = list(loader.get_glove_vocab('./data/', size=2000, d=200).keys())[4:]

squad_vocab =set()
squad_count = Counter()

start = time()
max_context_len=0
max_pos = None
debugstr = ""

c_lens=[]
q_lens=[]
for i,triple in enumerate(squad):
    # filtered,new_pos = preprocessing.filter_context(triple[0], triple[3], 1, 100)

    c_toks=  preprocessing.tokenise(triple[0], asbytes=False)
    q_toks=  preprocessing.tokenise(triple[1], asbytes=False)
    # context_set = sorted(set(c_toks))
    # context_set = c_toks
    # if len(context_set) > max_context_len:
    #     max_context_len = len(context_set)
    #     # max_pos = new_pos
    #     debugstr = triple[1]
    #     ix=i
    squad_count.update(c_toks)
    squad_count.update(q_toks)
    # squad_vocab |= set(c_toks)

    c_lens.append(len(c_toks))
    q_lens.append(len(q_toks))
end = time()
print(end-start)

print(np.sum(c_lens)+np.sum(q_lens))

# print("richard in glove", ("richard" in glove_short))
# print("doctor in glove", ("doctor" in glove_short))
# print(max_context_len," @ ",ix)
# print(squad[ix])
# print(debugstr)
#
# print(debugstr[max_pos:max_pos+10])
# print(len(preprocessing.tokenise(debugstr, asbytes=False)))

# print(len(squad_vocab))
# print(len(glove_vocab))
# print(len(squad_vocab-glove_vocab))
#
# print(squad_count.most_common(20))
# print(squad_count.most_common()[-20:])
#
top_words,top_n=zip(*squad_count.most_common(2000))

print([(i,w) for i,w in enumerate(top_words)])

plt.title('Frequency of words in SQuAD questions')
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.bar([x for x in range(2000)], [y for y in top_n], 1 ,log=True)
# plt.bar([x for x in range(2000)], [squad_count[y] for y in glove_short], 1 ,log=True)
# plt.savefig("/users/Tom/Dropbox/msc-ml/project-report/figures/squad_freq.pdf", format="pdf")
plt.show()
exit()

plt.title('Length of contexts against questions in SQuAD ')
plt.xlabel('Context Length')
plt.ylabel('Question Length')
heatmap, xedges, yedges = np.histogram2d(c_lens, q_lens, bins=(np.max(c_lens),np.max(q_lens)))
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
# plt.scatter(c_lens, q_lens, s=2)
plt.imshow(heatmap, extent=extent, origin="lower", aspect="auto")
# plt.bar([x for x in range(2000)], [squad_count[y] for y in glove_short], 1 ,log=True)
plt.savefig("/users/Tom/Dropbox/Apps/Overleaf/Question Generation/figures/squad_lengths.pdf", format="pdf")
print(np.corrcoef(c_lens,q_lens))
plt.show()
exit()

q_words=["who","when","what","why","how many","which","where","other"]
counts = {k:0 for k in q_words}
counts_dev = {k:0 for k in q_words}

for i,triple in enumerate(squad):
    qgold = triple[1]

    triggered = False
    for q in q_words:
        if q != "other" and q in qgold.lower():
            counts[q] += 1
            triggered=True
    if not triggered:
        counts["other"] += 1
for i,triple in enumerate(squad_dev):
    qgold = triple[1]

    triggered = False
    for q in q_words:
        if q != "other" and q in qgold.lower():
            counts_dev[q] += 1
            triggered=True
    if not triggered:
        counts_dev["other"] += 1

plt.title('Distribution of "wh" words in SQuAD questions')
# plt.xlabel('Interrogative')
plt.ylabel('Fraction')

bar_width=0.3

# plt.bar([x for x in range(len(q_words))], [np.mean(scores[q]) for q in q_words], tick_label=q_words)
plt.bar([x for x in range(len(q_words))], [counts[q]/len(squad) for q in q_words], tick_label=q_words, width=bar_width, label="SQuAD training set")
plt.bar([x+bar_width for x in range(len(q_words))], [counts_dev[q]/len(squad_dev) for q in q_words], width=bar_width, label="SQuAD dev set")
plt.legend()
plt.savefig("/users/Tom/Dropbox/msc-ml/project-report/figures/squad_wh_count.pdf", format="pdf")

plt.show()


# min_pos = 99999999
# num_out=0
# out_str=""
# for i,triple in enumerate(squad):
#     filt, filt_pos = preprocessing.filter_context(triple[0], triple[3], 1, 100)
#     out_str += filt +"\n"
#     # if "westwood one will carry the game throughout north america" in triple[0].lower():
#     #     # tokens = preprocessing.tokenise(triple[0], asbytes=False)
#     #     # tok_pos = preprocessing.char_pos_to_word(triple[0].encode(), tokens, triple[3])
#     #     # print(tokens[tok_pos])
#     #     # print(tok_pos)
#     #     # print(tokens)
#     #     # print(i,">>>"+triple[0][triple[3]:])
#     #     print(i,triple[1], triple[2], triple[3])
#     #     print(triple[0])
#     #     filt, filt_pos = preprocessing.filter_context(triple[0], triple[3], 1, 100)
#     #     print(filt[filt_pos:filt_pos+10])
#     #     num_out +=1
#     #     if num_out >5:
#     #         exit()
# with open('openie_dev.txt', 'w') as fp:
#     fp.write(out_str)
