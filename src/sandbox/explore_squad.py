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




from alignment.sequence import Sequence
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner

# Create sequences to be aligned.
a = Sequence(preprocessing.tokenise(squad[0][0], asbytes=False))
b = Sequence(preprocessing.tokenise(squad[0][1], asbytes=False))



# Create a vocabulary and encode the sequences.
v = Vocabulary()
aEncoded = v.encodeSequence(a)
bEncoded = v.encodeSequence(b)

print(a)
print(b)

# Create a scoring and align the sequences using global aligner.
scoring = SimpleScoring(2, -1)
aligner = GlobalSequenceAligner(scoring, -2)
score, encodeds = aligner.align(aEncoded, bEncoded, backtrace=True)

# Iterate over optimal alignments and print them.
for encoded in encodeds:
    alignment = v.decodeSequenceAlignment(encoded)
    print(encoded)
    print(alignment)
    print('Alignment score:', alignment.score)
    print('Percent identity:', alignment.percentIdentity())


exit()




glove_vocab = set(loader.get_glove_vocab('./data/', size=1e12, d=200).keys())
glove_short = list(loader.get_glove_vocab('./data/', size=2000, d=200).keys())[4:]

squad_vocab =set()
squad_count = Counter()

start = time()
max_context_len=0
max_pos = None
debugstr = ""
for i,triple in enumerate(squad):
    # filtered,new_pos = preprocessing.filter_context(triple[0], triple[3], 1, 100)

    c_toks=  preprocessing.tokenise(triple[1], asbytes=False)
    # context_set = sorted(set(c_toks))
    # context_set = c_toks
    # if len(context_set) > max_context_len:
    #     max_context_len = len(context_set)
    #     # max_pos = new_pos
    #     debugstr = triple[1]
    #     ix=i
    squad_count.update(c_toks)
    squad_vocab |= set(c_toks)
end = time()
print(end-start)
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
# _,top_n=zip(*squad_count.most_common(2000))
# plt.bar([x for x in range(2000)], [y for y in top_n], 1 ,log=True)
# # plt.bar([x for x in range(2000)], [squad_count[y] for y in glove_short], 1 ,log=True)
# plt.show()


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
