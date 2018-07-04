import sys
from time import time
sys.path.insert(0, "/Users/tom/Dropbox/msc-ml/project/src/")


from helpers import loader, preprocessing

from qa.qanet.prepro import word_tokenize

import string

squad =  loader.load_squad_triples('./data/',False,v2=False)#[9654:9655]

start = time()
max_context_len=0
max_pos = None
debugstr = ""
for i,triple in enumerate(squad):
    # filtered,new_pos = preprocessing.filter_context(triple[0], triple[3], 1, 100)

    c_toks=  word_tokenize(triple[1])
    context_set = sorted(set(c_toks))
    # context_set = c_toks
    if len(context_set) > max_context_len:
        max_context_len = len(context_set)
        # max_pos = new_pos
        debugstr = triple[1]
        ix=i
end = time()
print(end-start)
print(max_context_len," @ ",ix)
print(squad[ix])
print(debugstr)
#
# print(debugstr[max_pos:max_pos+10])
# print(len(preprocessing.tokenise(debugstr, asbytes=False)))


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
