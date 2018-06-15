import sys
from time import time
sys.path.insert(0, "/Users/tom/Dropbox/msc-ml/project/src/")


from helpers import loader, preprocessing

import string

squad =  loader.load_squad_triples('./data/',False)

start = time()
max_context_len=0
max_pos = None
debugstr = ""
for i,triple in enumerate(squad):
    c_toks=  preprocessing.tokenise(triple[0], asbytes=False)
    context_set = sorted(set(c_toks))
    # context_set = c_toks
    if len(context_set) > max_context_len:
        max_context_len = len(context_set)
        max_pos = i
        debugstr = triple[0]
end = time()
print(end-start)
print(max_context_len," @ ",i)
print(len(set(preprocessing.tokenise(debugstr, asbytes=False))))


# min_pos = 99999999
# for i,triple in enumerate(squad):
#     if triple[1].find("Halloween") >0:
#         # tokens = preprocessing.tokenise(triple[0], asbytes=False)
#         # tok_pos = preprocessing.char_pos_to_word(triple[0].encode(), tokens, triple[3])
#         # print(tokens[tok_pos])
#         # print(tok_pos)
#         # print(tokens)
#         # print(i,">>>"+triple[0][triple[3]:])
#         print(triple)
#         # exit()
