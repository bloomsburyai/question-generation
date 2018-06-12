import sys
sys.path.insert(0, "/Users/tom/Dropbox/msc-ml/project/src/")


from helpers import loader, preprocessing

import string

squad =  loader.load_squad_triples('./data/',False)


# max_context_len=0
# max_pos = None
# debugstr = ""
# for i,triple in enumerate(squad):
#     c_toks=  preprocessing.tokenise(triple[0], asbytes=False)
#     if len(c_toks) > max_context_len:
#         max_context_len = len(c_toks)
#         max_pos = i
#         debugstr = triple[0]
# print(max_context_len," @ ",i)
# print(len(preprocessing.tokenise(debugstr, asbytes=False)))


min_pos = 99999999
for i,triple in enumerate(squad):
    if "What is the name of the song that Jessica" in triple[1]:
        tokens = preprocessing.tokenise(triple[0], asbytes=False)
        tok_pos = preprocessing.char_pos_to_word(triple[0].encode(), tokens, triple[3])
        print(tokens[tok_pos])
        print(tok_pos)
        print(tokens)
        print(i,">>>"+triple[0][triple[3]:])
        exit()
