import sys
sys.path.insert(0, "/Users/tom/Dropbox/msc-ml/project/src/")


from helpers import loader, preprocessing

import string

squad =  loader.load_squad_triples('./data/',False)

def tokenise(text):

    for char in string.punctuation+'()-–':
        text = text.replace(char, ' '+char+' ')
    tokens = text.lower().split(' ')
    tokens = [w for w in tokens if w.strip() != '']
    # tokens = np.asarray(tokens)
    return tokens

max_context_len=0
max_pos = None
debugstr = ""
for i,triple in enumerate(squad):
    c_toks=  preprocessing.tokenise(triple[0], asbytes=False)
    if len(c_toks) > max_context_len:
        max_context_len = len(c_toks)
        max_pos = i
        debugstr = triple[0]
print(max_context_len," @ ",i)
print(len(preprocessing.tokenise(debugstr, asbytes=False)))


min_pos = 99999999
for i,triple in enumerate(squad):
    if triple[3] < min_pos:
        min_pos = triple[3]
print(min_pos)
