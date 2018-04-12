from helpers import loader, preprocessing

import string

squad =  loader.load_squad_triples('./data/',True)

def tokenise(text):

    for char in string.punctuation+'()-–':
        text = text.replace(char, ' '+char+' ')
    tokens = text.lower().split(' ')
    tokens = [w for w in tokens if w.strip() != '']
    # tokens = np.asarray(tokens)
    return tokens

max_context_len=0
for triple in squad:
    c_toks=  tokenise(triple[0])
    if len(c_toks) > max_context_len:
        max_context_len = len(c_toks)

print(max_context_len)
