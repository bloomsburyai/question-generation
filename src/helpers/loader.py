import json

import re
from collections import defaultdict

import numpy as np

SOS = '<Sent>'
EOS = '</Sent>'
PAD='<PAD>'
OOV='<OOV>'

def load_squad_dataset(path, dev=False):
    expected_version = '1.1'
    filename = 'train-v1.1.json' if not dev else 'dev-v1.1.json'
    with open(path+filename) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != expected_version):
            print('Expected SQuAD v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
        return(dataset)

def load_squad_triples(path, dev=False):
    raw_data = load_squad_dataset(path, dev)
    triples=[]
    for doc in raw_data:
        for para in doc['paragraphs']:
            for qa in para['qas']:
                # NOTE: this only takes the first answer per question! ToDo handle this more intelligently
                triples.append( (para['context'], qa['question'], qa['answers'][0]['text'], int(qa['answers'][0]['answer_start'])) )  #
    return triples

def get_vocab(corpus, vocab_size=2000):
    vocab = {PAD:0,OOV:1, SOS:2, EOS:3}

    for l in corpus:
        for w in l.lower().split():
            word_count[w] +=1
    vocab_list = sorted(word_count, key=word_count.__getitem__,reverse=True)[:min(vocab_size,len(word_count))]
    for w in vocab_list:
        vocab[w] = len(vocab)
    return vocab

def load_multiline(path, limit_length=32, vocab_size=5000):
    with open(path,'r') as fp:
        raw_data = fp.readlines()
    lines = [re.sub(r'([\,\?\!\.]+)',r' \1 ', line).lower() for line in raw_data]
    # lines = re.split('[\n]+',raw_data.lower())
    vocab = {PAD:0,OOV:1, SOS:2, EOS:3}
    word_count = defaultdict(float)
    ids=[]
    max_sent_len=0
    for l in lines:
        for w in l.split():
            word_count[w] +=1
    vocab_list = sorted(word_count, key=word_count.__getitem__,reverse=True)[:min(vocab_size,len(word_count))]
    for w in vocab_list:
        vocab[w] = len(vocab)
    for l in lines:
        # if l[0] == '<':
        #     print(l)
        # if len(l) ==0 or l[0] == '<':
        #     continue
        id_line=[vocab[SOS]]
        # id_line=[]
        # token_line = []
        for w in l.split():
            if len(id_line) > max_sent_len:
                max_sent_len = len(id_line)
            if len(id_line) >= limit_length-1:
                continue
            w = w.strip()
            if len(w) == 0:
                continue
            if w not in vocab.keys():
                # vocab[w] = len(vocab)
                w = '<OOV>'
            id_line.append(vocab[w])
        # if len(id_line) > 400:
        #     print(token_line)
        #     print(id_line)
        #     exit()
        id_line.append(vocab[EOS])

        # if len(token_line) >= limit_length:
        #     print(token_line)
        #     print(len(token_line))
        #     exit()
        ids.append(id_line)

    if limit_length:
        max_sent_len = min(max_sent_len+1, limit_length)
    id_arr = np.full([len(ids), max_sent_len], vocab['<PAD>'], dtype=np.int32)

    for i, sent in enumerate(ids):
        this_len = min(len(sent), max_sent_len)
        id_arr[i,  0:this_len] = (sent if len(sent) <= max_sent_len else sent[:max_sent_len])

    return id_arr, vocab

def get_vocab(corpus, vocab_size=1000):
    lines = [re.sub(r'([\,\?\!\.]+)',r' \1 ', line).lower() for line in corpus]
    # lines = re.split('[\n]+',raw_data.lower())
    vocab = {PAD:0,OOV:1, SOS:2, EOS:3}
    word_count = defaultdict(float)
    for l in lines:
        for w in l.split():
            word_count[w] +=1
    vocab_list = sorted(word_count, key=word_count.__getitem__,reverse=True)[:min(vocab_size,len(word_count))]
    for w in vocab_list:
        vocab[w] = len(vocab)
    return vocab

def get_line_ids(line, ref_line, vocab, limit_length):
    line_ids=[vocab[SOS]]

    for w in line:
        if len(line_ids) >= limit_length-1:
            break
        w = w.strip()
        if len(w) == 0:
            continue
        if w not in vocab.keys():
            if w in ref_line and ref_line.index(w) < limit_length+2:
                line_ids.append(len(vocab)+ref_line.index(w))
            else:
                line_ids.append(vocab[OOV])
        else:
            line_ids.append(vocab[w])

    line_ids.append(vocab[EOS])
    return line_ids

def load_multiline_aligned(path_src, path_tgt, limit_length=32, vocab_size=1000):
    with open(path_src,'r') as fp_src:
        raw_data_src = fp_src.readlines()
    with open(path_tgt,'r') as fp_tgt:
        raw_data_tgt = fp_tgt.readlines()
    vocab_src = get_vocab(raw_data_src, vocab_size=1000)
    vocab_tgt = get_vocab(raw_data_tgt, vocab_size=1000)

    assert len(raw_data_src) == len(raw_data_tgt)

    out_src=[]
    out_tgt=[]
    max_sent_len_src=0
    max_sent_len_tgt=0

    for l in range(len(raw_data_src)):
        line_src = re.sub(r'([\,\?\!\.]+)',r' \1 ', raw_data_src[l]).lower().split()
        line_tgt = re.sub(r'([\,\?\!\.]+)',r' \1 ', raw_data_tgt[l]).lower().split()
        if line_src == '' or line_tgt == '':
            continue

        line_ids_src = get_line_ids(line_src, line_tgt, vocab_src, limit_length)
        line_ids_tgt = get_line_ids(line_tgt, line_src, vocab_tgt, limit_length)

        if np.sum(line_ids_src) > vocab_src[SOS]+vocab_src[EOS] and np.sum(line_ids_tgt) > vocab_tgt[SOS]+vocab_tgt[EOS]:
            out_src.append(line_ids_src)
            out_tgt.append(line_ids_tgt)

            max_sent_len_src = max(max_sent_len_src, len(out_src))
            max_sent_len_tgt = max(max_sent_len_tgt, len(out_tgt))

    if limit_length:
        # max_sent_len = min(max(max_sent_len_src,max_sent_len_tgt)+1, limit_length)
        max_sent_len = limit_length

    id_arr_src = np.full([len(out_src), max_sent_len], vocab_src['<PAD>'], dtype=np.int32)
    id_arr_tgt = np.full([len(out_tgt), max_sent_len], vocab_tgt['<PAD>'], dtype=np.int32)

    for i, sent in enumerate(out_src):
        this_len = min(len(sent), max_sent_len)
        id_arr_src[i,  0:this_len] = (sent if len(sent) <= max_sent_len else sent[:max_sent_len])

    for i, sent in enumerate(out_tgt):
        this_len = min(len(sent), max_sent_len)
        id_arr_tgt[i,  0:this_len] = (sent if len(sent) <= max_sent_len else sent[:max_sent_len])

    return id_arr_src, id_arr_tgt, vocab_src, vocab_tgt

def load_glove(path, d=200):
    glove = {}
    with open(path+'glove.6B/glove.6B.'+str(d)+'d.txt') as fp:
        entries = fp.readlines()
    for row in entries:
        cols = row.strip().split(' ')
        if len(cols) < d+1:
            print(row)
        glove[cols[0]] = np.asarray(cols[1:], dtype=float)
    return glove

def get_embeddings(vocab, glove, D):
    rev_vocab = {v:k for k,v in vocab.items()}
    embeddings=[]

    # rand = np.random.normal(size=(len(vocab),D))
    # q,r = np.linalg.qr(rand)
    glorot_limit = np.sqrt(6 / (D + len(vocab)))

    # clunky, but guarantees the order will be correct
    for id in range(len(rev_vocab)):
        word = rev_vocab[id]
        if word in glove.keys():
            embeddings.append(glove[word])
        else:
            # embeddings.append(q[id,:])
            embeddings.append(np.random.uniform(-glorot_limit, glorot_limit, size=(D)).tolist())
    return np.asarray(embeddings, dtype=np.float32)

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/Users/tom/Dropbox/msc-ml/project/src/")
    from preprocessing import char_pos_to_word, tokenise
    item = load_squad_dataset('./data/',False)[0]['paragraphs'][0]
    a = item['qas'][0]['answers'][0]
    context = item['context']
    toks = tokenise(context,asbytes=False)
    print(context)
    print(a)
    print(context[a['answer_start']:])
    ans_span=char_pos_to_word(context.encode(), [t.encode() for t in toks], a['answer_start'])
    ans_span=(ans_span, ans_span+len(tokenise(a['text'],asbytes=False)))
    print(toks[ans_span[0]:ans_span[1]])
