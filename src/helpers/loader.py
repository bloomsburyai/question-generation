import json

import re
from collections import defaultdict

import numpy as np

SOS = '<Sent>'
EOS = '</Sent>'
PAD='<PAD>'
OOV='<OOV>'

def load_squad_dataset(dev=False):
    expected_version = '1.1'
    filename = 'train-v1.1.json' if not dev else 'dev-v1.1.json'
    with open('../data/'+filename) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != expected_version):
            print('Expected SQuAD v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
        return(dataset)


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



if __name__ == "__main__":
    print(load_squad_dataset(False)[0]['paragraphs'][0]['qas'][0])
