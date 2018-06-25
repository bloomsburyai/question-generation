import sys,json
sys.path.insert(0, "/Users/tom/Dropbox/msc-ml/project/src/")


import requests
import helpers.preprocessing as preprocessing
import helpers.loader as loader

import helpers.metrics as metrics

from langmodel.lm import LstmLmInstance
from qa.mpcm import MpcmQaInstance

import flags
import tensorflow as tf
import numpy as np
from tqdm import tqdm
FLAGS = tf.app.flags.FLAGS

import baseline_model


url = 'http://localhost:9000/?properties={"annotators":"openie,ner","outputFormat":"json","openie.affinity_probability_cap":0.01}'

train_data = loader.load_squad_triples('./data/', True)[:1500]

def get_q_word(ner):
    if ner in ["MISC","UNK","IDEOLOGY","RELIGION"]:
        return "what"
    elif ner in ["PERSON", "ORGANIZATION","TITLE"]:
        return "who"
    elif ner in ["NUMBER", "MONEY"]:
        return "how many"
    elif ner in ["DATE", "TIME"]:
        return "when"
    elif ner in ["STATE_OR_PROVINCE","COUNTRY","CITY","LOCATION"]:
        return "where"
    elif ner in ["DURATION"]:
        return "how long"
    elif ner in ["ORDINAL"]:
        return "which"
    elif ner in ["MONEY", "PERCENT"]:
        return "how much"
    elif ner in ["NATIONALITY"]:
        return "what nationality"
    elif ner in ["CAUSE_OF_DEATH"]:
        return "how"
    else:
        exit("Unknown ner "+ ner)

fallback = baseline_model.BaselineModel()

lm = LstmLmInstance()
qa = MpcmQaInstance()


lm.load_from_chkpt(FLAGS.model_dir+'saved/lmtest')
print("LM loaded")
qa.load_from_chkpt(FLAGS.model_dir+'saved/qatest')
print("QA loaded")

lm_vocab = lm.vocab
qa_vocab = qa.vocab

f1s=[]
bleus=[]
qa_scores=[]
lm_scores=[]

# Stupid model:
# F1:  0.16030058972235775
# BLEU:  0.00194936162888982
# QA:  0.04730156660601784
# LM:  603.4203351325989

# Without LM filtering:
# F1:  0.15268398668332012
# BLEU:  0.0021880572346774834
# QA:  0.06118585070207543
# LM:  534.2029482313792

# Filtered candidates by LM
# F1:  0.1523949117692406
# BLEU:  0.0015981573389975351
# QA:  0.060554504435173614
# LM:  480.1121001809438

for i in tqdm(range(len(train_data))):
    triple=train_data[i]

    ctxt,q,ans,ans_pos = triple

    ctxt_filt, ans_pos = preprocessing.filter_context(ctxt, ans_pos, 0, 30)
    ctxt_toks = preprocessing.tokenise(ctxt, asbytes=False)


    response = requests.post(url, data=ctxt_filt.encode('utf-8'))
    if response.status_code != 200:
        exit("There was a problem connecting to the CoreNLP server!")

    res = response.json()
    # print(ctxt_filt)
    candidates=[]

    # Run NER to get question word
    for ent in res['sentences'][0]['entitymentions']:
        if ent['text'].find(ans):
            ner = ent['ner']
        else:
            ner = "UNK"

    # Run information extraction to build context
    for relation in res['sentences'][0]['openie']:
        if relation['subject'].find(ans) > -1:
            # print(ans,relation)
            candidates.append(get_q_word(ner)+" "+relation['relation'] +" "+ relation['object'] +"?")
        if relation['object'].find(ans) > -1:
            # print(ans,relation)
            candidates.append(relation['subject'] +" was "+ relation['relation'] +" "+get_q_word(ner)+"?")
        if relation['relation'].find(ans) > -1:
            candidates.append("How was "+ relation['object'] +" by "+ relation['subject'] +" "+"?")

    if len(candidates) ==0:
        gen_q = fallback.get_q(ctxt_filt, ans, ans_pos)
    else:
        # gen_q = candidates[0]
        qs_for_lm = [preprocessing.lookup_vocab(preprocessing.tokenise(cand, asbytes=False), lm_vocab, do_tokenise=False, asbytes=False).tolist() for cand in candidates]
        max_len = max([len(lmq) for lmq in qs_for_lm])
        qs_for_lm = [lmq +[lm_vocab[loader.PAD] for j in range(max_len-len(lmq))] for lmq in qs_for_lm]
        ppls = lm.get_seq_perplexity(np.asarray(qs_for_lm))
        best_ix = np.argmin(ppls)
        gen_q = candidates[best_ix]

    gen_q_toks = preprocessing.tokenise(gen_q, asbytes=False)

    f1s.append(metrics.f1(triple[1], gen_q))
    bleus.append(metrics.bleu(triple[1], gen_q))

    qhat_for_lm = preprocessing.lookup_vocab(gen_q_toks, lm_vocab, do_tokenise=False, asbytes=False)
    ctxt_for_lm = preprocessing.lookup_vocab(ctxt_toks, lm_vocab, do_tokenise=False, asbytes=False)
    qhat_for_qa = preprocessing.lookup_vocab(gen_q_toks, qa_vocab, do_tokenise=False, asbytes=False)
    ctxt_for_qa = preprocessing.lookup_vocab(ctxt_toks, qa_vocab, do_tokenise=False, asbytes=False)

    qa_pred = qa.get_ans(np.asarray([ctxt_for_qa]), np.asarray([qhat_for_qa])).tolist()[0]
    pred_ans = " ".join([w for w in ctxt_toks[qa_pred[0]:qa_pred[1]]])

    qa_scores.append(metrics.f1(ans, pred_ans))
    lm_scores.append(lm.get_seq_perplexity([qhat_for_lm]).tolist()[0]) # lower perplexity is better

print("F1: ", np.mean(f1s))
print("BLEU: ", np.mean(bleus))
print("QA: ", np.mean(qa_scores))
print("LM: ", np.mean(lm_scores))
