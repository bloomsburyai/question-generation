import sys,json
sys.path.insert(0, "/Users/tom/Dropbox/msc-ml/project/src/")

import spacy
import helpers.preprocessing as preprocessing

from collections import Counter

# Spacy entity types
# PERSON	People, including fictional.
# NORP	Nationalities or religious or political groups.
# FAC	Buildings, airports, highways, bridges, etc.
# ORG	Companies, agencies, institutions, etc.
# GPE	Countries, cities, states.
# LOC	Non-GPE locations, mountain ranges, bodies of water.
# PRODUCT	Objects, vehicles, foods, etc. (Not services.)
# EVENT	Named hurricanes, battles, wars, sports events, etc.
# WORK_OF_ART	Titles of books, songs, etc.
# LAW	Named documents made into laws.
# LANGUAGE	Any named language.
# DATE	Absolute or relative dates or periods.
# TIME	Times smaller than a day.
# PERCENT	Percentage, including "%".
# MONEY	Monetary values, including unit.
# QUANTITY	Measurements, as of weight or distance.
# ORDINAL	"first", "second", etc.
# CARDINAL	Numerals that do not fall under another type.


# This is a really really really bad model! Find nearest entity and verb and squidge them together using hand coded rules
class BaselineModel():
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

        self.patterns = {
            'FAC': {
                'THING' : "What building {1} near {0}?",
                'PERSON' : "What building {1} by {0}?",
                'LOC' : "What building {1} in {0}?",
                'FAC' : "What building {1} in {0}?",
                'DATE' : "What building {1} on {0}?"
            },
            'ORDINAL' : {
                'FAC': "What number {0} {1}?",
                'THING': "What number {0} {1}?",
                'ORDINAL': "What number {0} {1}?",
                'PERSON': "What number {1} {0}?",
                'LOC': "What number {1} in {0}?"
            },
            'DATE' : {
                'DATE': "When was {0} {1}?",
                'THING': "When did {0} {1}?",
                'PERSON': "When did {0} {1}?",
                'FAC': "When did {0} {1}?",
                'LOC': "When did {0} {1}?",
                'ORDINAL': "When {1} {0}?",
                'PERCENT': "When {1} {0}?"
            },
            'TIME' : {
                'DATE': "When was {0} {1}?",
                'TIME': "When was {0} {1}?",
                'THING': "When did {0} {1}?",
                'PERSON': "When did {0} {1}?",
                'FAC': "When did {0} {1}?",
                'LOC': "When did {0} {1}?",
                'ORDINAL': "When {1} {0}?"
            },
            'THING': {
                'FAC': "What {1} in {0}?",
                'LOC': "What {1} in {0}?",
                'DATE': "What {1} in {0}?",
                'TIME': "What {1} at {0}?",
                'PERSON': "What {1} by {0}?",
                'THING': "What {1} {0}?",
                'ORDINAL': "What {1} {0}?",
                'PERCENT': "What {1}?"
            },
            'CARDINAL': {
                'THING': "How many {1} {0}?",
                'LOC': "How many {1} in {0}?",
                'DATE': "How many {1} in {0}?",
                'FAC': "How many {1} at {0}?",
                'PERSON': "How many {1} {0}?",
                'ORDINAL': "How many {1} {0}?",
                'TIME': "How many {1} at {0}?"
            },
            'LOC': {
                'THING': "Where {1} {0}?",
                'PERSON': "Where {1} {0}?",
                'ORDINAL': "Where {1} {0}?",
                'LOC' : "Where {1} in {0}?",
                'FAC' : "Where {1} at {0}?",
                'DATE' : "Where {1} in {0}?"
            },
            'PERSON': {
                'LOC': "Who {1} in {0}?",
                'THING': "Who {1} {0}?",
                'ORDINAL': "Who {1} {0}?",
                'DATE': "Who {1} in {0}?",
                'TIME': "Who {1} at {0}?",
                'PERSON': "Who {1} by {0}?",
                'FAC': "Who {1} at {0}?",
            },
            'PERCENT': {
                'PERCENT': "What percentage {1}?",
                'PERSON': "What percentage {0} {1}?",
                'ORDINAL': "What percentage {1} {0}?",
                'THING': "What percentage {0} {1}?",
                'LOC': "What percentage {1} in {0}?",
                'DATE': "What percentage {1} on {0}?"
            }

        }


    def format_q(self, ans_type, entity_type, entity_toks, verb):
        if ans_type not in self.patterns.keys():
            exit(ans_type + " is not a known answer type ("+entity_type+" entity type)")
        if entity_type not in self.patterns[ans_type].keys():
            exit(entity_type + " is not a known entity type under "+ ans_type+" answer type")
        q_template = self.patterns[ans_type][entity_type]
        return q_template.format(" ".join(entity_toks), verb)

    def get_q(self, ctxt, ans, ans_pos):
        ctxt_filt, ans_pos = preprocessing.filter_context(ctxt, ans_pos, 0, 30)
        ans_toks = preprocessing.tokenise(ans, asbytes=False)
        doc = self.nlp(ctxt_filt)
        ctxt_toks = [str(tok).lower() for tok in doc]
        # ans_ix = preprocessing.char_pos_to_word(ctxt_filt, ctxt_toks, ans_pos, asbytes=False)
        if ans_toks[0] not in ctxt_toks:
            # print(ans_toks[0], ctxt_toks)
            ans_ix=preprocessing.char_pos_to_word(ctxt_filt, ctxt_toks, ans_pos, asbytes=False)
            # print(ctxt_toks[ans_ix])
        else:
            ans_ix = ctxt_toks.index(ans_toks[0])


        ans_type = Counter([doc[i].ent_type_ for i in range(ans_ix, min(ans_ix+len(ans_toks), len(doc)))]).most_common()[0][0]
        # print(ans_type)

        type_distances=[]
        verb_distances=[]
        for offset in range(len(ctxt_toks)):
            # print(doc[offset].ent_type_, doc[offset])
            if str(doc[offset]).lower() not in ans_toks:
                # print(doc[offset], ans_toks)
                if doc[offset].pos_ == 'NOUN':
                    type_distances.append((max(offset-ans_ix-len(ans_toks)+1, ans_ix-offset), 'THING', doc[offset], offset))
                if doc[offset].ent_type_ != '' \
                    and not (doc[offset].ent_iob_ == 'B' and str(doc[min(offset+1, len(doc)-1)]).lower() in ans_toks) \
                    and self.type_translate(doc[offset].ent_type_) != 'CARDINAL':
                    type_distances.append((max(offset-ans_ix-len(ans_toks)+1, ans_ix-offset), doc[offset].ent_type_, doc[offset], offset))
                if doc[offset].tag_ in ['VBG','VBN']:
                    # print(doc[offset])
                    verb_distances.append((max(offset-ans_ix-len(ans_toks)+1, ans_ix-offset), doc[offset].tag_, doc[offset], offset))

        nearest_verb = sorted(verb_distances, key=lambda x: x[0])[0] if len(verb_distances) >0 else (0,'VBG', 'is',0)

        if len(type_distances) >0:
            nearest_entity = sorted(type_distances, key=lambda x: x[0])[0]
            ix= nearest_entity[3]
            entity_ixs=[ix]
            # print(nearest_entity)
            while ix+1 < len(doc) and doc[ix+1].ent_iob_ == 'I':
                entity_ixs.append(ix+1)
                ix+=1

            entity_toks = [str(tok) for tok in doc[entity_ixs[0]:entity_ixs[-1]+1]]
            entity_type=nearest_entity[1]
        else:
            entity_toks = ["thing"]
            entity_type="THING"

        # print(entity_toks)
        return self.format_q(self.type_translate(ans_type), self.type_translate(entity_type), entity_toks, nearest_verb[2])

    def type_translate(self,type):
        if type in ["WORK_OF_ART","","ORG","PRODUCT","EVENT","LANGUAGE","LAW"]:
            return "THING"
        if type in ["NORP","PERSON"]:
            return "PERSON"
        if type in ["QUANTITY","CARDINAL","MONEY"]:
            return "CARDINAL"
        if type in ["LOC","GPE"]:
            return "LOC"
        return type


if __name__ == "__main__":
    import helpers.loader as loader
    import helpers.metrics as metrics

    from langmodel.lm import LstmLmInstance
    from qa.mpcm import MpcmQaInstance


    import flags
    import tensorflow as tf
    import numpy as np
    from tqdm import tqdm
    FLAGS = tf.app.flags.FLAGS


    train_data = loader.load_squad_triples('./data/', True)[:1500]

    model = BaselineModel()

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

    for i in tqdm(range(len(train_data))):
        triple=train_data[i]
        ctxt,q,ans,ans_pos = triple
        ctxt_toks = preprocessing.tokenise(ctxt, asbytes=False)

        # print(triple[0])
        gen_q = model.get_q(triple[0], triple[2], triple[3])
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
