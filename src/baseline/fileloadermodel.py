import sys
sys.path.insert(0, "/Users/tom/Dropbox/msc-ml/project/src/")
sys.path.insert(0, "/cs/student/msc/ml/2017/thosking/dev/msc-project/src/")

from tqdm import tqdm
import numpy as np
import json

import helpers.loader as loader
import helpers.metrics as metrics
import helpers.preprocessing as preprocessing
import tensorflow as tf

import flags

from langmodel.lm import LstmLmInstance
# from qa.mpcm import MpcmQaInstance
from qa.qanet.instance import QANetInstance
from discriminator.instance import DiscriminatorInstance

FLAGS = tf.app.flags.FLAGS

class FileLoaderModel():
    def __init__(self, path):
        self.path = path
        with open(path+'/squad_dev_baseline.txt', 'r', encoding='utf-8') as fp:
            self.questions = [line.strip() for line in fp.readlines()]
        with open(path+'/dev.txt.shuffle.combined.id', 'r', encoding='utf-8') as fp:
            self.ids = [line.strip() for line in fp.readlines()]
        print(len(self.questions), len(self.ids))

    def get_q(self, id):
        if id in self.ids:
            ix = self.ids.index(id)
            return self.questions[ix]
        else:
            return None

def main(_):
    model = FileLoaderModel('./models/BASELINE')
    squad = loader.load_squad_triples(FLAGS.data_path, True, as_dict=True)

    disc_path = FLAGS.model_dir+'saved/discriminator-trained-latent'

    glove_embeddings = loader.load_glove(FLAGS.data_path)


    if FLAGS.eval_metrics:
        lm = LstmLmInstance()
        # qa = MpcmQaInstance()
        qa = QANetInstance()

        lm.load_from_chkpt(FLAGS.model_dir+'saved/lmtest')
        # qa.load_from_chkpt(FLAGS.model_dir+'saved/qatest')
        qa.load_from_chkpt(FLAGS.model_dir+'saved/qanet')

        discriminator = DiscriminatorInstance(trainable=False, path=disc_path)

    f1s=[]
    bleus=[]
    qa_scores=[]
    qa_scores_gold=[]
    lm_scores=[]
    nlls=[]
    disc_scores=[]
    sowe_similarities=[]

    qgolds=[]
    qpreds=[]
    ctxts=[]
    answers=[]
    ans_positions=[]

    metric_individuals=[]
    res=[]

    missing=0

    for id,el in tqdm(squad.items()):

        unfilt_ctxt_batch = [el[0]]
        a_text_batch = [el[2]]
        a_pos_batch = [el[3]]

        ctxts.extend(unfilt_ctxt_batch)
        answers.extend(a_text_batch)
        ans_positions.extend(a_pos_batch)

        pred_str = model.get_q(id)

        if pred_str is None:
            missing +=1
            continue
        gold_str = el[1]

        if FLAGS.eval_metrics:
            qa_pred = qa.get_ans(unfilt_ctxt_batch, [pred_str])
            gold_qa_pred = qa.get_ans(unfilt_ctxt_batch, [gold_str])

            qa_score = metrics.f1(el[2].lower(), qa_pred[0].lower())
            qa_score_gold = metrics.f1(el[2].lower(), gold_qa_pred[0].lower())
            lm_score = lm.get_seq_perplexity([pred_str]).tolist()
            disc_score = discriminator.get_pred(unfilt_ctxt_batch, [pred_str], a_text_batch, a_pos_batch).tolist()[0]



        f1s.append(metrics.f1(gold_str, pred_str))
        bleus.append(metrics.bleu(gold_str, pred_str))
        qgolds.append(gold_str)
        qpreds.append(pred_str)

        # calc cosine similarity between sums of word embeddings
        pred_sowe = np.sum(np.asarray([glove_embeddings[w] if w in glove_embeddings.keys() else np.zeros((FLAGS.embedding_size,)) for w in preprocessing.tokenise(pred_str ,asbytes=False)]) ,axis=0)
        gold_sowe = np.sum(np.asarray([glove_embeddings[w] if w in glove_embeddings.keys() else np.zeros((FLAGS.embedding_size,)) for w in preprocessing.tokenise(gold_str ,asbytes=False)]) ,axis=0)
        this_similarity = np.inner(pred_sowe, gold_sowe)/np.linalg.norm(pred_sowe, ord=2)/np.linalg.norm(gold_sowe, ord=2)

        sowe_similarities.append(this_similarity)


        this_metric_dict={
            'f1':f1s[-1],
            'bleu': bleus[-1],
            'nll': 0,
            'sowe': sowe_similarities[-1]
            }
        if FLAGS.eval_metrics:
            this_metric_dict={
            **this_metric_dict,
            'qa': qa_score,
            'lm': lm_score,
            'disc': disc_score}
            qa_scores.append(qa_score)
            lm_scores.append(lm_score)
            disc_scores.append(disc_score)
        metric_individuals.append(this_metric_dict)

        res.append({
            'c': el[0],
            'q_pred': pred_str,
            'q_gold': gold_str,
            'a_pos': el[3],
            'a_text': el[2],
            'metrics': this_metric_dict
        })


    metric_dict={
        'f1':np.mean(f1s),
        'bleu':np.mean(bleus),
        'nll':0,
        'sowe': np.mean(sowe_similarities)
        }
    if FLAGS.eval_metrics:
        metric_dict={**metric_dict,
        'qa':np.mean(qa_scores),
        'lm':np.mean(lm_scores),
        'disc': np.mean(disc_scores)}
    # print(res)
    with open(FLAGS.log_dir+'out_eval_BASELINE'+("_train" if not FLAGS.eval_on_dev else "")+'.json', 'w', encoding='utf-8') as fp:
        json.dump({"metrics":metric_dict, "results": res}, fp)


    print("F1: ", np.mean(f1s))
    print("BLEU: ", np.mean(bleus))
    print("NLL: ", 0)
    print("SOWE: ", np.mean(sowe_similarities))
    if FLAGS.eval_metrics:
        print("QA: ", np.mean(qa_scores))
        print("LM: ", np.mean(lm_scores))
        print("Disc: ", np.mean(disc_scores))

    print(missing," ids were missing")

if __name__ == "__main__":
    tf.app.run()
