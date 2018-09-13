import sys,json,time,os
sys.path.insert(0, "/Users/tom/Dropbox/msc-ml/project/src/")
sys.path.insert(0, "/cs/student/msc/ml/2017/thosking/dev/msc-project/src/")

import tensorflow as tf
import numpy as np

import discriminator.config
from discriminator.model import Model
from discriminator.prepro import convert_to_features, word_tokenize
import helpers.loader as loader
import flags

mem_limit=1


# This provides a somewhat normalised interface to a pre-trained QANet model - some tweaks have been made to get it to play nicely when other models are spun up
class DiscriminatorInstance():
    def __init__(self, trainable=False, path=None, log_slug=None, force_init=False):
        config = tf.app.flags.FLAGS
        self.run_id = str(int(time.time())) + ("-"+log_slug if log_slug is not None else "")
        self.trainable = trainable
        self.load_from_chkpt(path, force_init)
        if trainable:
            self.summary_writer = tf.summary.FileWriter(config.log_dir+'disc/'+self.run_id, self.model.graph)
    def __del__(self):
        self.sess.close()


    def load_from_chkpt(self, path=None, force_init=False):

        config = tf.app.flags.FLAGS
        with open(config.disc_word_emb_file, "r") as fh:
            word_mat = np.array(json.load(fh), dtype=np.float32)
        with open(config.disc_char_emb_file, "r") as fh:
            char_mat = np.array(json.load(fh), dtype=np.float32)
        # with open(config.disc_test_meta, "r") as fh:
        #     meta = json.load(fh)

        with open(config.disc_word_dictionary, "r") as fh:
            self.word_dictionary = json.load(fh)
        with open(config.disc_char_dictionary, "r") as fh:
            self.char_dictionary = json.load(fh)


        self.model = Model(config, None, word_mat, char_mat, trainable=self.trainable, demo = True, opt=False)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_limit,allow_growth = True,visible_device_list='0')
        self.sess = tf.Session(graph=self.model.graph, config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True))

        with self.model.graph.as_default():
            self.saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
            if force_init and path is not None:
                chkpt_path = tf.train.latest_checkpoint(path)
                print("Loading discriminator from ", chkpt_path)

                restore_vars= [v for v in tf.trainable_variables() if v.name[:13] != 'Output_Layer/']
                self.sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver(restore_vars)
                saver.restore(self.sess, chkpt_path)
            elif path is not None:


                chkpt_path = tf.train.latest_checkpoint(path)
                print("Loading discriminator from ", chkpt_path)
                self.saver.restore(self.sess, chkpt_path)
                if config.disc_decay < 1.0:
                    self.sess.run(self.model.assign_vars)
            else:

                os.makedirs(config.model_dir+'disc/'+self.run_id)
                self.sess.run(tf.global_variables_initializer())



    def save_to_chkpt(self, path, step):
        self.saver.save(self.sess, path+'disc/'+self.run_id+'/model.checkpoint', global_step=step)

    def char_pos_to_word(self, text, tokens, char_pos):
        ix=0
        for t,token in enumerate(tokens):
            # print(token, t, ix, char_pos)
            for char in token:
                ix = text.find(char, ix)
                # ix += 1
                if ix >= char_pos:
                    # print("***", token, char, t, ix, char_pos)
                    return t

    def prepro(self,contexts, questions, ans_text, ans_pos):
        config = tf.app.flags.FLAGS


        # query = zip(contexts, questions)
        toks = [word_tokenize(ctxt.replace("''", '" ').replace("``", '" ').lower()) for ctxt in contexts]
        ans_tok_pos = [self.char_pos_to_word(contexts[ix].lower(), toks[ix], ans_pos[ix]) for ix in range(len(toks))]
        ans_lens = [len(word_tokenize(ans)) for ans in ans_text]
        ans_toks = [toks[ix][ans:ans+ans_lens[ix]] for ix,ans in enumerate(ans_tok_pos)]

        # print(ans_pos)
        # print(ans_toks)
        # print(toks)
        # exit()
        # ans_start = [toks[i].index(ans_tok[0]) for i,ans_tok in enumerate(ans_toks)]
        # ans_end = [ans_start[i] + len(ans_toks[i])-1 for i in range(len(ans_toks))]
        ans_start = ans_pos
        ans_end = [ans+ans_lens[ix]-1 for ix,ans in enumerate(ans_pos)]
        questions = [q.replace(loader.PAD,"").replace(loader.EOS,"") for q in questions]
        query = list(zip(contexts, questions))

        # # the QANet code has fixed batch sizes - so pad it
        # length=config.batch_size
        # if len(query) < config.batch_size:
        #     length=len(query)
        #     query += [["blank","blank"] for i in range(config.batch_size-length)]
        #     ans_start += [0  for i in range(config.batch_size-length)]
        #     ans_end += [0  for i in range(config.batch_size-length)]

        feats=[convert_to_features(config, q, self.word_dictionary, self.char_dictionary)+(ans_start[ix],ans_end[ix]) for ix,q in enumerate(query)]
        return feats

    def get_pred(self, contexts, questions, ans_text, ans_pos):
        length = len(contexts)

        feats = self.prepro(contexts,questions,ans_text,ans_pos)
        c,ch,q,qh,ans_start,ans_end = zip(*feats)
        fd = {'context:0': c,
              'question:0': q,
              'context_char:0': ch,
              'question_char:0': qh,
              'answer_index1:0': ans_start,
              'answer_index2:0': ans_end}

        pred = self.sess.run(self.model.probs, feed_dict = fd)

        return pred[:length]

    def get_nll(self, contexts, questions, ans_text, ans_pos, gold_labels):
        length = len(contexts)

        feats = self.prepro(contexts,questions,ans_text,ans_pos)
        c,ch,q,qh,ans_start,ans_end = zip(*feats)
        fd = {'context:0': c,
              'question:0': q,
              'context_char:0': ch,
              'question_char:0': qh,
              'answer_index1:0': ans_start,
              'answer_index2:0': ans_end,
              'gold_class:0': gold_labels}

        nll = self.sess.run(self.model.nll, feed_dict = fd)

        return nll[:length]

    def train_step(self, contexts, questions, ans_text, ans_pos, gold_labels, step):
        if not self.trainable:
            exit('train_step called on non-trainable discriminator!')
        config = tf.app.flags.FLAGS

        length = len(contexts)
        gold_labels = gold_labels
        feats = self.prepro(contexts,questions,ans_text,ans_pos)
        c,ch,q,qh,ans_start,ans_end = zip(*feats)
        fd = {'context:0': c,
              'question:0': q,
              'context_char:0': ch,
              'question_char:0': qh,
              'answer_index1:0': ans_start,
              'answer_index2:0': ans_end,
              'gold_class:0': gold_labels,
              self.model.dropout: config.disc_dropout}

        _,summ,loss = self.sess.run([self.model.train_op, self.model.train_summary, self.model.loss], feed_dict = fd)

        # if step % 25 ==0:
        #     print(gold_labels, questions)

        self.summary_writer.add_summary(summ, global_step=step)

        return loss


def main(_):

    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import confusion_matrix
    import itertools
    from sklearn.metrics import roc_curve, auc

    # squad = loader.load_squad_triples(path="./data/", dev=True, v2=True, as_dict=True)
    # with open("./data/squad2_dev_official_output_fixed.json") as dataset_file:
    #     ans_preds = json.load(dataset_file)
    with open("./results/out_eval_MALUUBA-CROP-LATENT-GLOVE_test.json") as dataset_file:
        results = json.load(dataset_file)['results']



# 1535473379-MALUUBA-CROP-LATENT-GLOVE_train
# 1535474306-MALUUBA-CROP-LATENT-GLOVE_train_QAINIT
    # disc_path = "./models/saved/discriminator-trained"
    disc_path = "./models/saved2/1535473379-MALUUBA-CROP-LATENT-GLOVE_train"
    # disc_path = "./models/saved2/1535474306-MALUUBA-CROP-LATENT-GLOVE_train_QAINIT"


    disc = DiscriminatorInstance(path=disc_path)
    # disc = DiscriminatorInstance(path="./models/disc/1533307366-SQUAD-QANETINIT")

    # output={}
    # for id,candidates in tqdm(ans_preds.items()):
    #     ctxt, q, ans_gold, ans_gold_pos, label_gold = squad[id]
    #
    #     scores=[]
    #     for candidate in candidates:
    #         scores.append( disc.get_pred([ctxt], [q], [candidate['text']], [candidate['answer_start']]).tolist()[0] )
    #     cand_ix = np.argmax(scores)
    #
    #     pred_ans = candidates[cand_ix]['text']
    #     pred_score = scores[cand_ix]
    #     output[id] = pred_ans if pred_score > 0.5 else ""
    #
    # with open("./logs/squad2_dev_filtered.json","w") as fh:
    #     json.dump(output, fh)

    gold_labels=[]
    pred_labels=[]
    scores=[]

    for res in tqdm(results[:3000]):
        # print(res['q_gold'], res['q_pred'])
        gold_score = disc.get_pred([res['c']], [res['q_gold']],[res['a_text']],[res['a_pos']])
        pred_score = disc.get_pred([res['c']], [res['q_pred']],[res['a_text']],[res['a_pos']])


        gold_labels.append(1)
        gold_labels.append(0)
        pred_labels.append(1.0 * (gold_score[0] > 0.5))
        pred_labels.append(1.0 * (pred_score[0] > 0.5))
        scores.append(gold_score[0])
        scores.append(pred_score[0])


    print(disc_path)
    print("Acc: ", np.mean(np.equal(gold_labels, pred_labels)))



    # oh_labels =np.eye(2)[gold_labels]
    ### disc conf mat
    # gold_labels =['Generated' if l==0 else 'Ground truth' for l in gold_labels]
    # pred_labels =['Generated' if l==0 else 'Ground truth' for l in pred_labels]
    plt.figure(figsize=(4.5,3.5))
    cm = confusion_matrix(gold_labels, pred_labels)
    mat = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(mat)
    plt.imshow(mat, cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks,['Generated','Ground truth'],  rotation=0)
    plt.yticks(tick_marks, ['Generated','Ground truth'],  rotation=90)
    fmt = '.2f'
    thresh = mat.max() / 2.
    for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
        plt.text(j, i, format(mat[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if mat[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('Actual Source', fontsize=14)
    plt.xlabel('Predicted source', fontsize=14)
    plt.savefig("/users/Tom/Dropbox/Apps/Overleaf/Question Generation/figures/disc_cm_latent.pdf", format="pdf", bbox_inches="tight")
    # plt.show()
    # exit()



    ### disc Roc curves
    fpr, tpr, _ = roc_curve(gold_labels, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(4.5,3.5))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Discriminator RoC curve')
    plt.legend(loc="lower right")
    plt.savefig("/users/Tom/Dropbox/Apps/Overleaf/Question Generation/figures/disc_roc_latent.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    exit()


if __name__ == "__main__":
    tf.app.run()
