import sys,json,time,os,string,re
sys.path.insert(0, "/Users/tom/Dropbox/msc-ml/project/src/")
sys.path.insert(0, "/Users/tomhosking/Dropbox/msc-ml/project/src/")
sys.path.insert(0, "/cs/student/msc/ml/2017/thosking/dev/msc-project/src/")

import tensorflow as tf
import numpy as np

import flags
import qa.qanet.config
from qa.qanet.model import Model
from qa.qanet.prepro import convert_to_features, word_tokenize
from helpers.preprocessing import tokenise, char_pos_to_word
import helpers.loader as loader
import helpers.metrics as metrics

mem_limit=0.95




# This provides a somewhat normalised interface to a pre-trained QANet model - some tweaks have been made to get it to play nicely when other models are spun up
class QANetInstance():
    def load_from_chkpt(self, path, trainable=False):

        config = tf.app.flags.FLAGS
        with open(config.word_emb_file, "r") as fh:
            word_mat = np.array(json.load(fh), dtype=np.float32)
        with open(config.char_emb_file, "r") as fh:
            char_mat = np.array(json.load(fh), dtype=np.float32)
        # with open(config.test_meta, "r") as fh:
        #     meta = json.load(fh)

        with open(config.word_dictionary, "r") as fh:
            self.word_dictionary = json.load(fh)
        with open(config.char_dictionary, "r") as fh:
            self.char_dictionary = json.load(fh)

        self.model = Model(config, None, word_mat, char_mat, trainable=trainable, demo = True)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_limit,allow_growth = True,visible_device_list='0')
        self.sess = tf.Session(graph=self.model.graph, config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True))

        with self.model.graph.as_default():
            self.saver = tf.train.Saver()
            if trainable:
                self.sess.run(tf.global_variables_initializer())
            else:
                self.saver.restore(self.sess, tf.train.latest_checkpoint(path))
            if config.decay < 1.0:
                self.sess.run(self.model.assign_vars)
    def __del__(self):
        self.sess.close()


    def get_ans(self, contexts, questions):
        config = tf.app.flags.FLAGS

        # query = zip(contexts, questions)
        toks = [word_tokenize(ctxt.replace("''", '" ').replace("``", '" ')) for ctxt in contexts]
        questions = [q.replace(loader.PAD,"").replace(loader.EOS,"") for q in questions]
        query = list(zip(contexts, questions))

        length=config.batch_size
        if len(query) < config.batch_size:
            length=len(query)
            query += [["blank","blank"] for i in range(config.batch_size-len(query))]
        feats=[convert_to_features(config, q, self.word_dictionary, self.char_dictionary) for q in query]
        c,ch,q,qh = zip(*feats)
        fd = {'context:0': c,
              'question:0': q,
              'context_char:0': ch,
              'question_char:0': qh}

        yp1,yp2 = self.sess.run([self.model.yp1, self.model.yp2], feed_dict = fd)
        spans = list(zip(yp1, yp2))[:length]
        return [" ".join(toks[i][span[0]:span[1]+1]) for i,span in enumerate(spans)]

    def train_step(self, contexts, questions, answers):
        config = tf.app.flags.FLAGS

        # query = zip(contexts, questions)
        toks = [word_tokenize(ctxt.replace("''", '" ').replace("``", '" ')) for ctxt in contexts]
        questions = [q.replace(loader.PAD,"").replace(loader.EOS,"") for q in questions]
        query = list(zip(contexts, questions))

        y1,y2 = zip(*answers)

        length=config.batch_size
        if len(query) < config.batch_size:
            length=len(query)
            query += [["blank","blank"] for i in range(config.batch_size-len(query))]
        feats=[convert_to_features(config, q, self.word_dictionary, self.char_dictionary) for q in query]
        c,ch,q,qh = zip(*feats)
        fd = {'context:0': c,
              'question:0': q,
              'context_char:0': ch,
              'question_char:0': qh,
              'answer_index1:0': y1,
              'answer_index2:0': y2,
              'dropout:0': config.dropout}

        _,loss = self.sess.run([self.model.train_op, self.model.loss], feed_dict = fd)
        return loss


def main(_):
    from tqdm import tqdm
    FLAGS = tf.app.flags.FLAGS

    # questions = ["What colour is the car?","When was the car made?","Where was the date?", "What was the dog called?","Who was the oldest cat?"]
    # contexts=["The car is green, and was built in 1985. This sentence should make it less likely to return the date, when asked about a cat. The oldest cat was called creme puff and lived for many years!" for i in range(len(questions))]

    trainable=False

    squad_train_full = loader.load_squad_triples(path="./data/")
    squad_dev_full = loader.load_squad_triples(path="./data/", dev=True, ans_list=True)

    para_limit = FLAGS.test_para_limit
    ques_limit = FLAGS.test_ques_limit
    char_limit = FLAGS.char_limit

    def filter_func(example, is_test=False):
        return len(example["context_tokens"]) > para_limit or \
               len(example["ques_tokens"]) > ques_limit or \
               (example["y2s"][0] - example["y1s"][0]) > ans_limit

    qa = QANetInstance()
    qa.load_from_chkpt("./models/saved/qanet2/", trainable=trainable)

    squad_train=[]
    for x in squad_train_full:
        c_toks = word_tokenize(x[0])
        q_toks = word_tokenize(x[1])
        if len(c_toks) < para_limit and len(q_toks) < ques_limit:
            squad_train.append(x)

    squad_dev=[]
    for x in squad_dev_full:
        c_toks = word_tokenize(x[0])
        q_toks = word_tokenize(x[1])
        if len(c_toks) < para_limit and len(q_toks) < ques_limit:
            squad_dev.append(x)


    num_train_steps = len(squad_train)//FLAGS.batch_size
    num_eval_steps = len(squad_dev)//FLAGS.batch_size

    best_f1=0
    if trainable:
        run_id = str(int(time.time()))
        chkpt_path = FLAGS.model_dir+'qanet/'+run_id
        if not os.path.exists(chkpt_path):
            os.makedirs(chkpt_path)

        summary_writer = tf.summary.FileWriter(FLAGS.log_dir+'qanet/'+run_id, qa.model.graph)
        for i in tqdm(range(FLAGS.qa_num_epochs * num_train_steps)):
            if i%num_train_steps==0:
                print('Shuffling training set')
                np.random.shuffle(squad_train)

            this_batch = squad_train[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
            batch_contexts,batch_questions,batch_ans_text,batch_ans_charpos = zip(*this_batch)

            batch_answers=[]
            for j, ctxt in enumerate(batch_contexts):
                ans_span=char_pos_to_word(ctxt.encode(), [t.encode() for t in word_tokenize(ctxt)], batch_ans_charpos[j])
                ans_span=(np.eye(FLAGS.test_para_limit)[ans_span], np.eye(FLAGS.test_para_limit)[ans_span+len(word_tokenize(batch_ans_text[j]))-1])
                batch_answers.append(ans_span)
            this_loss = qa.train_step(batch_contexts, batch_questions, batch_answers)

            if i %50==0:
                losssummary = tf.Summary(value=[tf.Summary.Value(tag="train_loss/loss",
                                          simple_value=np.mean(this_loss))])

                summary_writer.add_summary(losssummary, global_step=i)

            if i> 0 and i%1000==0:
                qa_f1s=[]
                qa_em=[]

                for j in tqdm(range(num_eval_steps)):
                    this_batch = squad_dev[j*FLAGS.batch_size:(j+1)*FLAGS.batch_size]

                    spans = qa.get_ans([x[0] for x in this_batch] , [x[1] for x in this_batch])

                    for b in range(len(this_batch)):
                        qa_f1s.append(metrics.f1(metrics.normalize_answer(this_batch[b][2]), metrics.normalize_answer(spans[b])))
                        qa_em.append(1.0 * (metrics.normalize_answer(this_batch[b][2]) == metrics.normalize_answer(spans[b])))

                f1summary = tf.Summary(value=[tf.Summary.Value(tag="dev_perf/f1",
                                          simple_value=np.mean(qa_f1s))])

                summary_writer.add_summary(f1summary, global_step=i)
                if np.mean(qa_f1s) > best_f1:
                    print("New best F1! ", np.mean(qa_f1s), " Saving...")
                    best_f1 = np.mean(qa_f1s)
                    qa.saver.save(qa.sess, chkpt_path+'/model.checkpoint')

    qa_f1s=[]
    qa_em=[]

    for i in tqdm(range(num_eval_steps)):
        this_batch = squad_dev[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]

        spans = qa.get_ans([x[0] for x in this_batch] , [x[1] for x in this_batch])

        for b in range(len(this_batch)):
            this_f1s=[]
            this_em=[]
            for a in range(len(this_batch[b][2])):
                this_f1s.append(metrics.f1(metrics.normalize_answer(this_batch[b][2][a]), metrics.normalize_answer(spans[b])))
                this_em.append(1.0 * (metrics.normalize_answer(this_batch[b][2][a]) == metrics.normalize_answer(spans[b])))
            qa_em.append(max(this_em))
            qa_f1s.append(max(this_f1s))

        if i ==0 :
            print(qa_f1s, qa_em)
            print(this_batch[0])
            print(spans[0])

    print('EM: ', np.mean(qa_em))
    print('F1: ', np.mean(qa_f1s))

if __name__ == "__main__":
    tf.app.run()
