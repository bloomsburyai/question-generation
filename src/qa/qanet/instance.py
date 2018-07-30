import sys,json
sys.path.insert(0, "/Users/tom/Dropbox/msc-ml/project/src/")

import tensorflow as tf
import numpy as np

import qa.qanet.config
from qa.qanet.model import Model
from qa.qanet.prepro import convert_to_features, word_tokenize
from helpers.preprocessing import tokenise
import helpers.loader as loader

mem_limit=0.5


# This provides a somewhat normalised interface to a pre-trained QANet model - some tweaks have been made to get it to play nicely when other models are spun up
class QANetInstance():
    def load_from_chkpt(self, path):

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

        self.model = Model(config, None, word_mat, char_mat, trainable=False, demo = True)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_limit,allow_growth = True,visible_device_list='0')
        self.sess = tf.Session(graph=self.model.graph, config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True))

        with self.model.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, tf.train.latest_checkpoint(path))
            if config.decay < 1.0:
                self.sess.run(self.model.assign_vars)

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

def main(_):
    questions = ["What colour is the car?","When was the car made?","Where was the date?", "What was the dog called?","Who was the oldest cat?"]
    contexts=["The car is green, and was built in 1985. This sentence should make it less likely to return the date, when asked about a cat. The oldest cat was called creme puff and lived for many years!" for i in range(len(questions))]

    qa = QANetInstance()
    qa.load_from_chkpt("./models/saved/qanet/")

    spans = qa.get_ans(contexts, questions)

    print(contexts[0])
    for i, q in enumerate(questions):

        print(q, "->", spans[i])

if __name__ == "__main__":
    tf.app.run()
