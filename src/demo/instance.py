import sys,os

import tensorflow as tf
import numpy as np
from seq2seq_model import Seq2SeqModel
from maluuba_model import MaluubaModel
from helpers import preprocessing

import json

import flags
FLAGS = tf.app.flags.FLAGS

mem_limit=0.9

class AQInstance():
    def __init__(self, vocab):
        # self.model = Seq2SeqModel(vocab, training_mode=False)
        self.model = MaluubaModel(vocab, training_mode=False)
        with self.model.graph.as_default():
            self.model.ping = tf.constant("ack")
        # self.model = MaluubaModel(vocab, training_mode=False)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_limit,allow_growth = True,visible_device_list='0')
        self.sess = tf.Session(graph=self.model.graph, config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True))

    def load_from_chkpt(self, path):
        self.chkpt_path = path
        with self.model.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, tf.train.latest_checkpoint(path))
            print("Loaded model from "+path)

    def get_q(self, context, ans,ans_pos):
        # Process and create a batch of 1
        ctxt_dict = tuple([np.asarray([x]) for x in preprocessing.process_squad_context(self.model.vocab, context_as_set=FLAGS.context_as_set)(context)])
        ans_dict = tuple([np.asarray([x]) for x in preprocessing.process_squad_answer(self.model.vocab, context_as_set=FLAGS.context_as_set)(ans,ans_pos,context)])

        q,q_len = self.sess.run([self.model.q_hat_beam_string,self.model.q_hat_beam_lens], feed_dict={self.model.context_in: ctxt_dict, self.model.answer_in: ans_dict})
        q_str = " ".join([w.decode().replace('>','&gt;').replace('<','&lt;') for w in q[0][:q_len[0]-1]])
        return q_str

    def ping(self):
        return self.sess.run(self.model.ping)


def main(_):
    import matplotlib.pyplot as plt

    # chkpt_path = FLAGS.model_dir+'saved/qgen-s2s-shortlist'

    chkpt_path = FLAGS.model_dir+'saved2/' + 'MALUUBA-CROP-LATENT-GLOVE/1535108104'
    with open(chkpt_path+'/vocab.json') as f:
        vocab = json.load(f)
    generator = AQInstance(vocab=vocab)
    generator.load_from_chkpt(chkpt_path)

    ctxt="only several hundred are greater than magnitude 3.0 , and only about 15–20 are greater than magnitude 4.0 . the magnitude 6.7 1994 northridge earthquake was particularly destructive , causing a substantial number of deaths , injuries , and structural collapses . it caused the most property damage of any earthquake in u.s. history , estimated at over \$ 20 billion ."
    ans="6.7"
    ans_pos = ctxt.find(ans)

    q_pred = generator.get_q(ctxt.encode(), ans.encode(), ans_pos)
    aligns = generator.alignments

    print(q_pred)

    plt.matplot(alignments)
    plt.show()


if __name__ == '__main__':
    tf.app.run()
