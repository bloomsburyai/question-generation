import sys,os
sys.path.insert(0, "/Users/tom/Dropbox/msc-ml/project/src/")
sys.path.insert(0, "/home/tomhosking/webapps/qgen/qgen/src/")


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
        self.model = Seq2SeqModel(vocab, training_mode=False)
        with self.model.graph.as_default():
            self.model.ping = tf.constant("ack")
        # self.model = MaluubaModel(vocab, training_mode=False)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_limit,allow_growth = True,visible_device_list='0')
        self.sess = tf.Session(graph=self.model.graph, config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True))

    def load_from_chkpt(self, path):
        self.chkpt_path = path
        with self.model.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, path+ '/model.checkpoint')
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

model_list = ['qgen-s2s-filt1']

from flask import Flask, current_app, request, redirect

app = Flask(__name__)

@app.route("/")
def index():
    return redirect("/static/demo.htm")

@app.route("/api/generate")
def get_q():
    question = "" + request.args['context'] + request.args['answer']
    ctxt = request.args['context']
    ans = request.args['answer']
    ans_pos = ctxt.find(ans)

    ctxt,ans_pos = preprocessing.filter_context(ctxt, ans_pos, FLAGS.filter_window_size, FLAGS.filter_max_tokens)
    if ans_pos > -1:
        q =current_app.generator.get_q(ctxt.encode(), ans.encode(), ans_pos)
        return q
    else:
        return "Couldnt find ans in context!"

@app.route("/api/ping")
def ping():
    return app.generator.ping()

@app.route("/api/model_list")
def model_list():
    return json.dumps(model_list)

def init():
    print('Spinning up AQ demo app')

    # chkpt_path = FLAGS.model_dir+'saved/qgen-s2s-shortlist'

    if "WEB" in os.environ:
        FLAGS.data_path = '/home/tomhosking/webapps/qgen/qgen/data/'
        FLAGS.log_dir = 'home/tomhosking/webapps/qgen/qgen/logs/'
        chkpt_path = '/home/tomhosking/webapps/qgen/qgen/models/saved/qgen-s2s-filt1'
    else:
        chkpt_path = FLAGS.model_dir+'saved/qgen-s2s-filt1'
    with open(chkpt_path+'/vocab.json') as f:
        vocab = json.load(f)
    app.generator = AQInstance(vocab=vocab)
    app.generator.load_from_chkpt(chkpt_path)

if __name__ == '__main__':
    init()
    with app.app_context():
        app.run(port=14045)
