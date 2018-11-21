import sys,os

import tensorflow as tf
import numpy as np

from instance import AQInstance

from helpers import preprocessing

import json

import flags
FLAGS = tf.app.flags.FLAGS

mem_limit=0.9


# model_slug_list = ['MALUUBA-CROP-SET-GLOVE/1535568489','SEQ2SEQ/1533280948','qgen-s2s-filt1']
model_slug_list = ['MALUUBA-CROP-SET-GLOVE']
model_slug_curr = model_slug_list[0]

from flask import Flask, current_app, request, redirect

app = Flask(__name__)

@app.route("/")
def index():
    return redirect("/static/demo.htm")

@app.route("/api/generate")
def get_q():

    ctxt = request.args['context']
    ans = request.args['answer']
    ans_pos = ctxt.find(ans)

    if FLAGS.filter_window_size_before >-1:
        ctxt,ans_pos = preprocessing.filter_context(ctxt, ans_pos, FLAGS.filter_window_size_before, FLAGS.filter_window_size_after, FLAGS.filter_max_tokens)
    if ans_pos > -1:
        q =current_app.generator.get_q(ctxt.encode(), ans.encode(), ans_pos)
        return q
    else:
        print(request.args)
        print(ctxt)
        print(ans)
        return "Couldnt find ans in context!"

@app.route("/api/ping")
def ping():
    return app.generator.ping()

@app.route("/api/model_list")
def model_slug():
    return json.dumps(model_list)

@app.route("/api/model_current")
def model_list():
    return model_slug_curr

def init():
    print('Spinning up AQ demo app')

    # chkpt_path = FLAGS.model_dir+'saved/qgen-s2s-shortlist'

    if "WEB" in os.environ:
        FLAGS.data_path = '/home/tomhosking/webapps/qgen/qgen/data/'
        FLAGS.log_dir = 'home/tomhosking/webapps/qgen/qgen/logs/'
        chkpt_path = '/home/tomhosking/webapps/qgen/qgen/models/saved/' + model_slug_curr
    else:
        chkpt_path = FLAGS.model_dir+'qgen/' + model_slug_curr
    with open(chkpt_path+'/vocab.json') as f:
        vocab = json.load(f)
    app.generator = AQInstance(vocab=vocab)
    app.generator.load_from_chkpt(chkpt_path)

if __name__ == '__main__':
    init()
    with app.app_context():
        app.run(port=8000)
