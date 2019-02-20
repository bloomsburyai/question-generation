import sys,os

import tensorflow as tf
import numpy as np

from qanet.instance import QANetInstance

from helpers import preprocessing

import json

import flags
FLAGS = tf.app.flags.FLAGS

mem_limit=0.9


from flask import Flask, current_app, request, redirect

app = Flask(__name__)

@app.route("/")
def hello():
    return "The QANet service is running"


@app.route("/api/get_answers_batch", methods=['POST'])
def get_q_batch():
    args = request.get_json()

    ctxts, questions = map(list, zip(*args['queries']))

    
    if len(ctxts) > 32:
        ans = []
        for b in range(len(ctxts)//32 + 1):
            start_ix = b * 32
            end_ix = min(len(ctxts), (b + 1) * 32)
            ans.extend(current_app.generator.get_ans(ctxts[start_ix:end_ix], questions[start_ix:end_ix]))
    else:
        ans = current_app.generator.get_ans(ctxts, questions)

    resp = {'status': 'OK',
            'results': [{'a': ans[i]} for i in range(len(ans))]}
    return json.dumps(resp)

def init():
    print('Spinning up QANet service')

    # Note that this is irrelevant... it's set in config.py
    chkpt_path = FLAGS.model_dir+'qanet/'


    app.generator = QANetInstance()
    app.generator.load_from_chkpt(chkpt_path)

if __name__ == '__main__':
    init()
    with app.app_context():
        app.run(host="0.0.0.0", port=5005, processes=1)