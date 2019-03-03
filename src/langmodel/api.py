import sys,os

import tensorflow as tf
import numpy as np

from lm import LstmLmInstance

from helpers import preprocessing

import json

import flags
FLAGS = tf.app.flags.FLAGS

mem_limit=0.9


from flask import Flask, current_app, request, redirect

app = Flask(__name__)

@app.route("/")
def hello():
    return "The LangModel service is running"


@app.route("/api/get_log_prob", methods=['POST'])
def get_log_prob():
    args = request.get_json()

    questions = args['queries']

    
    if len(questions) > 32:
        log_probs = []
        for b in range(len(questions)//32 + 1):
            start_ix = b * 32
            end_ix = min(len(questions), (b + 1) * 32)
            log_probs.extend(current_app.generator.get_seq_perplexity(questions[start_ix:end_ix]))
    else:
        log_probs = current_app.generator.get_seq_perplexity(questions)

    resp = {'status': 'OK',
            'results': [{'log_probs': str(log_probs[i])} for i in range(len(log_probs))]}
    return json.dumps(resp)

def init():
    print('Spinning up LangModel service')

    # Note that this is irrelevant... it's set in config.py
    chkpt_path = FLAGS.model_dir+'lm'


    app.generator = LstmLmInstance()
    app.generator.load_from_chkpt(chkpt_path)

if __name__ == '__main__':
    init()
    with app.app_context():
        app.run(host="0.0.0.0", port=5006, processes=1)