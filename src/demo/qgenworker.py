import sys,os,traceback

import tensorflow as tf
import numpy as np



from helpers import preprocessing

import json

from demo.worker_flags import FLAGS

tf.app.flags.FLAGS = FLAGS

mem_limit=0.9


# model_slug_list = ['MALUUBA-CROP-SET-GLOVE/1535568489','SEQ2SEQ/1533280948','qgen-s2s-filt1']
model_slug_list = ['RL-S2S-1544356761']
model_slug_curr = model_slug_list[0]

from celery.signals import worker_init, worker_process_init
from celery.concurrency import asynpool
asynpool.PROC_ALIVE_TIMEOUT = 180.0 #set this long enough


BROKER_ENDPOINT = os.environ.get('BROKER_ENDPOINT', None)

BATCH_SIZE = 8

if BROKER_ENDPOINT is not None:
    from demo.instance import AQInstance
    from celery import Celery
    taskengine = Celery('tasks',
         broker='redis://' + BROKER_ENDPOINT +'/',
         backend='redis://' + BROKER_ENDPOINT +'/')
    taskengine.conf.update({'task_routes': {
             'demo.qgenworker.#': {'queue':'qgen'},
             'taskengine.#': {'queue': 'taskengine'}
         }})
    # init()
else:
    exit('BROKER_ENDPOINT not set!')


@worker_process_init.connect()
def init(**_):
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
    global taskengine
    taskengine.generator = AQInstance(vocab=vocab)
    taskengine.generator.load_from_chkpt(chkpt_path)



@taskengine.task
def get_q_batch_async(queries):
    print('Received qgen task')

    try:
    

        ctxts, ans, ans_pos = map(list, zip(*queries))

        if FLAGS.filter_window_size_before >-1:
            for i in range(len(ctxts)):
                ans_pos[i] = int(ans_pos[i])
                ctxts[i], ans_pos[i] = preprocessing.filter_context(ctxts[i], ans_pos[i], FLAGS.filter_window_size_before, FLAGS.filter_window_size_after, FLAGS.filter_max_tokens)
                
                ctxts[i] = ctxts[i].encode()
                ans[i] = ans[i].encode()

        print('Sending to model...')
        if len(ctxts) > BATCH_SIZE:
            qs = []
            for b in range(len(ctxts)//BATCH_SIZE + 1):
                start_ix = b * BATCH_SIZE
                end_ix = min(len(ctxts), (b + 1) * BATCH_SIZE)
                qs.extend(taskengine.generator.get_q_batch(ctxts[start_ix:end_ix], ans[start_ix:end_ix], ans_pos[start_ix:end_ix]))
        else:
            qs = taskengine.generator.get_q_batch(ctxts, ans, ans_pos)

        print('Done! Compiling results')
        resp = {'status': 'OK',
            'results': [{'q': qs[i], 'a': ans[i].decode()} for i in range(len(qs))]}
        return resp
    except Exception as e:
        print(e)
        traceback.print_exc()
        return str(e)

    
    