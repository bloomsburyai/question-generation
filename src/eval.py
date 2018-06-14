import os,time, json

# CUDA config
os.environ["CUDA_VISIBLE_DEVICES"]="2"
mem_limit=0.5

import tensorflow as tf
import numpy as np
import helpers.loader as loader
import helpers.metrics as metrics
from helpers.output import output_pretty, tokens_to_string
from tqdm import tqdm

from seq2seq_model import Seq2SeqModel

import flags

FLAGS = tf.app.flags.FLAGS

def main(_):
    chkpt_path = FLAGS.model_dir+'saved/qgen-maluuba'
    # chkpt_path = FLAGS.model_dir+'qgen/SEQ2SEQ/'+'1528886861'

    # load dataset
    train_data = loader.load_squad_triples(FLAGS.data_path, False)
    dev_data = loader.load_squad_triples(FLAGS.data_path, True)[:1500]

    print('Loaded SQuAD with ',len(train_data),' triples')
    print('Loaded SQuAD dev set with ',len(dev_data),' triples')
    train_contexts, train_qs, train_as,train_a_pos = zip(*train_data)
    dev_contexts, dev_qs, dev_as, dev_a_pos = zip(*dev_data)

    # vocab = loader.get_vocab(train_contexts, tf.app.flags.FLAGS.vocab_size)
    with open(chkpt_path+'/vocab.json') as f:
        vocab = json.load(f)
    with open(chkpt_path+'/lm_vocab.json') as f:
        lm_vocab = json.load(f)
    with open(chkpt_path+'/qa_vocab.json') as f:
        qa_vocab = json.load(f)

    # Create model

    model = Seq2SeqModel(vocab, batch_size=FLAGS.batch_size, training_mode=False)
    with model.graph.as_default():
        saver = tf.train.Saver()



    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_limit)
    with tf.Session(graph=model.graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if not os.path.exists(chkpt_path):
            exit('Checkpoint path doesnt exist! '+chkpt_path)
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir+str(int(time.time())), sess.graph)

        saver.restore(sess, chkpt_path+ '/model.checkpoint')
        # print('Loading not implemented yet')
        # else:
        #     sess.run(tf.global_variables_initializer())
        #     sess.run(model.glove_init_ops)

        num_steps = len(dev_data)//FLAGS.batch_size

        # Initialise the dataset
        sess.run(model.iterator.initializer, feed_dict={model.context_ph: dev_contexts,
                                          model.qs_ph: dev_qs, model.as_ph: dev_as, model.a_pos_ph: dev_a_pos})

        f1s=[]
        bleus=[]
        for e in range(1):
            for i in tqdm(range(num_steps), desc='Epoch '+str(e)):
                dev_batch, curr_batch_size = dev_data_source.get_batch()
                pred_batch,pred_lens,gold_batch, gold_lens= sess.run([model.q_hat_beam_string,model.q_hat_beam_lens,model.q_gold, model.question_length], feed_dict={model.input_batch: dev_batch ,model.is_training:False})

                out_str="<h1>"+str(e)+' - '+str(datetime.datetime.now())+'</h1>'
                for b, pred in enumerate(pred_batch):
                    pred_str = tokens_to_string(pred[:pred_lens[b]-1])
                    gold_str = tokens_to_string(gold_batch[b][:gold_lens[b]-1])
                    f1s.append(metrics.f1(gold_str, pred_str))
                    bleus.append(metrics.bleu(gold_str, pred_str))
                    out_str+=pred_str.replace('>','&gt;').replace('<','&lt;')+"<br/>"+gold_str.replace('>','&gt;').replace('<','&lt;')+"<hr/>"
                if i==0:
                    with open(FLAGS.log_dir+'out_eval_'+model_type+'.htm', 'w') as fp:
                        fp.write(out_str)

        print("F1: ", np.mean(f1s))
        print("BLEU: ", np.mean(bleus))

if __name__ == '__main__':
    tf.app.run()
