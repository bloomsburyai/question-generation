import os,time, json

# CUDA config
os.environ["CUDA_VISIBLE_DEVICES"]="3"
mem_limit=0.95

import tensorflow as tf
import numpy as np
import helpers.loader as loader
from helpers.output import output_pretty, output_basic, tokens_to_string
from tqdm import tqdm

from seq2seq_model import Seq2SeqModel

import flags

import helpers.metrics as metrics



FLAGS = tf.app.flags.FLAGS

def main(_):
    # load dataset
    train_data = loader.load_squad_triples(FLAGS.data_path, False)
    dev_data = loader.load_squad_triples(FLAGS.data_path, True)

    print('Loaded SQuAD with ',len(train_data),' triples')
    train_contexts, train_qs, train_as,train_a_pos = zip(*train_data)
    vocab = loader.get_vocab(train_contexts, tf.app.flags.FLAGS.vocab_size)

    # Create model

    model = Seq2SeqModel(vocab, batch_size=FLAGS.batch_size, training_mode=True)
    saver = tf.train.Saver()

    chkpt_path = FLAGS.model_dir+'qgen/'+str(int(time.time()))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_limit)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if not os.path.exists(chkpt_path):
            os.makedirs(chkpt_path)
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir+'qgen/'+str(int(time.time())), sess.graph)

        if not FLAGS.train:
            # saver.restore(sess, chkpt_path+ '/model.checkpoint')
            print('Loading not implemented yet')
        else:
            sess.run(tf.global_variables_initializer())
            sess.run(model.glove_init_ops)

        num_steps = len(train_data)//FLAGS.batch_size

        # Initialise the dataset
        sess.run(model.iterator.initializer, feed_dict={model.context_ph: train_contexts,
                                          model.qs_ph: train_qs, model.as_ph: train_as, model.a_pos_ph: train_a_pos})

        for e in range(FLAGS.num_epochs):
            for i in tqdm(range(num_steps), desc='Epoch '+str(e)):
                ops = [model.optimizer, model.train_summary]
                if i%FLAGS.eval_freq==0:
                    ops.extend([model.q_hat_string, model.q_hat_ids, model.q_gold]) #, tf.squeeze(model.switch), model.q_hat_ids, model.question_ids,model.crossent * model.target_weights])
                res= sess.run(ops, feed_dict={model.is_training:True})
                summary_writer.add_summary(res[1], global_step=(e*num_steps+i))
                if i%FLAGS.eval_freq==0:
                    # summary_writer.add_summary(res[2], global_step=(e*num_steps+i))

                    # q_hat_decoded = output_pretty(res[3].tolist(), res[4].tolist(), res[5].tolist(), res[6].tolist(), res[7].tolist())
                    with open(FLAGS.log_dir+'out.htm', 'w') as fp:
                        fp.write(output_basic(res[2], res[3], e, i))

                    f1s=[]
                    bleus=[]
                    for b, pred in enumerate(res[2]):
                        pred_str = tokens_to_string(pred)
                        gold_str = tokens_to_string(res[4][b])
                        f1s.append(metrics.f1(gold_str, pred_str))
                        bleus.append(metrics.bleu(gold_str, pred_str))


                    f1summary = tf.Summary(value=[tf.Summary.Value(tag="train_perf/f1",
                                                     simple_value=sum(f1s)/len(f1s))])
                    bleusummary = tf.Summary(value=[tf.Summary.Value(tag="train_perf/bleu",
                                              simple_value=sum(bleus)/len(bleus))])

                    summary_writer.add_summary(f1summary, global_step=(e*num_steps+i))
                    summary_writer.add_summary(bleusummary, global_step=(e*num_steps+i))
                    # a_raw, a_str, q_str = sess.run([model.answer_raw,model.a_string, model.q_hat_string])
                    # print(a_raw.tolist(), a_str, q_str)
                    # print(sess.run([tf.shape(model.context_condition_encoding), tf.shape(model.full_condition_encoding)]))

                    # print(sess.run([model.answer_length, model.question_length]))
                    # print(sess.run([tf.shape(model.s0)]))
                # ToDo: implement dev pipeline
                # if i % FLAGS.eval_freq == 0:
                #     dev_summary = sess.run([model.eval_summary], feed_dict={x: batch_xs, y: batch_ys, is_training:False})
                #     summary_writer.add_summary(dev_summary, global_step=(e*num_steps+i))
                if i%FLAGS.eval_freq==0:
                    saver.save(sess, chkpt_path+'/model.checkpoint')



if __name__ == '__main__':
    tf.app.run()
