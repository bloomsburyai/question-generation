import os,time, json

# CUDA config
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
mem_limit=0.5

import tensorflow as tf
import numpy as np
import helpers.loader as loader
from helpers.output import output_pretty
from tqdm import tqdm

from seq2seq_model import Seq2SeqModel




# config
tf.app.flags.DEFINE_boolean("train", True, "Training mode?")
tf.app.flags.DEFINE_integer("eval_freq", 100, "Evaluate the model after this many steps")
tf.app.flags.DEFINE_integer("num_epochs", 20, "Train the model for this many epochs")
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size")
tf.app.flags.DEFINE_string("data_path", '../data/', "Path to dataset")
tf.app.flags.DEFINE_string("log_dir", './logs/', "Path to logs")

# hyperparams - these should probably be within the model?
tf.app.flags.DEFINE_integer("embedding_size", 2**7, "Dimensionality to use for learned word embeddings")
tf.app.flags.DEFINE_integer("context_encoder_units", 768, "Number of hidden units for context encoder (ie 1st stage)")
tf.app.flags.DEFINE_integer("answer_encoder_units", 768, "Number of hidden units for answer encoder (ie 2nd stage)")
tf.app.flags.DEFINE_integer("decoder_units", 768, "Number of hidden units for decoder")
tf.app.flags.DEFINE_integer("vocab_size", 2000, "Shortlist vocab size")
tf.app.flags.DEFINE_float("learning_rate", 2e-4, "Optimizer learning rate")
tf.app.flags.DEFINE_float("dropout_rate", 0.3, "Dropout probability")

FLAGS = tf.app.flags.FLAGS

def main(_):
    # load dataset
    train_data = loader.load_squad_triples(FLAGS.data_path, False)
    dev_data = loader.load_squad_triples(FLAGS.data_path, True)

    print('Loaded SQuAD with ',len(train_data),' triples')
    train_contexts, train_qs, train_as,train_a_pos = zip(*train_data)
    vocab = loader.get_vocab(train_contexts, tf.app.flags.FLAGS.vocab_size)

    # Create model

    model = Seq2SeqModel(vocab, batch_size=FLAGS.batch_size)
    # saver = tf.train.Saver()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_limit)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # if not os.path.exists(FLAGS.log_dir+str(time.time())):
        #     os.makedirs(FLAGS.log_dir+str(time.time()))
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir+str(int(time.time())), sess.graph)

        if not FLAGS.train:
            # saver.restore(sess, chkpt_path+ '/model.checkpoint')
            print('Loading not implemented yet')
        else:
            sess.run(tf.global_variables_initializer())

        num_steps = len(train_data)//FLAGS.batch_size

        # Initialise the dataset
        sess.run(model.iterator.initializer, feed_dict={model.context_ph: train_contexts,
                                          model.qs_ph: train_qs, model.as_ph: train_as, model.a_pos_ph: train_a_pos})

        for e in range(FLAGS.num_epochs):
            for i in tqdm(range(num_steps), desc='Epoch '+str(e)):
                ops = [model.optimizer, model.train_summary]
                if i %100== 0:
                    ops.extend([model.output_summary, model.q_hat_string, tf.squeeze(model.switch), model.q_hat_ids, model.question_ids,model.crossent * model.target_weights])
                res= sess.run(ops, feed_dict={model.is_training:True})
                summary_writer.add_summary(res[1], global_step=(e*num_steps+i))
                if i %100== 0:
                    # summary_writer.add_summary(res[2], global_step=(e*num_steps+i))

                    q_hat_decoded = output_pretty(res[3].tolist(), res[4].tolist(), res[5].tolist(), res[6].tolist(), res[7].tolist())
                    with open(FLAGS.log_dir+'out.htm', 'w') as fp:
                        fp.write(q_hat_decoded)
                    # a_raw, a_str, q_str = sess.run([model.answer_raw,model.a_string, model.q_hat_string])
                    # print(a_raw.tolist(), a_str, q_str)
                    # print(sess.run([tf.shape(model.context_condition_encoding), tf.shape(model.full_condition_encoding)]))

                    # print(sess.run([model.answer_length, model.question_length]))
                    # print(sess.run([tf.shape(model.s0)]))
                # ToDo: implement dev pipeline
                # if i % FLAGS.eval_freq == 0:
                #     dev_summary = sess.run([model.eval_summary], feed_dict={x: batch_xs, y: batch_ys, is_training:False})
                #     summary_writer.add_summary(dev_summary, global_step=(e*num_steps+i))
                # if save_cond:
                #     saver.save(sess, chkpt_path+'/model.checkpoint')



if __name__ == '__main__':
    tf.app.run()
