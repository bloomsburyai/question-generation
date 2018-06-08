import os,time, json

# CUDA config
os.environ["CUDA_VISIBLE_DEVICES"]="3"
mem_limit=0.95

import tensorflow as tf
import numpy as np
import helpers.loader as loader
import helpers.preprocessing as preprocessing
from helpers.output import output_pretty, output_basic, tokens_to_string
from tqdm import tqdm

from seq2seq_model import Seq2SeqModel
# from maluuba_model import MaluubaModel

from datasources.squad_streamer import SquadStreamer

import flags

import helpers.metrics as metrics

model_type = "SEQ2SEQ"
# model_type = "MALUUBA"

FLAGS = tf.app.flags.FLAGS

def main(_):
    if FLAGS.testing:
        print('TEST MODE - reducing model size')
        FLAGS.context_encoder_units =100
        FLAGS.answer_encoder_units=100
        FLAGS.decoder_units=100
        FLAGS.batch_size =2
        # FLAGS.embedding_size=50

    # load dataset
    train_data = loader.load_squad_triples(FLAGS.data_path, False)
    dev_data = loader.load_squad_triples(FLAGS.data_path, True)

    print('Loaded SQuAD with ',len(train_data),' triples')
    train_contexts, train_qs, train_as,train_a_pos = zip(*train_data)
    vocab = loader.get_vocab(train_contexts, tf.app.flags.FLAGS.vocab_size)

    ext_vocab = loader.get_vocab(train_contexts, tf.app.flags.FLAGS.lm_vocab_size)


    # Create model
    if model_type == "SEQ2SEQ":
        model = Seq2SeqModel(vocab, batch_size=FLAGS.batch_size, training_mode=True)
    elif model_type == "MALUUBA":
        model = MaluubaModel(vocab, ext_vocab, ext_vocab, batch_size=FLAGS.batch_size, training_mode=True)
    else:
        exit("Unrecognised model type: "+model_type)

    # create data streamer
    data_source = SquadStreamer(vocab, FLAGS.batch_size, shuffle=True)

    with model.graph.as_default():
        saver = tf.train.Saver()

    chkpt_path = FLAGS.model_dir+'qgen/'+str(int(time.time()))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_limit)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=model.graph) as sess:
        if not os.path.exists(chkpt_path):
            os.makedirs(chkpt_path)
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir+'qgen/'+str(int(time.time())), sess.graph)

        data_source.initialise(train_data)

        if not FLAGS.train:
            # saver.restore(sess, chkpt_path+ '/model.checkpoint')
            print('Loading not implemented yet')
        else:
            sess.run(tf.global_variables_initializer())
            # sess.run(model.glove_init_ops)

        num_steps = len(train_data)//FLAGS.batch_size

        # Initialise the dataset
        # sess.run(model.iterator.initializer, feed_dict={model.context_ph: train_contexts,
        #                                   model.qs_ph: train_qs, model.as_ph: train_as, model.a_pos_ph: train_a_pos})


        for e in range(FLAGS.num_epochs):
            for i in tqdm(range(num_steps), desc='Epoch '+str(e)):
                # Get a batch
                train_batch, curr_batch_size = data_source.get_batch()

                if model_type == "MALUUBA":
                    # do a fwd pass first, get the score, then do another pass and optimize
                    res= sess.run(model.q_hat_string, feed_dict={model.input_batch: train_batch ,model.is_training:True})
                    qhat_for_lm = [preprocessing.lookup_vocab(q, ext_vocab, do_tokenise=False) for q in res.tolist()]
                    lm_score = model.lm.get_seq_prob(qhat_for_lm).tolist()
                    lm_summary = tf.Summary(value=[tf.Summary.Value(tag="rl_rewards/lm",
                                                     simple_value=sum(lm_score)/len(lm_score))])
                    summary_writer.add_summary(lm_summary, global_step=(e*num_steps+i))
                    # print(res)
                    # print(lm_score)
                    rl_dict={model.lm_score: lm_score,
                    model.qa_score: [0. for i in range(curr_batch_size)],
                    model.rl_lm_enabled: True,
                    model.rl_qa_enabled: False}
                else:
                    rl_dict={}

                ops = [model.optimizer, model.train_summary,model.q_hat_string]
                if i%FLAGS.eval_freq==0:
                    ops.extend([ model.q_hat_ids, model.question_ids, model.copy_prob, model.q_gold])
                res= sess.run(ops, feed_dict={model.input_batch: train_batch,
                    model.is_training:True,
                    **rl_dict})
                summary_writer.add_summary(res[1], global_step=(e*num_steps+i))




                if i%FLAGS.eval_freq==0:
                    # summary_writer.add_summary(res[2], global_step=(e*num_steps+i))

                    # q_hat_decoded = output_pretty(res[3].tolist(), res[4].tolist(), res[5].tolist(), res[6].tolist(), res[7].tolist())
                    with open(FLAGS.log_dir+'out.htm', 'w') as fp:
                        fp.write(output_pretty(res[2].tolist(), res[3], res[4], res[5], e, i))

                    f1s=[]
                    bleus=[]
                    for b, pred in enumerate(res[2]):
                        pred_str = tokens_to_string(pred)
                        gold_str = tokens_to_string(res[6][b])
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
