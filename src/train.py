import os,time, json

# model_type = "SEQ2SEQ"
model_type = "MALUUBA"

# CUDA config
os.environ["CUDA_VISIBLE_DEVICES"] = "1" if model_type == "MALUUBA" else "3"
mem_limit=0.95

import tensorflow as tf
import numpy as np
import helpers.loader as loader
import helpers.preprocessing as preprocessing
from helpers.output import output_pretty, output_basic, tokens_to_string
from tqdm import tqdm

from seq2seq_model import Seq2SeqModel
from maluuba_model import MaluubaModel

from datasources.squad_streamer import SquadStreamer

import flags

import helpers.metrics as metrics



FLAGS = tf.app.flags.FLAGS

def main(_):
    if FLAGS.testing:
        print('TEST MODE - reducing model size')
        FLAGS.context_encoder_units =100
        FLAGS.answer_encoder_units=100
        FLAGS.decoder_units=100
        FLAGS.batch_size =8
        # FLAGS.embedding_size=50

    # load dataset
    train_data = loader.load_squad_triples(FLAGS.data_path, False)
    dev_data = loader.load_squad_triples(FLAGS.data_path, True)

    print('Loaded SQuAD with ',len(train_data),' triples')
    train_contexts, train_qs, train_as,train_a_pos = zip(*train_data)
    vocab = loader.get_vocab(train_contexts, tf.app.flags.FLAGS.vocab_size)

    lm_vocab = loader.get_vocab(train_contexts, tf.app.flags.FLAGS.lm_vocab_size)
    qa_vocab = loader.get_vocab(train_contexts, tf.app.flags.FLAGS.qa_vocab_size)

    if FLAGS.testing:
        train_data=train_data[:1000]
        num_dev_samples=100
    else:
        num_dev_samples=1000

    # Create model
    if model_type == "SEQ2SEQ":
        model = Seq2SeqModel(vocab, batch_size=FLAGS.batch_size, training_mode=True)
    elif model_type == "MALUUBA":
        # TEMP
        FLAGS.qa_weight = 0
        FLAGS.lm_weight = 0
        model = MaluubaModel(vocab, lm_vocab, qa_vocab, batch_size=FLAGS.batch_size, training_mode=True, lm_weight=FLAGS.lm_weight, qa_weight=FLAGS.qa_weight)
    else:
        exit("Unrecognised model type: "+model_type)

    # create data streamer
    train_data_source = SquadStreamer(vocab, FLAGS.batch_size, FLAGS.num_epochs, shuffle=True)
    dev_data_source = SquadStreamer(vocab, FLAGS.batch_size, 1, shuffle=True)

    with model.graph.as_default():
        saver = tf.train.Saver()

    chkpt_path = FLAGS.model_dir+'qgen/'+model_type+'/'+str(int(time.time()))

    # change visible devices if using RL models
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_limit, visible_device_list='0',allow_growth = True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=False), graph=model.graph) as sess:
        if not os.path.exists(chkpt_path):
            os.makedirs(chkpt_path)
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir+'qgen/'+model_type+'/'+str(int(time.time())), sess.graph)

        train_data_source.initialise(train_data)


        if not FLAGS.train:
            # saver.restore(sess, chkpt_path+ '/model.checkpoint')
            print('Loading not implemented yet')
        else:
            sess.run(tf.global_variables_initializer())
            # sess.run(model.glove_init_ops)

        num_steps_train = len(train_data)//FLAGS.batch_size
        num_steps_dev = num_dev_samples//FLAGS.batch_size

        # Initialise the dataset
        # sess.run(model.iterator.initializer, feed_dict={model.context_ph: train_contexts,
        #                                   model.qs_ph: train_qs, model.as_ph: train_as, model.a_pos_ph: train_a_pos})

        f1summary = tf.Summary(value=[tf.Summary.Value(tag="dev_perf/f1",
                                         simple_value=0.0)])
        bleusummary = tf.Summary(value=[tf.Summary.Value(tag="dev_perf/bleu",
                                  simple_value=0.0)])

        summary_writer.add_summary(f1summary, global_step=0)
        summary_writer.add_summary(bleusummary, global_step=0)

        max_oos_f1=0
        perform_policy_gradient = False # update this during training

        for e in range(FLAGS.num_epochs):
            for i in tqdm(range(num_steps_train), desc='Epoch '+str(e)):
                # Get a batch
                train_batch, curr_batch_size = train_data_source.get_batch()

                if model_type == "MALUUBA" and perform_policy_gradient:
                    # do a fwd pass first, get the score, then do another pass and optimize
                    res= sess.run(model.q_hat_beam_string, feed_dict={model.input_batch: train_batch ,model.is_training:True})
                    qhat_for_lm = [preprocessing.lookup_vocab(q, lm_vocab, do_tokenise=False) for q in res.tolist()]
                    ctxt_for_lm = [preprocessing.lookup_vocab(ctxt, lm_vocab, do_tokenise=False) for ctxt in train_batch[0][0].tolist()]
                    qhat_for_qa = [preprocessing.lookup_vocab(q, qa_vocab, do_tokenise=False) for q in res.tolist()]
                    ctxt_for_qa = [preprocessing.lookup_vocab(ctxt, qa_vocab, do_tokenise=False) for ctxt in train_batch[0][0].tolist()]

                    lm_score = model.lm.get_seq_prob(qhat_for_lm).tolist()
                    lm_summary = tf.Summary(value=[tf.Summary.Value(tag="rl_rewards/lm",
                                                     simple_value=sum(lm_score)/len(lm_score))])
                    summary_writer.add_summary(lm_summary, global_step=(e*num_steps_train+i))

                    qa_pred = model.qa.get_ans(ctxt_for_qa, qhat_for_qa).tolist()

                    gold_str=[]
                    pred_str=[]
                    qa_f1s = []

                    for b in range(FLAGS.batch_size):
                        gold_str.append(" ".join([w.decode() for w in train_batch[2][0][b][:train_batch[2][2][b]-1].tolist()]))
                        pred_str.append(" ".join([w.decode() for w in train_batch[0][0][b].tolist()[qa_pred[b][0]:qa_pred[b][1]]]) )
                    if i == 0:
                        print(gold_str[0])
                        print(pred_str[0])
                        print(qa_pred[0])
                    qa_f1s.extend([metrics.f1(gold_str[b], pred_str[b]) for b in range(FLAGS.batch_size)])

                    qa_summary = tf.Summary(value=[tf.Summary.Value(tag="rl_rewards/qa",
                                                     simple_value=sum(qa_f1s)/len(qa_f1s))])
                    summary_writer.add_summary(qa_summary, global_step=(e*num_steps_train+i))

                    rl_dict={model.lm_score: lm_score,
                    model.qa_score: qa_f1s,
                    model.rl_lm_enabled: True,
                    model.rl_qa_enabled: True}
                elif model_type == "MALUUBA" and not perform_policy_gradient:
                    rl_dict={model.lm_score: [0 for b in range(curr_batch_size)],
                    model.qa_score: [0 for b in range(curr_batch_size)],
                    model.rl_lm_enabled: False,
                    model.rl_qa_enabled: False}
                else:
                    rl_dict={}

                ops = [model.optimizer, model.train_summary,model.q_hat_string]
                if i%FLAGS.eval_freq==0:
                    ops.extend([ model.q_hat_ids, model.question_ids, model.copy_prob, model.q_gold])
                res= sess.run(ops, feed_dict={model.input_batch: train_batch,
                    model.is_training:True,
                    **rl_dict})
                summary_writer.add_summary(res[1], global_step=(e*num_steps_train+i))




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

                    summary_writer.add_summary(f1summary, global_step=(e*num_steps_train+i))
                    summary_writer.add_summary(bleusummary, global_step=(e*num_steps_train+i))

            f1s=[]
            bleus=[]

            np.random.shuffle(dev_data)
            dev_subset = dev_data[:num_dev_samples]
            dev_data_source.initialise(dev_subset)
            for i in tqdm(range(num_steps_dev), desc='Eval '+str(e)):
                dev_batch, curr_batch_size = dev_data_source.get_batch()
                pred_batch,gold_batch= sess.run([model.q_hat_beam_string,model.q_gold], feed_dict={model.input_batch: train_batch ,model.is_training:False})

                out_str=""
                for b, pred in enumerate(pred_batch):
                    pred_str = tokens_to_string(pred)
                    gold_str = tokens_to_string(gold_batch[b])
                    f1s.append(metrics.f1(gold_str, pred_str))
                    bleus.append(metrics.bleu(gold_str, pred_str))
                    out_str+=pred_str.replace('>','&gt;').replace('<','&lt;')+"<br/>"+gold_str.replace('>','&gt;').replace('<','&lt;')+"<hr/>"
                if i==0:
                    with open(FLAGS.log_dir+'out_eval.htm', 'w') as fp:
                        fp.write(out_str)

            f1summary = tf.Summary(value=[tf.Summary.Value(tag="dev_perf/f1",
                                             simple_value=sum(f1s)/len(f1s))])
            bleusummary = tf.Summary(value=[tf.Summary.Value(tag="dev_perf/bleu",
                                      simple_value=sum(bleus)/len(bleus))])

            summary_writer.add_summary(f1summary, global_step=((e+1)*num_steps_train))
            summary_writer.add_summary(bleusummary, global_step=((e+1)*num_steps_train))

            mean_f1=sum(f1s)/len(f1s)
            if mean_f1 > max_oos_f1:
                print("New best F1! ", mean_f1, " Saving...")
                max_oos_f1 = mean_f1
                saver.save(sess, chkpt_path+'/model.checkpoint')
            else:
                print("F1 not improved ", mean_f1)
if __name__ == '__main__':
    tf.app.run()
