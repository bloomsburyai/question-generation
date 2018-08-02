import os,time,json

# CUDA config
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
mem_limit=0.5

import tensorflow as tf
import numpy as np
from tqdm import tqdm

import flags

from langmodel.lm import LstmLm
from helpers.preprocessing import tokenise
from helpers import loader


FLAGS = tf.app.flags.FLAGS

def main(_):
    run_id=str(int(time.time()))
    chkpt_path = FLAGS.model_dir+'lm/'+run_id

    if not os.path.exists(chkpt_path):
        os.makedirs(chkpt_path)

    train_data = loader.load_squad_triples(FLAGS.data_path, False)
    dev_data = loader.load_squad_triples(FLAGS.data_path, True)

    np.random.shuffle(train_data)

    print('Loaded SQuAD with ',len(train_data),' triples')
    train_contexts, train_qs, train_as,train_a_pos = zip(*train_data)
    _, dev_qs, _,_ = zip(*dev_data)
    vocab = loader.get_vocab(train_qs, tf.app.flags.FLAGS.lm_vocab_size)

    with open(chkpt_path+'/vocab.json', 'w') as outfile:
        json.dump(vocab, outfile)

    unique_sents = list(set(train_qs))
    print(len(unique_sents)," unique sentences")

    # Create model

    model = LstmLm(vocab, num_units=FLAGS.lm_units)
    with model.graph.as_default():
        saver = tf.train.Saver()


    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_limit)
    with tf.Session(graph = model.graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if not os.path.exists(chkpt_path):
            os.makedirs(chkpt_path)
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir+'lm/'+run_id, sess.graph)

        if FLAGS.restore:
            # saver.restore(sess, chkpt_path+ '/model.checkpoint')
            print('Loading not implemented yet')
        else:
            print("Building graph, loading glove")
            sess.run(tf.global_variables_initializer())

        num_steps = len(unique_sents)//FLAGS.batch_size

        best_perp = 1e6

        for e in range(FLAGS.lm_num_epochs):
            np.random.shuffle(unique_sents)
            for i in tqdm(range(num_steps), desc='Epoch '+str(e)):
                seq_batch = unique_sents[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]

                seq_batch_ids = [[vocab[loader.SOS]]+[vocab[tok if tok in vocab.keys() else loader.OOV] for tok in tokenise(sent, asbytes=False)]+[vocab[loader.EOS]] for sent in seq_batch]
                max_seq_len = max([len(seq) for seq in seq_batch_ids])
                padded_batch = np.asarray([seq + [vocab[loader.PAD] for i in range(max_seq_len-len(seq))] for seq in seq_batch_ids])

                summ, _, pred, gold, seq = sess.run([model.train_summary, model.optimise, model.preds, model.tgt_output, model.input_seqs], feed_dict={model.input_seqs: padded_batch})
                summary_writer.add_summary(summ, global_step=(e*num_steps+i))

                # print(pred, gold, seq)
                # exit()

                # if i%FLAGS.eval_freq==0:
                #     saver.save(sess, chkpt_path+'/model.checkpoint')
                    # print(pred, gold, seq)

            perps=[]
            num_steps_dev = len(dev_qs)//FLAGS.batch_size
            for i in tqdm(range(num_steps_dev), desc="Eval"):
                seq_batch = dev_qs[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
                seq_batch_ids = [[vocab[loader.SOS]]+[vocab[tok if tok in vocab.keys() else loader.OOV] for tok in tokenise(sent, asbytes=False)]+[vocab[loader.EOS]] for sent in seq_batch]
                max_seq_len = max([len(seq) for seq in seq_batch_ids])
                padded_batch = np.asarray([seq + [vocab[loader.PAD] for i in range(max_seq_len-len(seq))] for seq in seq_batch_ids])

                perp = sess.run(model.perplexity, feed_dict={model.input_seqs: padded_batch})
                perps.extend(perp)

            perpsummary = tf.Summary(value=[tf.Summary.Value(tag="dev_perf/perplexity",
                                      simple_value=sum(perps)/len(perps))])

            summary_writer.add_summary(perpsummary, global_step=((e+1)*num_steps))

            if np.mean(perps) < best_perp:
                print(np.mean(perps), " Saving!")
                saver.save(sess, chkpt_path+'/model.checkpoint')
                best_perp = np.mean(perps)

if __name__ == '__main__':
    tf.app.run()
