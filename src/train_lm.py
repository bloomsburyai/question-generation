import os,time

# CUDA config
os.environ["CUDA_VISIBLE_DEVICES"]="3"
mem_limit=0.95

import tensorflow as tf
import numpy as np
from tqdm import tqdm

import flags

from langmodel.lm import LstmLm
from helpers.preprocessing import tokenise
from helpers import loader


FLAGS = tf.app.flags.FLAGS

def main(_):
    train_data = loader.load_squad_triples(FLAGS.data_path, False)
    dev_data = loader.load_squad_triples(FLAGS.data_path, True)

    print('Loaded SQuAD with ',len(train_data),' triples')
    train_contexts, train_qs, train_as,train_a_pos = zip(*train_data)
    vocab = loader.get_vocab(train_contexts, tf.app.flags.FLAGS.vocab_size)

    unique_contexts = list(set(train_contexts))
    print(len(unique_contexts)," unique contexts")

    # Create model

    model = LstmLm(vocab, num_units=512)
    saver = tf.train.Saver()

    chkpt_path = FLAGS.model_dir+'lm/'+str(int(time.time()))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_limit)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if not os.path.exists(chkpt_path):
            os.makedirs(chkpt_path)
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir+'lm/'+str(int(time.time())), sess.graph)

        if not FLAGS.train:
            # saver.restore(sess, chkpt_path+ '/model.checkpoint')
            print('Loading not implemented yet')
        else:
            sess.run(tf.global_variables_initializer())
            sess.run(model.glove_init_ops)

        num_steps = len(unique_contexts)//FLAGS.batch_size

        for e in range(FLAGS.num_epochs):
            for i in tqdm(range(num_steps), desc='Epoch '+str(e)):
                seq_batch = unique_contexts[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]

                seq_batch_ids = [[vocab[loader.SOS]]+[vocab[tok if tok in vocab.keys() else loader.OOV] for tok in tokenise(sent, asbytes=False)]+[vocab[loader.EOS]] for sent in seq_batch]
                max_seq_len = max([len(seq) for seq in seq_batch_ids])
                padded_batch = np.asarray([seq + [vocab[loader.PAD] for i in range(max_seq_len-len(seq))] for seq in seq_batch_ids])

                summ, _ = sess.run([model.train_summary, model.optimise], feed_dict={model.input_seqs: padded_batch})
                summary_writer.add_summary(summ, global_step=(e*num_steps+i))


                if i%FLAGS.eval_freq==0:
                    saver.save(sess, chkpt_path+'/model.checkpoint')



if __name__ == '__main__':
    tf.app.run()
