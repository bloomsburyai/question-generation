import os,time

# CUDA config
os.environ["CUDA_VISIBLE_DEVICES"]="2"
mem_limit=0.5

import tensorflow as tf
import numpy as np
from tqdm import tqdm

import flags

from qa.mpcm import MpcmQa
from helpers.preprocessing import tokenise, char_pos_to_word
from helpers import loader

def get_padded_batch(seq_batch, vocab):
    seq_batch_ids = [[vocab[loader.SOS]]+[vocab[tok if tok in vocab.keys() else loader.OOV] for tok in tokenise(sent, asbytes=False)]+[vocab[loader.EOS]] for sent in seq_batch]
    max_seq_len = max([len(seq) for seq in seq_batch_ids])
    padded_batch = np.asarray([seq + [vocab[loader.PAD] for i in range(max_seq_len-len(seq))] for seq in seq_batch_ids])
    return padded_batch

FLAGS = tf.app.flags.FLAGS

def main(_):
    train_data = loader.load_squad_triples(FLAGS.data_path, False)
    dev_data = loader.load_squad_triples(FLAGS.data_path, True)

    print('Loaded SQuAD with ',len(train_data),' triples')
    train_contexts, train_qs, train_as,train_a_pos = zip(*train_data)
    vocab = loader.get_vocab(train_qs, tf.app.flags.FLAGS.vocab_size)

    model = MpcmQa(vocab)
    saver = tf.train.Saver()

    chkpt_path = FLAGS.model_dir+'qa/'+str(int(time.time()))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_limit)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if not os.path.exists(chkpt_path):
            os.makedirs(chkpt_path)
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir+'qa/'+str(int(time.time())), sess.graph)

        if not FLAGS.train:
            # saver.restore(sess, chkpt_path+ '/model.checkpoint')
            print('Loading not implemented yet')
        else:
            sess.run(tf.global_variables_initializer())
            sess.run(model.glove_init_ops)

        num_steps = len(train_data)//FLAGS.batch_size

        for e in range(FLAGS.num_epochs):
            for i in tqdm(range(num_steps), desc='Epoch '+str(e)):
                # TODO: this keeps coming up - refactor it
                batch_contexts = train_contexts[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
                batch_questions = train_qs[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
                batch_ans_text = train_as[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
                batch_answer_charpos = train_a_pos[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]

                batch_answers=[]
                for j, ctxt in enumerate(batch_contexts):
                    ans_span=char_pos_to_word(ctxt.encode(), [t.encode() for t in tokenise(ctxt, asbytes=False)], batch_answer_charpos[j])
                    ans_span=(ans_span, ans_span+len(tokenise(batch_ans_text[j],asbytes=False)))
                    batch_answers.append(ans_span)

                # print(batch_answers[:3])
                # exit()

                _,summ = sess.run([model.optimise, model.train_summary],
                        feed_dict={model.context_in: get_padded_batch(batch_contexts,vocab),
                                model.question_in: get_padded_batch(batch_questions,vocab),
                                model.answer_spans_in: batch_answers,
                                model.is_training: True})

                summary_writer.add_summary(summ, global_step=(e*num_steps+i))

                if i%FLAGS.eval_freq==0:
                    saver.save(sess, chkpt_path+'/model.checkpoint')

if __name__ == '__main__':
    tf.app.run()
