import os,time,json

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

from helpers.metrics import f1

def get_padded_batch(seq_batch, vocab):
    seq_batch_ids = [[vocab[loader.SOS]]+[vocab[tok if tok in vocab.keys() else loader.OOV] for tok in tokenise(sent, asbytes=False)]+[vocab[loader.EOS]] for sent in seq_batch]
    max_seq_len = max([len(seq) for seq in seq_batch_ids])
    padded_batch = np.asarray([seq + [vocab[loader.PAD] for i in range(max_seq_len-len(seq))] for seq in seq_batch_ids])
    return padded_batch

FLAGS = tf.app.flags.FLAGS

def main(_):
    train_data = loader.load_squad_triples(FLAGS.data_path, False)
    dev_data = loader.load_squad_triples(FLAGS.data_path, True)[:500]

    chkpt_path = FLAGS.model_dir+'saved/qatest'
    # chkpt_path = FLAGS.model_dir+'qa/1528885583'

    print('Loaded SQuAD with ',len(train_data),' triples')
    train_contexts, train_qs, train_as,train_a_pos = zip(*train_data)
    dev_contexts, dev_qs, dev_as, dev_a_pos = zip(*dev_data)

    # vocab = loader.get_vocab(train_contexts, tf.app.flags.FLAGS.qa_vocab_size)
    with open(chkpt_path+'/vocab.json') as f:
        vocab = json.load(f)

    model = MpcmQa(vocab, training_mode=False)
    with model.graph.as_default():
        saver = tf.train.Saver()





    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_limit)
    with tf.Session(graph=model.graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if not os.path.exists(chkpt_path):
            os.makedirs(chkpt_path)
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir+'qa/'+str(int(time.time())), sess.graph)

        saver.restore(sess, chkpt_path+ '/model.checkpoint')


        num_steps = len(dev_data)//FLAGS.batch_size

        f1s = []
        exactmatches= []
        for e in range(1):
            np.random.shuffle(train_data)
            train_contexts, train_qs, train_as,train_a_pos = zip(*train_data)
            for i in tqdm(range(num_steps), desc='Epoch '+str(e)):
                # TODO: this keeps coming up - refactor it
                batch_contexts = dev_contexts[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
                batch_questions = dev_qs[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
                batch_ans_text = dev_as[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
                batch_answer_charpos = dev_a_pos[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]

                batch_answers=[]
                for j, ctxt in enumerate(batch_contexts):
                    ans_span=char_pos_to_word(ctxt.encode(), [t.encode() for t in tokenise(ctxt, asbytes=False)], batch_answer_charpos[j])
                    ans_span=(ans_span, ans_span+len(tokenise(batch_ans_text[j],asbytes=False)))
                    batch_answers.append(ans_span)

                # print(batch_answers[:3])
                # exit()

                summ, pred = sess.run([model.eval_summary, model.pred_span],
                        feed_dict={model.context_in: get_padded_batch(batch_contexts,vocab),
                                model.question_in: get_padded_batch(batch_questions,vocab),
                                model.answer_spans_in: batch_answers,
                                model.is_training: False})

                summary_writer.add_summary(summ, global_step=(e*num_steps+i))

                gold_str=[]
                pred_str=[]
                for b in range(FLAGS.batch_size):
                    gold_str.append(" ".join(tokenise(batch_contexts[b],asbytes=False)[batch_answers[b][0]:batch_answers[b][1]]))
                    pred_str.append( " ".join(tokenise(batch_contexts[b],asbytes=False)[pred[b][0]:pred[b][1]]) )

                f1s.extend([f1(gold_str[b], pred_str[b]) for b in range(FLAGS.batch_size)])
                exactmatches.extend([ np.product(pred[b] == batch_answers[b])*1.0 for b in range(FLAGS.batch_size) ])

                if i%FLAGS.eval_freq==0:
                    out_str="<h1>" + "Eval - Dev set" + "</h1>"
                    for b in range(FLAGS.batch_size):
                        out_str += batch_contexts[b] + '<br/>'
                        out_str += batch_questions[b] + '<br/>'
                        out_str += str(batch_answers[b])+ str(tokenise(batch_contexts[b],asbytes=False)[batch_answers[b][0]:batch_answers[b][1]]) + '<br/>'
                        out_str += str(pred[b]) + str(tokenise(batch_contexts[b],asbytes=False)[pred[b][0]:pred[b][1]]) + '<br/>'
                        out_str += batch_ans_text[b] + '<br/>'
                        out_str += pred_str[b] + '<br/>'
                        out_str += "F1: " + str(f1(gold_str[b], pred_str[b])) + '<br/>'
                        out_str += "EM: " + str(np.product(pred[b] == batch_answers[b])*1.0)
                        out_str += "<hr/>"
                    with open(FLAGS.log_dir+'out_qa_eval.htm', 'w') as fp:
                        fp.write(out_str)
        print("F1: ", np.mean(f1s))
        print("EM: ", np.mean(exactmatches))

if __name__ == '__main__':
    tf.app.run()
