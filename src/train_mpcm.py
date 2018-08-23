import os,time,datetime,json

# CUDA config
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
mem_limit=0.95

import tensorflow as tf
import numpy as np
from tqdm import tqdm

import flags

from qa.mpcm import MpcmQa
from helpers.preprocessing import tokenise, char_pos_to_word, filter_squad
from helpers import loader
from helpers.metrics import f1, normalize_answer


def get_padded_batch(seq_batch, vocab):
    seq_batch_ids = [[vocab[loader.SOS]]+[vocab[tok if tok in vocab.keys() else loader.OOV] for tok in tokenise(sent, asbytes=False)]+[vocab[loader.EOS]] for sent in seq_batch]
    max_seq_len = max([len(seq) for seq in seq_batch_ids])
    padded_batch = np.asarray([seq + [vocab[loader.PAD] for i in range(max_seq_len-len(seq))] for seq in seq_batch_ids])
    return padded_batch

FLAGS = tf.app.flags.FLAGS

def main(_):
    if FLAGS.testing:
        print('TEST MODE - reducing model size')
        FLAGS.qa_encoder_units =32
        FLAGS.qa_match_units=32
        FLAGS.qa_batch_size =16
        FLAGS.embedding_size=50

    run_id = str(int(time.time()))

    chkpt_path = FLAGS.model_dir+'qa/'+run_id
    restore_path=FLAGS.model_dir+'qa/1529056867'

    if not os.path.exists(chkpt_path):
        os.makedirs(chkpt_path)

    train_data = loader.load_squad_triples(FLAGS.data_path, False)
    dev_data = loader.load_squad_triples(FLAGS.data_path, dev=True, ans_list=True)

    train_data = filter_squad(train_data, window_size=FLAGS.filter_window_size, max_tokens=FLAGS.filter_max_tokens)
    # dev_data = filter_squad(dev_data, window_size=FLAGS.filter_window_size, max_tokens=FLAGS.filter_max_tokens)

    if FLAGS.testing:
        train_data=train_data[:1000]
        num_dev_samples=100
    else:
        num_dev_samples=3000

    print('Loaded SQuAD with ',len(train_data),' triples')
    train_contexts, train_qs, train_as,train_a_pos = zip(*train_data)
    dev_contexts, dev_qs, dev_as,dev_a_pos = zip(*dev_data)

    if FLAGS.restore:
        with open(restore_path+'/vocab.json') as f:
            vocab = json.load(f)
    else:
        vocab = loader.get_vocab(train_contexts+train_qs, tf.app.flags.FLAGS.qa_vocab_size)
        with open(chkpt_path+'/vocab.json', 'w') as outfile:
            json.dump(vocab, outfile)



    model = MpcmQa(vocab)
    with model.graph.as_default():
        saver = tf.train.Saver()



    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_limit, allow_growth = True)
    with tf.Session(graph=model.graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        summary_writer = tf.summary.FileWriter(FLAGS.log_dir+'qa/'+run_id, sess.graph)

        if FLAGS.restore:
            saver.restore(sess, restore_path+ '/model.checkpoint')
            start_e=40#FLAGS.qa_num_epochs
            print('Loaded model')
        else:
            print("Building graph, loading glove")
            start_e=0
            sess.run(tf.global_variables_initializer())

        num_steps_train = len(train_data)//FLAGS.qa_batch_size
        num_steps_dev = num_dev_samples//FLAGS.qa_batch_size

        f1summary = tf.Summary(value=[tf.Summary.Value(tag="dev_perf/f1",
                                         simple_value=0.0)])
        emsummary = tf.Summary(value=[tf.Summary.Value(tag="dev_perf/em",
                                  simple_value=0.0)])

        summary_writer.add_summary(f1summary, global_step=start_e*num_steps_train)
        summary_writer.add_summary(emsummary, global_step=start_e*num_steps_train)

        best_oos_nll=1e6

        for e in range(start_e,start_e+FLAGS.qa_num_epochs):
            np.random.shuffle(train_data)
            train_contexts, train_qs, train_as,train_a_pos = zip(*train_data)

            for i in tqdm(range(num_steps_train), desc='Epoch '+str(e)):
                # TODO: this keeps coming up - refactor it
                batch_contexts = train_contexts[i*FLAGS.qa_batch_size:(i+1)*FLAGS.qa_batch_size]
                batch_questions = train_qs[i*FLAGS.qa_batch_size:(i+1)*FLAGS.qa_batch_size]
                batch_ans_text = train_as[i*FLAGS.qa_batch_size:(i+1)*FLAGS.qa_batch_size]
                batch_answer_charpos = train_a_pos[i*FLAGS.qa_batch_size:(i+1)*FLAGS.qa_batch_size]

                batch_answers=[]
                for j, ctxt in enumerate(batch_contexts):
                    ans_span=char_pos_to_word(ctxt.encode(), [t.encode() for t in tokenise(ctxt, asbytes=False)], batch_answer_charpos[j])
                    ans_span=(ans_span, ans_span+len(tokenise(batch_ans_text[j],asbytes=False))-1)
                    batch_answers.append(ans_span)

                # print(batch_answers[:3])
                # exit()
                # run_metadata = tf.RunMetadata()
                # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                _,summ, pred = sess.run([model.optimizer, model.train_summary, model.pred_span],
                        feed_dict={model.context_in: get_padded_batch(batch_contexts,vocab),
                                model.question_in: get_padded_batch(batch_questions,vocab),
                                model.answer_spans_in: batch_answers,
                                model.is_training: True})
                                # ,run_metadata=run_metadata, options=run_options)

                summary_writer.add_summary(summ, global_step=(e*num_steps_train+i))
                # summary_writer.add_run_metadata(run_metadata, tag="step "+str(i), global_step=(e*num_steps_train+i))

                if i%FLAGS.eval_freq==0:
                    gold_str=[]
                    pred_str=[]
                    f1s = []
                    exactmatches= []
                    for b in range(FLAGS.qa_batch_size):
                        gold_str.append(" ".join(tokenise(batch_contexts[b],asbytes=False)[batch_answers[b][0]:batch_answers[b][1]+1]))
                        pred_str.append( " ".join(tokenise(batch_contexts[b],asbytes=False)[pred[b][0]:pred[b][1]+1]) )

                    f1s.extend([f1(gold_str[b], pred_str[b]) for b in range(FLAGS.qa_batch_size)])
                    exactmatches.extend([ np.product(pred[b] == batch_answers[b])*1.0 for b in range(FLAGS.qa_batch_size) ])

                    f1summary = tf.Summary(value=[tf.Summary.Value(tag="train_perf/f1",
                                                     simple_value=sum(f1s)/len(f1s))])
                    emsummary = tf.Summary(value=[tf.Summary.Value(tag="train_perf/em",
                                              simple_value=sum(exactmatches)/len(exactmatches))])

                    summary_writer.add_summary(f1summary, global_step=(e*num_steps_train+i))
                    summary_writer.add_summary(emsummary, global_step=(e*num_steps_train+i))


                    # saver.save(sess, chkpt_path+'/model.checkpoint')


            f1s=[]
            exactmatches=[]
            nlls=[]

            np.random.shuffle(dev_data)
            dev_subset = dev_data[:num_dev_samples]
            for i in tqdm(range(num_steps_dev), desc='Eval '+str(e)):
                dev_contexts,dev_qs,dev_as,dev_a_pos = zip(*dev_subset)
                batch_contexts = dev_contexts[i*FLAGS.qa_batch_size:(i+1)*FLAGS.qa_batch_size]
                batch_questions = dev_qs[i*FLAGS.qa_batch_size:(i+1)*FLAGS.qa_batch_size]
                batch_ans_text = dev_as[i*FLAGS.qa_batch_size:(i+1)*FLAGS.qa_batch_size]
                batch_answer_charpos = dev_a_pos[i*FLAGS.qa_batch_size:(i+1)*FLAGS.qa_batch_size]

                batch_answers=[]
                for j, ctxt in enumerate(batch_contexts):
                    ans_span=char_pos_to_word(ctxt.encode(), [t.encode() for t in tokenise(ctxt, asbytes=False)], batch_answer_charpos[j][0])
                    ans_span=(ans_span, ans_span+len(tokenise(batch_ans_text[j][0],asbytes=False))-1)
                    batch_answers.append(ans_span)


                pred,nll = sess.run([model.pred_span, model.nll],
                        feed_dict={model.context_in: get_padded_batch(batch_contexts,vocab),
                                model.question_in: get_padded_batch(batch_questions,vocab),
                                model.answer_spans_in: batch_answers,
                                model.is_training: False})
                gold_str=[]
                pred_str=[]

                for b in range(FLAGS.qa_batch_size):
                    pred_str = " ".join(tokenise(batch_contexts[b],asbytes=False)[pred[b][0]:pred[b][1]+1])
                    this_f1=[]
                    this_em=[]
                    for a in range(len(batch_ans_text[b])):
                        this_f1.append(f1(normalize_answer(batch_ans_text[b][a]), normalize_answer(pred_str)))
                        this_em.append(1.0*(normalize_answer(batch_ans_text[b][a]) == normalize_answer(pred_str)))
                    f1s.append(max(this_f1))
                    exactmatches.append(max(this_em))
                nlls.extend(nll.tolist())
            f1summary = tf.Summary(value=[tf.Summary.Value(tag="dev_perf/f1",
                                             simple_value=sum(f1s)/len(f1s))])
            emsummary = tf.Summary(value=[tf.Summary.Value(tag="dev_perf/em",
                                      simple_value=sum(exactmatches)/len(exactmatches))])
            nllsummary = tf.Summary(value=[tf.Summary.Value(tag="dev_perf/nll",
                                      simple_value=np.mean(nlls))])

            summary_writer.add_summary(f1summary, global_step=((e+1)*num_steps_train))
            summary_writer.add_summary(emsummary, global_step=((e+1)*num_steps_train))
            summary_writer.add_summary(nllsummary, global_step=((e+1)*num_steps_train))

            mean_nll=np.mean(nlls)
            if mean_nll < best_oos_nll:
                print("New best NLL! ", mean_nll, " Saving... F1: ", np.mean(f1s))
                best_oos_nll = mean_nll
                saver.save(sess, chkpt_path+'/model.checkpoint')
            else:
                print("NLL not improved ", mean_nll)

if __name__ == '__main__':
    tf.app.run()
