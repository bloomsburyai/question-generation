import os,time, json, datetime

# model_type = "SEQ2SEQ"
# model_type = "MALUUBA"

# CUDA config
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
mem_limit=0.5

import tensorflow as tf
import numpy as np
import helpers.loader as loader
import helpers.metrics as metrics
import helpers.preprocessing as preprocessing
import helpers.ops as ops
from helpers.output import output_pretty, tokens_to_string, output_eval
from tqdm import tqdm

from seq2seq_model import Seq2SeqModel
from rl_model import RLModel
from datasources.squad_streamer import SquadStreamer
from langmodel.lm import LstmLmInstance
# from qa.mpcm import MpcmQaInstance
from qa.qanet.instance import QANetInstance
from discriminator.instance import DiscriminatorInstance

import flags

FLAGS = tf.app.flags.FLAGS

def main(_):

    model_type=FLAGS.model_type
    # chkpt_path = FLAGS.model_dir+'saved/qgen-maluuba-crop-glove-smart'
    # chkpt_path = FLAGS.model_dir+'qgen-saved/MALUUBA-CROP-LATENT/1533247183'
    disc_path = FLAGS.model_dir+'saved/discriminator-trained-latent'
    chkpt_path = FLAGS.model_dir+'qgen/'+ model_type+'/'+FLAGS.eval_model_id

    # load dataset
    # train_data = loader.load_squad_triples(FLAGS.data_path, False)
    dev_data = loader.load_squad_triples(FLAGS.data_path, dev=FLAGS.eval_on_dev, test=FLAGS.eval_on_test)

    if len(dev_data) < FLAGS.num_eval_samples:
        exit('***ERROR*** Eval dataset is smaller than the num_eval_samples flag!')
    if len(dev_data) > FLAGS.num_eval_samples:
        print('***WARNING*** Eval dataset is larger than the num_eval_samples flag!')

    # train_contexts_unfilt, _,_,train_a_pos_unfilt = zip(*train_data)
    dev_contexts_unfilt, _,_,dev_a_pos_unfilt = zip(*dev_data)

    if FLAGS.filter_window_size_before >-1:
        # train_data = preprocessing.filter_squad(train_data, window_size=FLAGS.filter_window_size, max_tokens=FLAGS.filter_max_tokens)
        dev_data = preprocessing.filter_squad(dev_data, window_size_before=FLAGS.filter_window_size_before, window_size_after=FLAGS.filter_window_size_after, max_tokens=FLAGS.filter_max_tokens)


    # print('Loaded SQuAD with ',len(train_data),' triples')
    print('Loaded SQuAD dev set with ',len(dev_data),' triples')
    # train_contexts, train_qs, train_as,train_a_pos = zip(*train_data)
    dev_contexts, dev_qs, dev_as, dev_a_pos = zip(*dev_data)


    # vocab = loader.get_vocab(train_contexts, tf.app.flags.FLAGS.vocab_size)
    with open(chkpt_path+'/vocab.json') as f:
        vocab = json.load(f)

    with SquadStreamer(vocab, FLAGS.eval_batch_size, 1, shuffle=False) as dev_data_source:

        glove_embeddings = loader.load_glove(FLAGS.data_path)


        # Create model
        if model_type[:7] == "SEQ2SEQ":
            model = Seq2SeqModel(vocab, training_mode=False)
        elif model_type[:2] == "RL":
            # TEMP - no need to spin up the LM or QA model at eval time
            FLAGS.qa_weight = 0
            FLAGS.lm_weight = 0
            model = RLModel(vocab, training_mode=False)
        else:
            exit("Unrecognised model type: "+model_type)

        with model.graph.as_default():
            saver = tf.train.Saver()

        if FLAGS.eval_metrics:
            lm = LstmLmInstance()
            # qa = MpcmQaInstance()
            qa = QANetInstance()

            lm.load_from_chkpt(FLAGS.model_dir+'saved/lmtest')
            # qa.load_from_chkpt(FLAGS.model_dir+'saved/qatest')
            qa.load_from_chkpt(FLAGS.model_dir+'saved/qanet2')

            discriminator = DiscriminatorInstance(trainable=False, path=disc_path)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_limit)
        with tf.Session(graph=model.graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            if not os.path.exists(chkpt_path):
                exit('Checkpoint path doesnt exist! '+chkpt_path)
            # summary_writer = tf.summary.FileWriter(FLAGS.log_dir+"eval/"+str(int(time.time())), sess.graph)

            saver.restore(sess, tf.train.latest_checkpoint(chkpt_path))
            # print('Loading not implemented yet')
            # else:
            #     sess.run(tf.global_variables_initializer())
            #     sess.run(model.glove_init_ops)

            num_steps = FLAGS.num_eval_samples//FLAGS.eval_batch_size

            # Initialise the dataset

            # np.random.shuffle(dev_data)
            dev_data_source.initialise(dev_data)

            f1s=[]
            bleus=[]
            qa_scores=[]
            qa_scores_gold=[]
            lm_scores=[]
            nlls=[]
            disc_scores=[]
            sowe_similarities=[]
            copy_probs=[]

            qgolds=[]
            qpreds=[]
            qpred_ids=[]
            qgold_ids=[]
            ctxts=[]
            answers=[]
            ans_positions=[]

            metric_individuals=[]
            res=[]
            for e in range(1):
                for i in tqdm(range(num_steps), desc='Epoch '+str(e)):
                    dev_batch, curr_batch_size = dev_data_source.get_batch()
                    pred_batch,pred_beam,pred_beam_lens,pred_ids,pred_lens,gold_batch, gold_lens,gold_ids,ctxt,ctxt_len,ans,ans_len,nll,copy_prob= sess.run([model.q_hat_beam_string, model.q_hat_full_beam_str, model.q_hat_full_beam_lens,model.q_hat_beam_ids,model.q_hat_beam_lens,model.question_raw, model.question_length, model.question_ids, model.context_raw, model.context_length, model.answer_locs, model.answer_length, model.nll, model.mean_copy_prob], feed_dict={model.input_batch: dev_batch ,model.is_training:False})

                    unfilt_ctxt_batch = [dev_contexts_unfilt[ix] for ix in dev_batch[3]]
                    a_text_batch = ops.byte_token_array_to_str(dev_batch[2][0], dev_batch[2][2], is_array=False)
                    unfilt_apos_batch = [dev_a_pos_unfilt[ix] for ix in dev_batch[3]]

                    # subtract 1 to remove the "end sent token"
                    pred_q_batch = [q.replace(' </Sent>',"").replace(" <PAD>","") for q in ops.byte_token_array_to_str(pred_batch, pred_lens-1)]

                    ctxts.extend(unfilt_ctxt_batch)
                    answers.extend(a_text_batch)
                    ans_positions.extend([dev_a_pos_unfilt[ix] for ix in dev_batch[3]])
                    copy_probs.extend(copy_prob.tolist())



                    # get QA score

                    # gold_str=[]
                    # pred_str=[]


                    gold_ans = ops.byte_token_array_to_str(dev_batch[2][0], dev_batch[2][2], is_array=False)
                    # pred_str = ops.byte_token_array_to_str([dev_batch[0][0][b][qa_pred[b][0]:qa_pred[b][1]] for b in range(curr_batch_size)], is_array=False)
                    nlls.extend(nll.tolist())

                    if FLAGS.eval_metrics:
                        qa_pred = qa.get_ans(unfilt_ctxt_batch, ops.byte_token_array_to_str(pred_batch, pred_lens))
                        gold_qa_pred = qa.get_ans(unfilt_ctxt_batch, ops.byte_token_array_to_str(dev_batch[1][0], dev_batch[1][3]))

                        qa_score_batch = [metrics.f1(metrics.normalize_answer(gold_ans[b]), metrics.normalize_answer(qa_pred[b])) for b in range(curr_batch_size)]
                        qa_score_gold_batch = [metrics.f1(metrics.normalize_answer(gold_ans[b]), metrics.normalize_answer(gold_qa_pred[b])) for b in range(curr_batch_size)]
                        lm_score_batch = lm.get_seq_perplexity(pred_q_batch).tolist()
                        disc_score_batch = discriminator.get_pred(unfilt_ctxt_batch, pred_q_batch, gold_ans, unfilt_apos_batch).tolist()

                    for b, pred in enumerate(pred_batch):
                        pred_str = pred_q_batch[b].replace(' </Sent>',"").replace(" <PAD>","")
                        gold_str = tokens_to_string(gold_batch[b][:gold_lens[b]-1])
                        f1s.append(metrics.f1(gold_str, pred_str))
                        bleus.append(metrics.bleu(gold_str, pred_str))
                        qgolds.append(gold_str)
                        qpreds.append(pred_str)

                        # calc cosine similarity between sums of word embeddings
                        pred_sowe = np.sum(np.asarray([glove_embeddings[w] if w in glove_embeddings.keys() else np.zeros((FLAGS.embedding_size,)) for w in preprocessing.tokenise(pred_str ,asbytes=False)]) ,axis=0)
                        gold_sowe = np.sum(np.asarray([glove_embeddings[w] if w in glove_embeddings.keys() else np.zeros((FLAGS.embedding_size,)) for w in preprocessing.tokenise(gold_str ,asbytes=False)]) ,axis=0)
                        this_similarity = np.inner(pred_sowe, gold_sowe)/np.linalg.norm(pred_sowe, ord=2)/np.linalg.norm(gold_sowe, ord=2)

                        sowe_similarities.append(this_similarity)



                        this_metric_dict={
                            'f1':f1s[-1],
                            'bleu': bleus[-1],
                            'nll': nlls[-1],
                            'sowe': sowe_similarities[-1]
                            }
                        if FLAGS.eval_metrics:
                            this_metric_dict={
                            **this_metric_dict,
                            'qa': qa_score_batch[b],
                            'lm': lm_score_batch[b],
                            'disc': disc_score_batch[b]}
                            qa_scores.extend(qa_score_batch)
                            lm_scores.extend(lm_score_batch)
                            disc_scores.extend(disc_score_batch)
                        metric_individuals.append(this_metric_dict)

                        res.append({
                            'c':unfilt_ctxt_batch[b],
                            'q_pred': pred_str,
                            'q_gold': gold_str,
                            'a_pos': unfilt_apos_batch[b],
                            'a_text': a_text_batch[b],
                            'metrics': this_metric_dict,

                            'q_pred_ids': pred_ids.tolist()[b],
                            'q_gold_ids': dev_batch[1][1][b].tolist()

                        })

                    # Quick output
                    if i==0:
                        # print(copy_prob.tolist())
                        # print(copy_probs)
                        pred_str = tokens_to_string(pred_batch[0][:pred_lens[0]-1])
                        gold_str = tokens_to_string(gold_batch[0][:gold_lens[0]-1])
                        # print(pred_str)
                        print(qpreds[0])
                        print(gold_str)


                        title=chkpt_path
                        out_str = output_eval(title,pred_batch,  pred_ids, pred_lens, gold_batch, gold_lens, ctxt, ctxt_len, ans, ans_len)
                        with open(FLAGS.log_dir+'out_eval_'+model_type+'.htm', 'w', encoding='utf-8') as fp:
                            fp.write(out_str)

            # res = list(zip(qpreds,qgolds,ctxts,answers,ans_positions,metric_individuals))
            metric_dict={
                'f1':np.mean(f1s),
                'bleu': metrics.bleu_corpus(qgolds, qpreds),
                'nll':np.mean(nlls),
                'sowe': np.mean(sowe_similarities)
                }
            if FLAGS.eval_metrics:
                metric_dict={**metric_dict,
                'qa':np.mean(qa_scores),
                'lm':np.mean(lm_scores),
                'disc': np.mean(disc_scores)}
            # print(res)
            with open(FLAGS.log_dir+'out_eval_'+model_type+("_test" if FLAGS.eval_on_test else "")+("_train" if (not FLAGS.eval_on_dev and not FLAGS.eval_on_test) else "")+'.json', 'w', encoding='utf-8') as fp:
                json.dump({"metrics":metric_dict, "results": res}, fp)


            print("F1: ", np.mean(f1s))
            print("BLEU: ", metrics.bleu_corpus(qgolds, qpreds))
            print("NLL: ", np.mean(nlls))
            print("SOWE: ", np.mean(sowe_similarities))

            print("Copy prob: ", np.mean(copy_probs))
            if FLAGS.eval_metrics:
                print("QA: ", np.mean(qa_scores))
                print("LM: ", np.mean(lm_scores))
                print("Disc: ", np.mean(disc_scores))

if __name__ == '__main__':
    tf.app.run()
