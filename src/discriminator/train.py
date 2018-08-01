import sys,json,math
sys.path.insert(0, "/Users/tom/Dropbox/msc-ml/project/src/")
sys.path.insert(0, "/cs/student/msc/ml/2017/thosking/dev/msc-project/src/")

import tensorflow as tf
import numpy as np

from instance import DiscriminatorInstance

import helpers.loader as loader

from tqdm import tqdm

def main(_):

    FLAGS = tf.app.flags.FLAGS

    with open('./logs'+'/out_eval_MALUUBA_CROP_GLOVE_SMART_train.json') as f:
        results = json.load(f)

    # results=results[:32]

    squad_dev = loader.load_squad_triples(FLAGS.data_path, False, v2=True)

    # dev_ctxts, dev_qs,dev_ans,dev_ans_pos, dev_correct = zip(*squad_dev)

    num_train = math.floor(len(results)*0.8)
    train_data=results[:num_train]
    dev_data=results[num_train:]


    disc = DiscriminatorInstance()
    disc.load_from_chkpt() # this loads the embeddings etc


    num_steps_train = len(train_data)//FLAGS.batch_size
    num_steps_dev = len(dev_data)//FLAGS.batch_size
    num_steps_squad = num_steps_dev

    best_oos_nll=1e6

    for e in range(FLAGS.num_epochs):
        np.random.shuffle(train_data)
        # Train for one epoch
        for i in tqdm(range(num_steps_train), desc='Epoch '+str(e)):
            batch = train_data[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
            qpred,qgold,ctxt,ans_text,ans_pos = zip(*batch)

            # print(ans_text)
            # print(ans_pos)
            # print(ctxt)
            # exit()

            ixs = np.round(np.random.binomial(1,0.5,FLAGS.batch_size))


            # +qpred[ix].replace("</Sent>","").replace("<PAD>","")
            qbatch = [qpred[ix].replace(" </Sent>","").replace(" <PAD>","") if ixs[ix] < 0.5 else qgold[ix] for ix in range(FLAGS.batch_size)]
            # qbatch = ["fake " if ixs[ix] < 0.5 else "real " for ix in range(FLAGS.batch_size)]
            # print(qbatch, ixs)
            loss = disc.train_step(ctxt, qbatch, ans_text, ans_pos, ixs, (e*num_steps_train+i))

        dev_acc=[]
        dev_nll=[]
        for i in tqdm(range(num_steps_dev), desc='Epoch '+str(e) + " dev"):
            batch = dev_data[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
            qpred,qgold,ctxt,ans_text,ans_pos = zip(*batch)

            # print(ans_text)
            # print(ans_pos)
            # print(ctxt)
            # exit()

            ixs = np.round(np.random.binomial(1,0.5,FLAGS.batch_size))

            qbatch = [qpred[ix].replace("</Sent>","").replace("<PAD>","") if ixs[ix] < 0.5 else qgold[ix] for ix in range(FLAGS.batch_size)]

            pred = disc.get_pred(ctxt, qbatch, ans_text, ans_pos)
            nll = disc.get_nll(ctxt, qbatch, ans_text, ans_pos, ixs)
            acc = 1.0*np.equal(np.round(pred), ixs)
            dev_acc.extend(acc.tolist())
            dev_nll.extend(nll.tolist())

        accsummary = tf.Summary(value=[tf.Summary.Value(tag="dev_perf/acc",
                                         simple_value=np.mean(dev_acc))])
        nllsummary = tf.Summary(value=[tf.Summary.Value(tag="dev_perf/nll",
                                  simple_value=np.mean(dev_nll))])

        disc.summary_writer.add_summary(accsummary, global_step=(e+1)*num_steps_train)
        disc.summary_writer.add_summary(nllsummary, global_step=(e+1)*num_steps_train)

        print(np.mean(dev_acc))
        if np.mean(dev_nll) < best_oos_nll:
            best_oos_nll=np.mean(dev_nll)
            disc.save_to_chkpt(FLAGS.model_dir, e)
            print("New best NLL, saving")

        squad_acc=[]
        squad_nll=[]
        squad_ixs=[]
        np.random.shuffle(squad_dev)
        for i in tqdm(range(num_steps_squad), desc='Epoch '+str(e)+ " squad"):
            batch = squad_dev[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
            ctxt,qbatch,ans_text,ans_pos,labels = zip(*batch)

            # print(ans_text)
            # print(ans_pos)
            # print(ctxt)
            # exit()

            # ixs = np.round(np.random.binomial(1,0.5,FLAGS.batch_size))

            # qbatch = [qpred[ix].replace("</Sent>","").replace("<PAD>","") if ixs[ix] < 0.5 else qgold[ix] for ix in range(FLAGS.batch_size)]

            pred = disc.get_pred(ctxt, qbatch, ans_text, ans_pos)
            nll = disc.get_nll(ctxt, qbatch, ans_text, ans_pos, labels)
            acc = 1.0*np.equal(np.round(pred), labels)
            squad_acc.extend(acc)
            squad_nll.extend(nll)
            squad_ixs.extend(labels)
        print("Squad: ", np.mean(squad_acc), " (bias ",np.mean(squad_ixs),")")

if __name__ == "__main__":
    tf.app.run()
