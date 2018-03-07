import tensorflow as tf
import numpy as np
import tqdm,os
import helpers.loader as loader

from qgen_model import QGenMaluuba


# load dataset
train_data = loader.load_squad_triples(False)
dev_data = loader.load_squad_triples(True)

# CUDA config
os.environ["CUDA_VISIBLE_DEVICES"]="2"
mem_limit=0.5

# Hyperparameters
tf.app.flags.DEFINE_boolean("train", False, "Training mode?")
tf.app.flags.DEFINE_integer("eval_freq", 100, "Evaluate the model after this many steps")
tf.app.flags.DEFINE_integer("num_epoch", 5, "Train the model for this many epochs")
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size")

FLAGS = tf.app.flags.FLAGS

# Data loader
# Should return a tuple of a batch of x,y
def get_batch(epoch, step, dev=False, test=False, size=FLAGS.batch_size):
    pass


def main(_):
    model = TFModel()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_limit)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        summary_writer = tf.train.SummaryWriter(summary_dir_name, sess.graph)
        saver = tf.train.Saver()

        if to_restore:
            saver.restore(sess, chkpt_path+ '/model.checkpoint')
        else:
            sess.run(tf.global_variables_initializer())

        x,y,is_training = model.placeholders()
        for e in range(num_epochs):
            for i in tqdm(range(num_steps), desc='Epoch '+str(e)):
                batch_xs, batch_ys = get_batch(e,i)
                _,train_summary = sess.run([model.optimizer, model.train_summary], feed_dict={x: batch_xs, y: batch_ys, is_training:True})
                summary_writer.add_summary(train_summary, global_step=(e*num_steps+i))

                if i % FLAGS.eval_freq == 0:
                    batch_xs, batch_ys = get_batch(e,i, dev=True)
                    dev_summary = sess.run([model.eval_summary], feed_dict={x: batch_xs, y: batch_ys, is_training:False})
                    summary_writer.add_summary(dev_summary, global_step=(e*num_steps+i))
                # if save_cond:
                #     saver.save(sess, chkpt_path+'/model.checkpoint')

if __name__ == '__main__':
    # print(train_data[1]['paragraphs'][0].keys())
    # print(train_data[1]['paragraphs'][0]['context'])
    # print(train_data[1]['paragraphs'][0]['qas'][0]['question'])
    # print(train_data[1]['paragraphs'][0]['qas'][0]['answers'][0]['text'])

    print(len(train_data))
    # tf.app.run()

    train_contexts, train_qs, train_as = zip(*train_data)
    vocab = loader.get_vocab(train_contexts, 2000)

    model = QGenMaluuba(vocab, batch_size=FLAGS.batch_size)

    with tf.Session() as sess:
        sess.run(model.iterator.initializer, feed_dict={model.context_ph: train_contexts,
                                          model.qs_ph: train_qs, model.as_ph: train_as})
        print(sess.run([model.this_q, model.this_a]))
        # print(sess.run(model.this_q))
