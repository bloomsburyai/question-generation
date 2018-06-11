import sys
sys.path.insert(0, "/Users/tom/Dropbox/msc-ml/project/src/")


import tensorflow as tf

from base_model import TFModel

import helpers.loader as loader
import helpers.ops as ops


import flags
FLAGS = tf.app.flags.FLAGS

# mem_limit=0.25


# This should handle the mechanics of the model - basically it's a wrapper around the TF graph
class LstmLm(TFModel):
    def __init__(self, vocab, num_units=128, training_mode=True):
        self.embedding_size = tf.app.flags.FLAGS.embedding_size
        self.num_units = num_units
        self.vocab = vocab
        self.training_mode = training_mode
        super().__init__()


    def build_model(self):

        with tf.device('/cpu:*'):
            # Load glove embeddings
            glove_embeddings = loader.load_glove(FLAGS.data_path, d=FLAGS.embedding_size)
            embeddings_init = tf.constant(loader.get_embeddings(self.vocab, glove_embeddings, D=FLAGS.embedding_size))
            self.embeddings = tf.get_variable('word_embeddings', initializer=embeddings_init, dtype=tf.float32)
            assert self.embeddings.shape == [len(self.vocab), self.embedding_size]

        # input placeholder
        self.input_seqs = tf.placeholder(tf.int32, [None, None])

        self.input_lengths = tf.reduce_sum(tf.cast(tf.not_equal(self.input_seqs, 0), tf.int32), axis=1)

        self.input_embedded = tf.nn.embedding_lookup(self.embeddings, self.input_seqs)

        self.tgt_input = self.input_embedded[:,:-1,:] # start:end-1 - embedded
        self.tgt_output = self.input_seqs[:,1:]  # start+1:end - ids

        # RNN
        cell = tf.contrib.rnn.BasicLSTMCell(self.num_units)

        outputs, states = tf.nn.dynamic_rnn(cell, self.tgt_input, dtype=tf.float32)

        self.logits = tf.layers.dense(outputs, len(self.vocab))
        self.probs = tf.nn.softmax(self.logits)
        self.preds = tf.argmax(self.probs, axis=2, output_type=tf.int32)



        # loss fn + opt
        self.target_weights = tf.sequence_mask(
                    self.input_lengths-1, tf.shape(self.input_seqs)[1]-1, dtype=tf.float32)
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.tgt_output)*self.target_weights,axis=1)/tf.cast(tf.reduce_sum(self.target_weights,axis=1),tf.float32),axis=0)

        if self.training_mode:
            self.optimise = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        # seq evaluation
        self.log_probs = tf.reduce_sum(tf.one_hot(self.tgt_output, depth=len(self.vocab))*self.probs,axis=2)
        self.seq_log_prob = tf.reduce_sum(ops.safe_log(self.log_probs)*self.target_weights, axis=1)/(tf.cast(tf.reduce_sum(self.target_weights,axis=1),tf.float32)+1e-6)

        # metrics
        self.perplexity = tf.minimum(1000.0,tf.pow(2.0, 1/tf.cast(self.input_lengths,tf.float32) * tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.tgt_output)*self.target_weights,axis=1)))
        self._train_summaries.append(tf.summary.scalar("train_perf/perplexity", tf.reduce_mean(self.perplexity)))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.preds, self.tgt_output), tf.float32))

# This should handle a concrete instance of a LM, loading params, spinning up the graph etc, to be used by other models
class LstmLmInstance():
    def __init__(self, vocab):
        self.model = LstmLm(vocab, num_units=512, training_mode=False)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_limit,allow_growth = True)
        self.sess = tf.Session(graph=self.model.graph, config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True))

    def __del__(self):
        self.sess.close()

    def load_from_chkpt(self, path):
        with self.model.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, path+ '/model.checkpoint')

    def get_seq_prob(self, seqs):
        probs = self.sess.run(self.model.seq_log_prob, feed_dict={self.model.input_seqs: seqs})
        return probs


def main(_):
    train_data = loader.load_squad_triples("./data/", False)

    import numpy as np
    from helpers.preprocessing import tokenise

    print('Loaded SQuAD with ',len(train_data),' triples')
    train_contexts, train_qs, train_as,train_a_pos = zip(*train_data)
    vocab = loader.get_vocab(train_qs, tf.app.flags.FLAGS.lm_vocab_size)

    lm = LstmLmInstance(vocab)
    lm.load_from_chkpt(FLAGS.model_dir+'saved/lmtest')

    # random words, basic q, common words, real q, real context
    seq_batch = [" ".join(np.random.choice(list(vocab.keys()), 20)), "what ?","what is is is the ?", "Where is the Baha'i national office located in Nepal?", "Kathmandu Metropolitan City (KMC), in order to promote international relations has established an International Relations Secretariat (IRC). KMC's first international relationship was established in 1975 with the city of Eugene, Oregon, United States. This activity has been further enhanced by establishing formal relationships with 8 other cities"]
    seq_batch_ids = [[vocab[loader.SOS]]+[vocab[tok if tok in vocab.keys() else loader.OOV] for tok in tokenise(sent, asbytes=False)]+[vocab[loader.EOS]] for sent in seq_batch]
    max_seq_len = max([len(seq) for seq in seq_batch_ids])
    padded_batch = np.asarray([seq + [vocab[loader.PAD] for i in range(max_seq_len-len(seq))] for seq in seq_batch_ids])

    print(lm.get_seq_prob(padded_batch))
if __name__ == "__main__":
    tf.app.run()
