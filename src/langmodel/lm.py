import tensorflow as tf

from base_model import TFModel

from helpers.loader import load_glove

FLAGS = tf.app.flags.FLAGS


# This should handle the mechanics of the model - basically it's a wrapper around the TF graph
class LstmLm(TFModel):
    def __init__(self, vocab, num_units=128):
        self.embedding_size = tf.app.flags.FLAGS.embedding_size
        self.num_units = num_units
        self.vocab = vocab
        super().__init__()


    def build_model(self):

        self.embeddings = tf.get_variable('word_embeddings', [len(self.vocab), self.embedding_size], initializer=tf.orthogonal_initializer)

        # Load glove embeddings
        self.glove_init_ops =[]
        glove_embeddings = load_glove(FLAGS.data_path, d=FLAGS.embedding_size)
        for word,id in self.vocab.items():
            if word in glove_embeddings.keys():
                self.glove_init_ops.append(tf.assign(self.embeddings[id,:], glove_embeddings[word]))

        # input placeholder
        self.input_seqs = tf.placeholder(tf.int32, [None, None])

        self.input_embedded = tf.nn.embedding_lookup(self.embeddings, self.input_seqs)

        self.tgt_input = self.input_embedded[:,:-1,:] # start:end-1 - embedded
        self.tgt_output = self.input_seqs[:,1:]  # start+1:end - ids

        # RNN
        cell = tf.contrib.rnn.BasicLSTMCell(self.num_units)

        outputs, states = tf.nn.dynamic_rnn(cell, self.tgt_input, dtype=tf.float32)

        self.logits = tf.layers.dense(outputs, len(self.vocab))
        self.pred = tf.nn.softmax(self.logits)

        # loss fn + opt
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.tgt_output))

        self.optimise = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.pred, axis=2, output_type=tf.int32), self.tgt_output), tf.int32))

# This should handle a concrete instance of a LM, loading params, spinning up the graph etc, to be used by other models
class LstmLmInstance():
    def __init__(self, vocab):
        pass

    def load_from_chkpt(self, path):
        pass

    def get_prob(self, contexts, toks):
        pass

    # Get total log prob of input sequences, given themselves
    def get_seq_prob(self, seqs):
        pass
