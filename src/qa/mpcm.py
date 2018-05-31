import sys
sys.path.insert(0, "/Users/tom/Dropbox/msc-ml/project/src/")


import tensorflow as tf

from base_model import TFModel

import helpers.loader as loader


import flags
FLAGS = tf.app.flags.FLAGS

mem_limit=0.25


# This should handle the mechanics of the model - basically it's a wrapper around the TF graph
class MpcmQa(TFModel):
    def __init__(self, vocab):
        self.embedding_size = tf.app.flags.FLAGS.embedding_size
        self.vocab = vocab
        super().__init__()


    def build_model(self):

        self.context_in = tf.placeholder(tf.int32, [None, None])
        self.question_in = tf.placeholder(tf.int32, [None, None])

        self.answer_spans_in = tf.placeholder(tf.int32, [None, 2])

        self.embeddings = tf.get_variable('word_embeddings', [len(self.vocab), self.embedding_size], initializer=tf.orthogonal_initializer)

        # Load glove embeddings
        self.glove_init_ops =[]
        glove_embeddings = loader.load_glove(FLAGS.data_path, d=FLAGS.embedding_size)
        for word,id in self.vocab.items():
            if word in glove_embeddings.keys():
                self.glove_init_ops.append(tf.assign(self.embeddings[id,:], glove_embeddings[word]))

        # Layer 1: representation layer
        self.context_embedded = tf.nn.embedding_lookup(self.embeddings, self.context_in)
        self.question_embedded = tf.nn.embedding_lookup(self.embeddings, self.question_in)

        # Layer 2: Filter. r is batch x con_len x q_len
        r = tf.matmul(self.context_embedded, tf.transpose(self.question_embedded, [0,2,1]))/(tf.norm(self.context_embedded, ord=1, axis=2)*tf.norm(self.question_embedded, ord=1, axis=2))
        r_context = tf.reduce_max(r, axis=2)
        r_question = tf.reduce_max(r, axis=1)

        self.context_filtered = r_context * self.context_embedded
        self.question_filtered = r_context * self.question_embedded

        # Layer 3: Context representation (BiLSTM encoder)
        cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units=32)
        cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units=32)
        self.context_encodings,_ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.context_filtered, dtype=tf.float32)
        self.question_encodings,_ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.question_filtered, dtype=tf.float32)

        # Layer 4: context matching layer
        #?????
        self.matches = tf.concat([], axis=2)

        # Layer 5: aggregate with BiLSTM
        cell_fw2 = tf.contrib.rnn.BasicLSTMCell(num_units=32)
        cell_bw2 = tf.contrib.rnn.BasicLSTMCell(num_units=32)
        self.aggregated_matches,_ = tf.nn.bidirectional_dynamic_rnn(cell_fw2, cell_bw2, self.self.matches, dtype=tf.float32)

        # Layer 6: Fully connected to get logits
        self.logits_start = tf.squeeze(tf.layers.fully_connected(self.aggregated_matches, 1, activation_fn=None))
        self.logits_end = tf.squeeze(tf.layers.fully_connected(self.aggregated_matches, 1, activation_fn=None))

        self.prob_start = tf.softmax(self.logits_start, axis=-1)
        self.probs_end = tf.softmax(self.logits_end, axis=-1)

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.answer_spans_in[:,0], logits=self.logits_start)+tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.answer_spans_in[:,1], logits=self.logits_end))


        self.optimise = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.concat([tf.argmax(self.probs_start, axis=-1),tf.argmax(self.probs_end, axis=-1)],axis=1), self.answer_spans_in), tf.float32))
