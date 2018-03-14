# https://arxiv.org/pdf/1705.02012.pdf

import numpy as np
import tensorflow as tf


from squad_model import SQuADModel
from helpers.loader import OOV, PAD, EOS, SOS


class QGenMaluuba(SQuADModel):
    def __init__(self, vocab, batch_size):
        self.embedding_size = tf.app.flags.FLAGS.embedding_size
        self.context_encoder_units = tf.app.flags.FLAGS.context_encoder_units
        super().__init__(vocab, batch_size)

    def build_model(self):

        self.build_data_pipeline(self.batch_size)

        self.W = tf.get_variable('testvar', [len(self.vocab), len(self.vocab)], initializer=tf.orthogonal_initializer)

        a_oh = tf.one_hot(tf.mod(self.answer_ids, len(self.vocab)), depth=len(self.vocab))
        s = tf.shape(a_oh)
        x = tf.reshape(a_oh, [-1, len(self.vocab)])
        self.answer_hat = tf.reshape(tf.matmul(x, self.W), s)

        # build teacher output - coerce to vocab and pad with SOS/EOS
        # also build output for loss - one hot over vocab+context
        self.answer_onehot = tf.one_hot(self.answer_ids, depth=tf.constant(len(self.vocab), shape=[self.batch_size])+self.context_length)
        answer_coerced = tf.where(tf.greater_equal(self.answer_ids, len(self.vocab)), tf.tile(tf.constant([[self.vocab[OOV]]]), tf.shape(self.answer_ids)), self.answer_ids)
        self.answer_teach = tf.concat([tf.constant(self.vocab[SOS], shape=[self.batch_size, 1]), answer_coerced[:,:-1]], axis=1)

        # TODO: augment doc embeddings with in(answer)=true

        # Embed c,q,a
        self.embeddings = tf.get_variable('word_embeddings', [len(self.vocab), self.embedding_size], initializer=tf.orthogonal_initializer)

        # First, coerce them to the shortlist vocab. Then embed
        context_coerced = tf.where(tf.greater_equal(self.context_ids, len(self.vocab)), tf.tile(tf.constant([[self.vocab[OOV]]]), tf.shape(self.context_ids)), self.context_ids)
        self.context_embedded = tf.nn.embedding_lookup(self.embeddings, context_coerced)

        question_coerced = tf.where(tf.greater_equal(self.question_ids, len(self.vocab)), tf.tile(tf.constant([[self.vocab[OOV]]]), tf.shape(self.question_ids)), self.question_ids)
        self.question_embedded = tf.nn.embedding_lookup(self.embeddings, context_coerced)

        self.answer_teach_embedded = tf.nn.embedding_lookup(self.embeddings, self.answer_teach)

        # Build encoder for context
        # Build RNN cell for encoder
        with tf.variable_scope('context_encoder'):
            context_encoder_cell_fwd = tf.nn.rnn_cell.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(
                    cell=tf.contrib.rnn.BasicLSTMCell(num_units=self.context_encoder_units),
                    input_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.))) for n in range(1)])
            context_encoder_cell_bwd = tf.nn.rnn_cell.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(
                    cell=tf.contrib.rnn.BasicLSTMCell(num_units=self.context_encoder_units),
                    input_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.))) for n in range(1)])

            # Unroll encoder RNN
            context_encoder_outputs, context_encoder_state = tf.nn.bidirectional_dynamic_rnn(
                context_encoder_cell_fwd, context_encoder_cell_bwd, self.context_embedded,
                sequence_length=self.context_length, dtype=tf.float32)

        # Build encoder for mean(encoder(context)) + answer
        # Build RNN cell for encoder
        with tf.variable_scope('q_encoder'):
            # To build the "extractive condition encoding" input, take embeddings of answer words concated with encoded context at that position
            self.indices = tf.concat([[tf.range(self.answer_pos[i], self.answer_pos[i]+tf.reduce_max(self.answer_length)) for i in range(self.batch_size)]], axis=1)
            # max_ix=tf.tile(tf.expand_dims(self.context_length), [1,tf.reduce_max(self.answer_length)])
            # self.indices = tf.minimum(self.indices, max_ix)

            q_encoder_cell_fwd = tf.nn.rnn_cell.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(
                    cell=tf.contrib.rnn.BasicLSTMCell(num_units=self.context_encoder_units),
                    input_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.))) for n in range(1)])
            q_encoder_cell_bwd = tf.nn.rnn_cell.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(
                    cell=tf.contrib.rnn.BasicLSTMCell(num_units=self.context_encoder_units),
                    input_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.))) for n in range(1)])

            # Unroll encoder RNN
            q_encoder_outputs, q_encoder_state = tf.nn.bidirectional_dynamic_rnn(
                q_encoder_cell_fwd, q_encoder_cell_bwd, self.context_embedded,
                sequence_length=self.question_length, dtype=tf.float32)

        # build init state

        # decode

        # calc switch prob

        # get pointer location

        # build overall prediction prob vector

        self.loss = tf.reduce_mean(tf.square(self.answer_hat-a_oh))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(a_oh,tf.round(self.answer_hat)),tf.float32))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

    def predict(self):
        return self.answer_hat
