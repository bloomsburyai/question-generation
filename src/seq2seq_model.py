# https://arxiv.org/pdf/1705.02012.pdf

import numpy as np
import tensorflow as tf

from tensorflow.python.layers import base
from tensorflow.python.framework import tensor_shape

import helpers.preprocessing as preprocessing


from base_model import TFModel

from helpers.loader import OOV, PAD, EOS, SOS
import helpers.loader as loader

from copy_mechanism import copy_attention_wrapper, copy_layer

import helpers.ops as ops
from helpers.misc_utils import debug_shape, debug_tensor

FLAGS = tf.app.flags.FLAGS



class Seq2SeqModel(TFModel):
    def __init__(self, vocab, batch_size, advanced_condition_encoding=False, training_mode=False):
        self.vocab=vocab
        self.rev_vocab = {v:k for k,v in self.vocab.items()}
        self.batch_size = batch_size

        self.training_mode = training_mode

        self.embedding_size = tf.app.flags.FLAGS.embedding_size
        self.context_encoder_units = tf.app.flags.FLAGS.context_encoder_units
        self.answer_encoder_units = tf.app.flags.FLAGS.answer_encoder_units
        self.decoder_units = tf.app.flags.FLAGS.decoder_units
        self.advanced_condition_encoding = advanced_condition_encoding
        super().__init__()

    def build_model(self):

        # self.build_data_pipeline(self.batch_size)
        self.context_raw = tf.placeholder(tf.string, [None, None])  # source vectors of unknown size
        self.context_ids = tf.placeholder(tf.int32, [None, None])  # source vectors of unknown size
        self.context_length  = tf.placeholder(tf.int32, [None])     # size(source)
        self.question_raw  = tf.placeholder(tf.string, [None, None])  # target vectors of unknown size
        self.question_ids = tf.placeholder(tf.int32, [None, None])  # target vectors of unknown size
        self.question_length  = tf.placeholder(tf.int32, [None])     # size(source)
        self.answer_raw  = tf.placeholder(tf.string, [None, None])  # target vectors of unknown size
        self.answer_ids  = tf.placeholder(tf.int32, [None, None])  # target vectors of unknown size
        self.answer_length  = tf.placeholder(tf.int32, [None])
        self.answer_locs  = tf.placeholder(tf.int32, [None,None])


        self.this_context = (self.context_raw, self.context_ids, self.context_length)
        self.this_question = (self.question_raw, self.question_ids, self.question_length)
        self.this_answer = (self.answer_raw, self.answer_ids, self.answer_length, self.answer_locs)
        self.input_batch = (self.this_context, self.this_question, self.this_answer)

        curr_batch_size = tf.shape(self.answer_ids)[0]


        # build teacher output - coerce to vocab and pad with SOS/EOS
        # also build output for loss - one hot over vocab+context
        self.question_onehot = tf.one_hot(self.question_ids, depth=tf.tile([len(self.vocab)+FLAGS.max_copy_size], [curr_batch_size])+self.context_length)
        self.question_coerced = tf.where(tf.greater_equal(self.question_ids, len(self.vocab)), tf.tile(tf.constant([[self.vocab[OOV]]]), tf.shape(self.question_ids)), self.question_ids)
        self.question_teach = tf.concat([tf.tile(tf.constant(self.vocab[SOS], shape=[1, 1]), [curr_batch_size,1]), self.question_ids[:,:-1]], axis=1)
        self.question_teach_oh = tf.one_hot(self.question_teach, depth=len(self.vocab)+FLAGS.max_copy_size)
        # Embed c,q,a


        # init embeddings
        glove_embeddings = loader.load_glove(FLAGS.data_path, d=FLAGS.embedding_size)
        embeddings_init = tf.constant(loader.get_embeddings(self.vocab, glove_embeddings, D=FLAGS.embedding_size))
        self.embeddings = tf.get_variable('word_embeddings', initializer=embeddings_init, dtype=tf.float32)
        assert self.embeddings.shape == [len(self.vocab), self.embedding_size]


        # First, coerce them to the shortlist vocab. Then embed
        self.context_coerced = tf.where(tf.greater_equal(self.context_ids, len(self.vocab)), tf.tile(tf.constant([[self.vocab[OOV]]]), tf.shape(self.context_ids)), self.context_ids)
        self.context_embedded = tf.nn.embedding_lookup(self.embeddings, self.context_coerced)

        self.question_teach_embedded = tf.nn.embedding_lookup(self.embeddings, self.question_teach)
        self.question_embedded = tf.nn.embedding_lookup(self.embeddings, self.question_coerced)

        self.answer_coerced = tf.where(tf.greater_equal(self.answer_ids, len(self.vocab)), tf.tile(tf.constant([[self.vocab[OOV]]]), tf.shape(self.answer_ids)), self.answer_ids)
        self.answer_embedded = tf.nn.embedding_lookup(self.embeddings, self.answer_coerced) # batch x seq x embed

        # Is context token in answer?
        max_context_len = tf.reduce_max(self.context_length)
        context_ix = tf.tile(tf.expand_dims(tf.range(max_context_len),axis=0), [curr_batch_size,1])
        gt_start = tf.greater_equal(context_ix, tf.tile(tf.expand_dims(self.answer_locs[:,0],axis=1), [1, max_context_len]))
        lt_end = tf.less(context_ix, tf.tile(tf.expand_dims(self.answer_locs[:,0]+self.answer_length,axis=1), [1, max_context_len]))
        in_answer_feature = tf.expand_dims(tf.cast(tf.logical_and(gt_start, lt_end), tf.float32),axis=2)

        # augment embedding
        self.context_embedded = tf.concat([self.context_embedded, in_answer_feature], axis=2)

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
            context_encoder_output_parts, context_encoder_state = tf.nn.bidirectional_dynamic_rnn(
                context_encoder_cell_fwd, context_encoder_cell_bwd, self.context_embedded,
                sequence_length=self.context_length, dtype=tf.float32)
            self.context_encoder_output = tf.concat([context_encoder_output_parts[0], context_encoder_output_parts[1]], axis=2) # batch x seq x 2*units


        # Build encoder for mean(encoder(context)) + answer
        # Build RNN cell for encoder
        with tf.variable_scope('a_encoder'):
            # To build the "extractive condition encoding" input, take embeddings of answer words concated with encoded context at that position

            # This is super involved! Even though we have the right indices we have to do a LOT of massaging to get them in the right shape
            seq_length = tf.reduce_max(self.answer_length)
            # self.indices = tf.concat([[tf.range(self.answer_pos[i], self.answer_pos[i]+tf.reduce_max(self.answer_length)) for i in range(self.batch_size)]], axis=1)
            self.indices = self.answer_locs
            # cap the indices to be valid
            self.indices = tf.minimum(self.indices, tf.tile(tf.expand_dims(self.context_length-1,axis=1),[1,tf.reduce_max(self.answer_length)]))

            batch_ix = tf.expand_dims(tf.transpose(tf.tile(tf.expand_dims(tf.range(curr_batch_size),axis=0),[seq_length,1]),[1,0]),axis=2)
            full_ix = tf.concat([batch_ix,tf.expand_dims(self.indices,axis=-1)], axis=2)
            self.context_condition_encoding = tf.gather_nd(self.context_encoder_output, full_ix)


            self.full_condition_encoding = tf.concat([self.context_condition_encoding, self.answer_embedded], axis=2)

            a_encoder_cell_fwd = tf.nn.rnn_cell.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(
                    cell=tf.contrib.rnn.BasicLSTMCell(num_units=self.context_encoder_units),
                    input_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.))) for n in range(1)])
            a_encoder_cell_bwd = tf.nn.rnn_cell.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(
                    cell=tf.contrib.rnn.BasicLSTMCell(num_units=self.context_encoder_units),
                    input_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.))) for n in range(1)])

            # Unroll encoder RNN
            a_encoder_output_parts, a_encoder_state_parts = tf.nn.bidirectional_dynamic_rnn(
                a_encoder_cell_fwd, a_encoder_cell_bwd, self.full_condition_encoding,
                sequence_length=self.answer_length, dtype=tf.float32)

            self.a_encoder_final_state = tf.concat([a_encoder_state_parts[0][0].c, a_encoder_state_parts[1][0].c], axis=1) # batch x 2*a_encoder_units

        # concat direction outputs again

        # build init state
        with tf.variable_scope('decoder_initial_state'):
            L = tf.get_variable('decoder_L', [self.context_encoder_units*2, self.context_encoder_units*2], initializer=tf.orthogonal_initializer(), dtype=tf.float32)
            W0 = tf.get_variable('decoder_W0', [self.context_encoder_units*2, self.decoder_units], initializer=tf.orthogonal_initializer(), dtype=tf.float32)
            b0 = tf.get_variable('decoder_b0', [self.decoder_units], initializer=tf.zeros_initializer(), dtype=tf.float32)

            # This is a bit cheeky - this should be injected by the more advanced model. Consider refactoring into separate methods then overloading the one that handles this
            if self.advanced_condition_encoding:
                self.context_encoding = self.a_encoder_final_state # this would be the maluuba model
            else:
                self.context_encoding = tf.reduce_mean(self.context_condition_encoding, axis=1) # this is the baseline model

            r = tf.reduce_sum(self.context_encoder_output, axis=1)/tf.tile(tf.expand_dims(tf.cast(self.context_length,tf.float32),axis=1),[1,self.context_encoder_units*2]) + tf.matmul(self.context_encoding,L)
            self.s0 = tf.nn.tanh(tf.matmul(r,W0) + b0)

        # decode
        # TODO: for Maluuba model, decoder inputs are concat of context and answer encoding
        with tf.variable_scope('decoder'):

            if not self.training_mode:
                memory = tf.contrib.seq2seq.tile_batch( self.context_encoder_output, multiplier=FLAGS.beam_width )
                memory_sequence_length = tf.contrib.seq2seq.tile_batch( self.context_length, multiplier=FLAGS.beam_width)
                s0_tiled = tf.contrib.seq2seq.tile_batch( self.s0, multiplier=FLAGS.beam_width)
                init_state = tf.contrib.rnn.LSTMStateTuple(s0_tiled, tf.contrib.seq2seq.tile_batch(tf.zeros([curr_batch_size, self.decoder_units]), multiplier=FLAGS.beam_width))
                # init_state = tf.contrib.rnn.LSTMStateTuple(self.s0, tf.zeros([curr_batch_size, self.decoder_units]))
                # init_state = tf.contrib.seq2seq.tile_batch( init_state, multiplier=FLAGS.beam_width)
            else:
                memory = self.context_encoder_output
                memory_sequence_length = self.context_length
                init_state = tf.contrib.rnn.LSTMStateTuple(self.s0, tf.zeros([curr_batch_size, self.decoder_units]))



            attention_mechanism = copy_attention_wrapper.BahdanauAttention(
                            num_units=self.decoder_units, memory=memory,
                            memory_sequence_length=memory_sequence_length)

            # copy_mechanism = copy_attention_wrapper.BahdanauAttention(
            #                 num_units=self.decoder_units, memory=memory,
            #                 memory_sequence_length=memory_sequence_length)

            decoder_cell = tf.contrib.rnn.DropoutWrapper(
                    cell=tf.contrib.rnn.BasicLSTMCell(num_units=self.decoder_units),
                    input_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.)))



            # decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell,
            #                                                     attention_mechanism,
            #                                                     attention_layer_size=self.decoder_units / 2,
            #                                                     alignment_history=True)

            decoder_cell = copy_attention_wrapper.CopyAttentionWrapper(decoder_cell,
                                                                attention_mechanism,
                                                                attention_layer_size=self.decoder_units / 2,
                                                                alignment_history=False,
                                                                copy_mechanism=attention_mechanism,
                                                                output_attention=True,
                                                                initial_cell_state=init_state)

            init_state = decoder_cell.zero_state(curr_batch_size*(FLAGS.beam_width if not self.training_mode else 1), tf.float32).clone(cell_state=init_state)

            projection_layer = copy_layer.CopyLayer(FLAGS.decoder_units//2, FLAGS.max_copy_size,
                                            source_provider=lambda: self.context_ids,
                                            condition_encoding=lambda: self.context_encoding,
                                            vocab_size=len(self.vocab))

            if self.training_mode:
                # Helper - training
                helper = tf.contrib.seq2seq.TrainingHelper(
                    self.question_teach_oh, self.question_length)
                    # decoder_emb_inp, length(decoder_emb_inp)+1)

                # Decoder - training
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell, helper,
                    initial_state=init_state,
                    # initial_state=encoder_state
                    # TODO: hardcoded FLAGS.max_copy_size is longest context in SQuAD - this will need changing for a new dataset!!!
                    output_layer=projection_layer
                    )

                # Unroll the decoder
                outputs, decoder_states,out_lens = tf.contrib.seq2seq.dynamic_decode(decoder,impute_finished=True, maximum_iterations=tf.reduce_max(self.question_length))

                logits=outputs.rnn_output
            else:
                start_tokens = tf.tile(tf.constant([self.vocab[SOS]], dtype=tf.int32), [ curr_batch_size  ] )
                end_token = self.vocab[EOS]

                # init_state = tf.contrib.seq2seq.tile_batch( init_state, multiplier=FLAGS.beam_width )
                # init_state = decoder_cell.zero_state(curr_batch_size * FLAGS.beam_width, tf.float32).clone(cell_state=init_state)
                # init_state = decoder_cell.zero_state(curr_batch_size, tf.float32).clone(cell_state=init_state)



                my_decoder = tf.contrib.seq2seq.BeamSearchDecoder( cell = decoder_cell,
                                                                   embedding = tf.eye(len(self.vocab) + FLAGS.max_copy_size),
                                                                   start_tokens = start_tokens,
                                                                   end_token = end_token,
                                                                   initial_state = init_state,
                                                                   beam_width = FLAGS.beam_width,
                                                                   output_layer = projection_layer )

                # helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                #       embedding=tf.eye(len(self.vocab) + FLAGS.max_copy_size),
                #       start_tokens=tf.tile(tf.constant([self.vocab[SOS]], dtype=tf.int32), [ curr_batch_size ] ),
                #       end_token=end_token)
                # my_decoder = tf.contrib.seq2seq.BasicDecoder( cell = decoder_cell,
                #                                                 helper=helper,
                #                                                   initial_state = init_state,
                #                                                   output_layer = projection_layer )

                outputs, decoder_states,out_lens = tf.contrib.seq2seq.dynamic_decode(  my_decoder,
                                                                        impute_finished=False,
                                                                       maximum_iterations=32 )

                # logits = outputs.rnn_output
                pred_ids = outputs.predicted_ids
                # pred_ids = debug_shape(pred_ids, "pred ids")
                logits = tf.one_hot(pred_ids[:,:,0], depth=len(self.vocab)+FLAGS.max_copy_size)
                # logits2 =  tf.one_hot(pred_ids[:,:,1], depth=len(self.vocab)+FLAGS.max_copy_size)


        self.q_hat = tf.nn.softmax(logits, dim=2)

        # self.q_hat = debug_shape(self.q_hat, "q hat")

        with tf.variable_scope('train_loss'):
            self.target_weights = tf.sequence_mask(
                        self.question_length, tf.reduce_max(self.question_length), dtype=tf.float32)
            logits = ops.safe_log(self.q_hat)

            self.crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.question_ids, logits=logits)
            qlen_float = tf.cast(self.question_length, tf.float32)
            self.xe_loss = tf.reduce_mean(tf.reduce_sum(self.crossent * self.target_weights,axis=1)/qlen_float,axis=0)

            # TODO: Check these should be included in baseline?
            # get sum of all probabilities for words that are also in answer
            answer_oh = tf.one_hot(self.answer_ids, depth=len(self.vocab) +FLAGS.max_copy_size)
            answer_mask = tf.tile(tf.reduce_sum(answer_oh, axis=1,keep_dims=True), [1,tf.reduce_max(self.question_length),1])
            self.suppression_loss = tf.reduce_mean(tf.reduce_sum(answer_mask * self.q_hat,axis=[1,2])/qlen_float,axis=0)

            # entropy maximiser
            self.entropy_loss = tf.reduce_mean(tf.reduce_sum(self.q_hat * ops.safe_log(self.q_hat),axis=[1,2])/qlen_float,axis=0)

            self._train_summaries.append(tf.summary.scalar('train_loss/xe_loss', self.xe_loss))

        self.loss = self.xe_loss + 0.01*self.suppression_loss + 0.01*self.entropy_loss


        with tf.variable_scope('output'):
            self.q_hat_ids = tf.argmax(self.q_hat,axis=2,output_type=tf.int32)
            self.a_string = ops.id_tensor_to_string(self.answer_coerced, self.rev_vocab, self.context_raw)
            self.q_hat_string = ops.id_tensor_to_string(self.q_hat_ids, self.rev_vocab, self.context_raw)

            # q_hat_ids2 = tf.argmax(tf.nn.softmax(logits2, dim=2),axis=2,output_type=tf.int32)
            # self.q_hat_string2 = ops.id_tensor_to_string(q_hat_ids2, self.rev_vocab, self.context_raw)

            self.q_gold = ops.id_tensor_to_string(self.question_ids, self.rev_vocab, self.context_raw)
            self._output_summaries.extend(
                [tf.summary.text("q_hat", self.q_hat_string),
                tf.summary.text("q_gold", self.q_gold),
                # tf.summary.text("q_gold_ids", tf.as_string(self.question_ids)),
                # tf.summary.text("q_raw", self.question_raw),
                # tf.summary.text("context", self.context_raw),
                tf.summary.text("answer", self.answer_raw)])

        # Calculate and clip gradients
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, 5)

        # Optimization
        self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).apply_gradients(
            zip(clipped_gradients, params)) if self.training_mode else tf.no_op()

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.question_ids,tf.argmax(self.q_hat,axis=2,output_type=tf.int32)),tf.float32)*self.target_weights)
