# https://arxiv.org/pdf/1705.02012.pdf

import numpy as np
import tensorflow as tf

import helpers.preprocessing as preprocessing


from base_model import TFModel

from helpers.loader import OOV, PAD, EOS, SOS
import helpers.loader as loader

from copy_mechanism import copy_attention_wrapper, copy_layer

from dbs.diverse_bs import DiverseBeamSearchDecoder

import helpers.ops as ops
from helpers.misc_utils import debug_shape, debug_tensor, debug_op

FLAGS = tf.app.flags.FLAGS



class Seq2SeqModel(TFModel):
    def __init__(self, vocab, training_mode=False, use_embedding_loss=False):
        self.vocab=vocab
        self.rev_vocab = {v:k for k,v in self.vocab.items()}

        self.training_mode = training_mode
        self.use_embedding_loss = use_embedding_loss

        self.embedding_size = tf.app.flags.FLAGS.embedding_size
        self.context_encoder_units = tf.app.flags.FLAGS.context_encoder_units
        self.answer_encoder_units = tf.app.flags.FLAGS.answer_encoder_units
        self.decoder_units = tf.app.flags.FLAGS.decoder_units
        self.advanced_condition_encoding = FLAGS.advanced_condition_encoding
        super().__init__()

    def build_model(self):

        with tf.device('/cpu:*'):
            self.context_raw = tf.placeholder(tf.string, [None, None], name='placeholder_context_raw')  # source vectors of unknown size
            self.question_raw  = tf.placeholder(tf.string, [None, None])  # target vectors of unknown size
            self.answer_raw  = tf.placeholder(tf.string, [None, None], name='placeholder_ans_raw')  # target vectors of unknown size
        self.context_ids = tf.placeholder(tf.int32, [None, None], name='placeholder_context_ids')  # source vectors of unknown size
        self.context_copy_ids = tf.placeholder(tf.int32, [None, None], name='placeholder_context_cp_ids')  # source vectors of unknown size
        self.context_length  = tf.placeholder(tf.int32, [None], name='placeholder_context_len')     # size(source)
        self.context_vocab_size  = tf.placeholder(tf.int32, [None], name='placeholder_context_vocsize')     # size(source_vocab)
        self.question_ids = tf.placeholder(tf.int32, [None, None])  # target vectors of unknown size
        self.question_onehot = tf.placeholder(tf.float32, [None, None, None])  # target vectors of unknown size
        self.question_length  = tf.placeholder(tf.int32, [None])     # size(source)
        self.answer_ids  = tf.placeholder(tf.int32, [None, None], name='placeholder_ans_ids')  # target vectors of unknown size
        self.answer_length  = tf.placeholder(tf.int32, [None], name='placeholder_ans_len')
        self.answer_locs  = tf.placeholder(tf.int32, [None,None], name='placeholder_ans_len')
        self.original_ix  = tf.placeholder(tf.int32, [None]) # unused - gives the index of the input in the unshuffled dataset

        self.hide_answer_in_copy = tf.placeholder_with_default(False, (),"hide_answer_in_copy")


        self.context_in = (self.context_raw, self.context_ids, self.context_copy_ids, self.context_length, self.context_vocab_size)
        self.question_in = (self.question_raw, self.question_ids, self.question_onehot, self.question_length)
        self.answer_in = (self.answer_raw, self.answer_ids, self.answer_length, self.answer_locs)
        self.input_batch = (self.context_in, self.question_in, self.answer_in, self.original_ix)

        curr_batch_size = tf.shape(self.answer_ids)[0]

        with tf.variable_scope('input_pipeline'):
            # build teacher output - coerce to vocab and pad with SOS/EOS
            # also build output for loss - one hot over vocab+context

            self.question_teach = tf.concat([tf.tile(tf.constant(self.vocab[SOS], shape=[1, 1]), [curr_batch_size,1]), self.question_ids[:,:-1]], axis=1)
            # self.question_teach_oh = tf.concat([tf.one_hot(tf.tile(tf.constant(self.vocab[SOS], shape=[1, 1]), [curr_batch_size,1]), depth=len(self.vocab)+FLAGS.max_copy_size), self.question_onehot[:,:-1,:]], axis=1)


            # init embeddings
            with tf.device('/cpu:*'):
                glove_embeddings = loader.load_glove(FLAGS.data_path, d=FLAGS.embedding_size)
                embeddings_init = tf.constant(loader.get_embeddings(self.vocab, glove_embeddings, D=FLAGS.embedding_size))
                self.embeddings = tf.get_variable('word_embeddings', initializer=embeddings_init, dtype=tf.float32)
                if FLAGS.loc_embeddings:
                    self.copy_embeddings = tf.get_variable('copy_embeddings', shape=(FLAGS.max_copy_size, FLAGS.embedding_size), dtype=tf.float32)
                else:
                    self.copy_embeddings = tf.nn.embedding_lookup(self.embeddings, tf.tile([self.vocab[OOV]], [FLAGS.max_copy_size]))
                self.full_embeddings = tf.concat([self.embeddings, self.copy_embeddings], axis=0)
                assert self.embeddings.shape == [len(self.vocab), self.embedding_size]

                # this uses a load of memory, dont create unless it's actually needed
                if self.use_embedding_loss:
                    self.glove_vocab = loader.get_glove_vocab(FLAGS.data_path, size=-1, d=FLAGS.embedding_size, filter_to_squad=True)
                    extended_embeddings_init = tf.constant(loader.get_embeddings(self.glove_vocab, glove_embeddings, D=FLAGS.embedding_size))
                    self.extended_embeddings = tf.get_variable('full_word_embeddings', initializer=extended_embeddings_init, dtype=tf.float32, trainable=False)

                self.question_teach_ids = tf.concat([tf.tile(tf.constant(self.vocab[SOS], shape=[1, 1]), [curr_batch_size, 1]), self.question_ids[:, :-1]], axis=1)
                self.question_teach_embedded = tf.nn.embedding_lookup(self.full_embeddings, self.question_teach_ids)

                del glove_embeddings

            # First, coerce them to the shortlist vocab. Then embed
            self.context_coerced = tf.where(tf.greater_equal(self.context_ids, len(self.vocab)), tf.tile(tf.constant([[self.vocab[OOV]]]), tf.shape(self.context_ids)), self.context_ids)
            self.context_embedded = tf.nn.embedding_lookup(self.embeddings, self.context_coerced)

            self.answer_coerced = tf.where(tf.greater_equal(self.answer_ids, len(self.vocab)), tf.tile(tf.constant([[self.vocab[OOV]]]), tf.shape(self.answer_ids)), self.answer_ids)
            self.answer_embedded = tf.nn.embedding_lookup(self.embeddings, self.answer_coerced) # batch x seq x embed

            # Is context token in answer?
            max_context_len = tf.reduce_max(self.context_length)
            context_ix = tf.tile(tf.expand_dims(tf.range(max_context_len),axis=0), [curr_batch_size,1])
            gt_start = tf.greater_equal(context_ix, tf.tile(tf.expand_dims(self.answer_locs[:,0],axis=1), [1, max_context_len]))
            lt_end = tf.less(context_ix, tf.tile(tf.expand_dims(self.answer_locs[:,0]+self.answer_length,axis=1), [1, max_context_len]))
            self.in_answer_feature = tf.expand_dims(tf.cast(tf.logical_and(gt_start, lt_end), tf.float32),axis=2)

            embed_feats =[self.context_embedded, self.in_answer_feature]

            if FLAGS.begin_ans_feat:
                self.begin_ans_feat = tf.expand_dims(tf.one_hot(self.answer_locs[:,0], depth=max_context_len), axis=2)
                embed_feats.append(self.begin_ans_feat)

            # augment embedding
            self.context_embedded = tf.concat(embed_feats, axis=2)

        # Build encoder for context
        # Build RNN cell for encoder
        with tf.variable_scope('context_encoder'):
            context_encoder_cell_fwd = tf.contrib.rnn.DropoutWrapper(
                    cell=tf.nn.rnn_cell.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(num_units=self.context_encoder_units) for n in range(FLAGS.ctxt_encoder_depth)]),
                    input_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.)),
                    state_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.)),
                    output_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.)),
                    input_size=self.embedding_size+1+(1 if FLAGS.begin_ans_feat else 0),
                    variational_recurrent=True,
                    dtype=tf.float32)
            context_encoder_cell_bwd = tf.contrib.rnn.DropoutWrapper(
                    cell=tf.nn.rnn_cell.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(num_units=self.context_encoder_units) for n in range(FLAGS.ctxt_encoder_depth)]),
                    input_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.)),
                    state_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.)),
                    output_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.)),
                    input_size=self.embedding_size+1+(1 if FLAGS.begin_ans_feat else 0),
                    variational_recurrent=True,
                    dtype=tf.float32)

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

            self.indices = self.answer_locs

            # cap the indices to be valid
            self.indices = tf.minimum(self.indices, tf.tile(tf.expand_dims(self.context_length-1,axis=1),[1,tf.reduce_max(self.answer_length)]))

            batch_ix = tf.expand_dims(tf.transpose(tf.tile(tf.expand_dims(tf.range(curr_batch_size),axis=0),[seq_length,1]),[1,0]),axis=2)
            full_ix = tf.concat([batch_ix,tf.expand_dims(self.indices,axis=-1)], axis=2)
            self.context_condition_encoding = tf.gather_nd(self.context_encoder_output, full_ix)


            self.full_condition_encoding = tf.concat([self.context_condition_encoding, self.answer_embedded], axis=2)

            a_encoder_cell_fwd = tf.contrib.rnn.DropoutWrapper(cell=tf.nn.rnn_cell.MultiRNNCell([
                    tf.contrib.rnn.BasicLSTMCell(num_units=self.answer_encoder_units) for n in range(FLAGS.ans_encoder_depth)]),
                    input_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.)),
                    state_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.)),
                    output_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.)),
                    input_size=self.context_encoder_units*2+self.embedding_size,
                    variational_recurrent=True,
                    dtype=tf.float32)
            a_encoder_cell_bwd = tf.contrib.rnn.DropoutWrapper(cell=tf.nn.rnn_cell.MultiRNNCell([
                    tf.contrib.rnn.BasicLSTMCell(num_units=self.answer_encoder_units) for n in range(FLAGS.ans_encoder_depth)]),
                    input_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.)),
                    state_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.)),
                    output_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.)),
                    input_size=self.context_encoder_units*2+self.embedding_size,
                    variational_recurrent=True,
                    dtype=tf.float32)

            # Unroll encoder RNN
            a_encoder_output_parts, a_encoder_state_parts = tf.nn.bidirectional_dynamic_rnn(
                a_encoder_cell_fwd, a_encoder_cell_bwd, self.full_condition_encoding,
                sequence_length=self.answer_length, dtype=tf.float32)

            # This is actually wrong! It should take last element of the fwd RNN, and first element of the bwd RNN. It doesn't seem to matter in experiments, and fixing it would be a breaking change.
            # self.a_encoder_final_state = tf.concat([ops.get_last_from_seq(a_encoder_output_parts[0], self.answer_length-1), ops.get_last_from_seq(a_encoder_output_parts[1], self.answer_length-1)], axis=1)
            # Fixed!
            self.a_encoder_final_state = tf.concat([ops.get_last_from_seq(a_encoder_output_parts[0], self.answer_length-1), a_encoder_output_parts[1][:,0,:]], axis=1)

        # build init state
        with tf.variable_scope('decoder_initial_state'):
            L = tf.get_variable('decoder_L', [self.context_encoder_units*2, self.context_encoder_units*2], initializer=tf.glorot_uniform_initializer(), dtype=tf.float32)
            W0 = tf.get_variable('decoder_W0', [self.context_encoder_units*2, self.decoder_units], initializer=tf.glorot_uniform_initializer(), dtype=tf.float32)
            b0 = tf.get_variable('decoder_b0', [self.decoder_units], initializer=tf.zeros_initializer(), dtype=tf.float32)

            # This is a bit cheeky - this should be injected by the more advanced model. Consider refactoring into separate methods then overloading the one that handles this
            if self.advanced_condition_encoding:
                self.context_encoding = self.a_encoder_final_state # this would be the maluuba model
            else:
                self.context_encoding = tf.reduce_mean(self.context_condition_encoding, axis=1) # this is the baseline model

            r = tf.reduce_sum(self.context_encoder_output, axis=1)/tf.expand_dims(tf.cast(self.context_length,tf.float32),axis=1) + tf.matmul(self.context_encoding,L)
            self.s0 = tf.nn.tanh(tf.matmul(r,W0) + b0)

        if self.advanced_condition_encoding and FLAGS.full_context_encoding:
            # for Maluuba model, decoder inputs are concat of context and answer encoding
            # Strictly speaking this is still wrong - the attn mech uses only the context encoding
            self.context_encoder_output = tf.concat([self.context_encoder_output, tf.tile(tf.expand_dims(self.a_encoder_final_state,axis=1),[1,max_context_len,1])], axis=2)

        # decode
        with tf.variable_scope('decoder_init'):

            beam_memory = tf.contrib.seq2seq.tile_batch( self.context_encoder_output, multiplier=FLAGS.beam_width )
            beam_memory_sequence_length = tf.contrib.seq2seq.tile_batch( self.context_length, multiplier=FLAGS.beam_width)
            s0_tiled = tf.contrib.seq2seq.tile_batch( self.s0, multiplier=FLAGS.beam_width)
            beam_init_state = tf.contrib.rnn.LSTMStateTuple(s0_tiled, tf.contrib.seq2seq.tile_batch(tf.zeros([curr_batch_size, self.decoder_units]), multiplier=FLAGS.beam_width))

            train_memory = self.context_encoder_output
            train_memory_sequence_length = self.context_length
            train_init_state = tf.contrib.rnn.LSTMStateTuple(self.s0, tf.zeros([curr_batch_size, self.decoder_units]))

        with tf.variable_scope('attn_mech') as scope:
            train_attention_mechanism = copy_attention_wrapper.BahdanauAttention(
                            num_units=self.decoder_units, memory=train_memory,
                            memory_sequence_length=train_memory_sequence_length, name='bahdanau_attn')

            if FLAGS.separate_copy_mech:
                train_copy_mechanism = copy_attention_wrapper.BahdanauAttention(
                                num_units=self.decoder_units, memory=train_memory,
                                memory_sequence_length=train_memory_sequence_length, name='bahdanau_attn_copy')
            else:
                train_copy_mechanism = train_attention_mechanism

            with tf.variable_scope('decoder_cell'):
                train_decoder_cell = tf.contrib.rnn.DropoutWrapper(
                        cell=tf.contrib.rnn.BasicLSTMCell(num_units=self.decoder_units),
                        input_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.)),
                        state_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.)),
                        output_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.)),
                        input_size=self.embedding_size+self.decoder_units//2,
                        variational_recurrent=True,
                        dtype=tf.float32)

            train_decoder_cell = copy_attention_wrapper.CopyAttentionWrapper(train_decoder_cell,
                                                                train_attention_mechanism,
                                                                attention_layer_size=self.decoder_units / 2,
                                                                alignment_history=False,
                                                                copy_mechanism=train_copy_mechanism,
                                                                output_attention=True,
                                                                initial_cell_state=train_init_state, name='copy_attention_wrapper')

            train_init_state = train_decoder_cell.zero_state(curr_batch_size*(1), tf.float32).clone(cell_state=train_init_state)

        # copy_mechanism = copy_attention_wrapper.BahdanauAttention(
        #                 num_units=self.decoder_units, memory=memory,
        #                 memory_sequence_length=memory_sequence_length)

        with tf.variable_scope('attn_mech', reuse=True) as scope:
            scope.reuse_variables()
            beam_attention_mechanism = copy_attention_wrapper.BahdanauAttention(
                            num_units=self.decoder_units, memory=beam_memory,
                            memory_sequence_length=beam_memory_sequence_length, name='bahdanau_attn')

            if FLAGS.separate_copy_mech:
                beam_copy_mechanism = copy_attention_wrapper.BahdanauAttention(
                                num_units=self.decoder_units, memory=beam_memory,
                                memory_sequence_length=beam_memory_sequence_length, name='bahdanau_attn_copy')
            else:
                beam_copy_mechanism = beam_attention_mechanism

            with tf.variable_scope('decoder_cell', reuse=True):
                beam_decoder_cell = tf.contrib.rnn.DropoutWrapper(
                        cell=tf.contrib.rnn.BasicLSTMCell(num_units=self.decoder_units),
                        input_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.)),
                        state_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.)),
                        output_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.)),
                        input_size=self.embedding_size+self.decoder_units//2,
                        variational_recurrent=True,
                        dtype=tf.float32)

            beam_decoder_cell = copy_attention_wrapper.CopyAttentionWrapper(beam_decoder_cell,
                                                                beam_attention_mechanism,
                                                                attention_layer_size=self.decoder_units / 2,
                                                                alignment_history=False,
                                                                copy_mechanism=beam_copy_mechanism,
                                                                output_attention=True,
                                                                initial_cell_state=beam_init_state, name='copy_attention_wrapper')

            beam_init_state = beam_decoder_cell.zero_state(curr_batch_size*(FLAGS.beam_width), tf.float32).clone(cell_state=beam_init_state)


        # We have to make two copies of the layer as beam search uses different shapes - but force them to share variables
        with tf.variable_scope('copy_layer') as scope:
            ans_mask = 1-tf.reshape(self.in_answer_feature,[curr_batch_size,-1])
            self.answer_mask = tf.cond(self.hide_answer_in_copy, lambda: ans_mask, lambda: tf.ones(tf.shape(ans_mask)))

            train_projection_layer = copy_layer.CopyLayer(FLAGS.decoder_units//2, FLAGS.max_context_len,
                                            switch_units=FLAGS.switch_units,
                                            source_provider=lambda: self.context_copy_ids if FLAGS.context_as_set else self.context_ids,
                                            source_provider_sl=lambda: self.context_ids,
                                            condition_encoding=lambda: self.context_encoding,
                                            vocab_size=len(self.vocab),
                                            training_mode=self.is_training,
                                            output_mask=lambda: self.answer_mask,
                                            context_as_set=FLAGS.context_as_set,
                                            max_copy_size=FLAGS.max_copy_size,
                                            mask_oovs=tf.logical_not(self.is_training),
                                            name="copy_layer")

            scope.reuse_variables()
            answer_mask_beam = tf.contrib.seq2seq.tile_batch(self.answer_mask, multiplier=FLAGS.beam_width)


            beam_projection_layer = copy_layer.CopyLayer(FLAGS.decoder_units//2, FLAGS.max_context_len,
                                            switch_units=FLAGS.switch_units,
                                            source_provider=lambda: self.context_copy_ids if FLAGS.context_as_set else self.context_ids,
                                            source_provider_sl=lambda: self.context_ids,
                                            condition_encoding=lambda: self.context_encoding,
                                            vocab_size=len(self.vocab),
                                            training_mode=self.is_training,
                                            output_mask=lambda: answer_mask_beam,
                                            context_as_set=FLAGS.context_as_set,
                                            max_copy_size=FLAGS.max_copy_size,
                                            mask_oovs=tf.logical_not(self.is_training),
                                            name="copy_layer")

        with tf.variable_scope('decoder_unroll') as scope:
            # Helper - training
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                self.question_teach_embedded, self.question_length)
                # self.question_teach_oh, self.question_length)
                # decoder_emb_inp, length(decoder_emb_inp)+1)

            # Decoder - training
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                train_decoder_cell, training_helper,
                initial_state=train_init_state,
                # initial_state=encoder_state
                # TODO: hardcoded FLAGS.max_copy_size is longest context in SQuAD - this will need changing for a new dataset!!!
                output_layer=train_projection_layer
                )

            # Unroll the decoder
            training_outputs, training_decoder_states,training_out_lens = tf.contrib.seq2seq.dynamic_decode(training_decoder,impute_finished=True, maximum_iterations=tf.reduce_max(self.question_length))

            training_probs=training_outputs.rnn_output

        with tf.variable_scope(scope, reuse=True):
            start_tokens = tf.tile(tf.constant([self.vocab[SOS]], dtype=tf.int32), [ curr_batch_size  ] )
            end_token = self.vocab[EOS]

            # DBS degrades to normal BS with groups=1, but my implementation is 1) probably slower and 2) wont receive updates from upstream
            if FLAGS.diverse_bs:
                beam_decoder = DiverseBeamSearchDecoder( cell = beam_decoder_cell,
                                                               embedding = self.full_embeddings,
                                                               start_tokens = start_tokens,
                                                               end_token = end_token,
                                                               initial_state = beam_init_state,
                                                               beam_width = FLAGS.beam_width,
                                                               output_layer = beam_projection_layer ,
                                                               length_penalty_weight=FLAGS.length_penalty,
                                                               num_groups=FLAGS.beam_groups,
                                                               diversity_param=FLAGS.beam_diversity)
            else:
               beam_decoder = tf.contrib.seq2seq.BeamSearchDecoder( cell = beam_decoder_cell,
                                                                  embedding = self.full_embeddings,
                                                                  start_tokens = start_tokens,
                                                                  end_token = end_token,
                                                                  initial_state = beam_init_state,
                                                                  beam_width = FLAGS.beam_width,
                                                                  output_layer = beam_projection_layer ,
                                                                  length_penalty_weight=FLAGS.length_penalty)

            beam_outputs, beam_decoder_states,beam_out_lens = tf.contrib.seq2seq.dynamic_decode(  beam_decoder,
                                                                    impute_finished=False,
                                                                   maximum_iterations=40 )

            beam_pred_ids = beam_outputs.predicted_ids[:,:,0]

            # tf1.4 (and maybe others) return -1 for parts of the sequence outside the valid length, replace this with PAD (0)
            beam_mask = tf.sequence_mask(beam_out_lens[:,0], tf.shape(beam_pred_ids)[1], dtype=tf.int32)
            beam_pred_ids = beam_pred_ids*beam_mask

            beam_pred_scores = beam_outputs.beam_search_decoder_output.scores

            # pred_ids = debug_shape(pred_ids, "pred ids")
            beam_probs = tf.one_hot(beam_pred_ids, depth=len(self.vocab)+FLAGS.max_copy_size)


        self.q_hat = training_probs#tf.nn.softmax(logits, dim=2)

        # because we've done a few logs of softmaxes, there can be some precision problems that lead to non zero probability outside of the valid vocab, fix it here:
        self.max_vocab_size = tf.tile(tf.expand_dims(self.context_vocab_size+len(self.vocab),axis=1),[1,tf.shape(self.question_onehot)[1]])
        output_mask = tf.sequence_mask(self.max_vocab_size, FLAGS.max_copy_size+len(self.vocab), dtype=tf.float32)
        # self.q_hat = self.q_hat*output_mask


        with tf.variable_scope('output'), tf.device('/cpu:*'):
            self.q_hat_ids = tf.argmax(self.q_hat,axis=2,output_type=tf.int32)
            self.a_string = ops.id_tensor_to_string(self.answer_coerced, self.rev_vocab, self.context_raw, context_as_set=FLAGS.context_as_set)
            self.q_hat_string = ops.id_tensor_to_string(self.q_hat_ids, self.rev_vocab, self.context_raw, context_as_set=FLAGS.context_as_set)

            self.q_hat_beam_ids = beam_pred_ids
            self.q_hat_beam_string = ops.id_tensor_to_string(self.q_hat_beam_ids, self.rev_vocab, self.context_raw, context_as_set=FLAGS.context_as_set)

            self.q_hat_full_beam_str = [ops.id_tensor_to_string(ids*tf.sequence_mask(beam_out_lens[:,i], tf.shape(beam_pred_ids)[1], dtype=tf.int32), self.rev_vocab, self.context_raw, context_as_set=FLAGS.context_as_set) for i,ids in enumerate(tf.unstack(beam_outputs.predicted_ids,axis=2))]
            self.q_hat_full_beam_lens = [len for len in tf.unstack(beam_out_lens,axis=1)]
            self.q_hat_beam_lens = beam_out_lens[:,0]

            self.q_gold = ops.id_tensor_to_string(self.question_ids, self.rev_vocab, self.context_raw, context_as_set=FLAGS.context_as_set)
            self._output_summaries.extend(
                [tf.summary.text("q_hat", self.q_hat_string),
                tf.summary.text("q_gold", self.q_gold),
                tf.summary.text("answer", self.answer_raw)])



        with tf.variable_scope('train_loss'):
            self.target_weights = tf.sequence_mask(
                        self.question_length, tf.shape(self.q_hat)[1], dtype=tf.float32)

            logits = ops.safe_log(self.q_hat)

            # if the switch variable is fully latent, this gets a bit fiddly - we have to sum probabilities over all correct tokens, *then* take CE loss
            # otherwise the built in fn is fine (and almost certainly faster)
            if FLAGS.latent_switch:
                self.crossent =-1*ops.safe_log(tf.reduce_sum(self.q_hat*self.question_onehot, axis=2))
            else:
                self.crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.question_ids, logits=logits)
            qlen_float = tf.cast(self.question_length, tf.float32)
            self.xe_loss = tf.reduce_mean(tf.reduce_sum(self.crossent * self.target_weights,axis=1)/qlen_float,axis=0)
            self.nll = tf.reduce_sum(self.crossent * self.target_weights,axis=1)

            # TODO: Check these should be included in baseline?
            # get sum of all probabilities for words that are also in answer
            answer_oh = tf.one_hot(self.answer_ids, depth=len(self.vocab) +FLAGS.max_copy_size)
            answer_mask = tf.tile(tf.reduce_sum(answer_oh, axis=1,keep_dims=True), [1,tf.reduce_max(self.question_length),1])
            self.suppression_loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(answer_mask * self.q_hat,axis=2)*self.target_weights,axis=1)/qlen_float,axis=0)

            # entropy maximiser
            self.entropy_loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(self.q_hat *ops.safe_log(self.q_hat),axis=2)*self.target_weights,axis=1)/qlen_float,axis=0)

            if self.use_embedding_loss:
                vocab_cap = tf.tile(tf.expand_dims(self.context_vocab_size+len(self.vocab)-1,axis=1),[1,FLAGS.max_copy_size+len(self.vocab)])
                with tf.device('/cpu:*'):
                    self.local_vocab_string = ops.id_tensor_to_string(tf.minimum(tf.tile(tf.expand_dims(tf.range(FLAGS.max_copy_size+len(self.vocab)),axis=0), [curr_batch_size,1]), vocab_cap), self.rev_vocab, self.context_raw, context_as_set=FLAGS.context_as_set)
                    self.local_vocab_to_extended = ops.string_tensor_to_id(self.local_vocab_string, self.glove_vocab)
                    self.local_embeddings = tf.reshape(tf.nn.embedding_lookup(self.extended_embeddings, self.local_vocab_to_extended), [curr_batch_size, FLAGS.max_copy_size+len(self.vocab),self.embedding_size])

                    self.q_gold_ids_extended = ops.string_tensor_to_id(self.question_raw, self.glove_vocab)

                # self.q_hat_extended = tf.matmul(self.q_hat, tf.stop_gradient(self.local_vocab_to_extended)) # batch x seq x ext_vocab

                self.q_gold_embedded_extended = tf.nn.embedding_lookup(self.extended_embeddings, self.q_gold_ids_extended)

                self.q_hat_embedded_extended = tf.matmul(self.q_hat,self.local_embeddings)
                # self.q_hat_embedded_extended = tf.matmul(self.extended_embeddings, tf.cast(self.q_hat_ids_extended, tf.int32), b_is_sparse=True)

                self.similarity = tf.reduce_sum(self.q_hat_embedded_extended * tf.stop_gradient(self.q_gold_embedded_extended), axis=-1)/(1e-5+tf.norm(self.q_gold_embedded_extended, axis=-1)*tf.norm(self.q_hat_embedded_extended, axis=-1)) # batch x seq
                self.dist = tf.reduce_sum(tf.square(self.q_hat_embedded_extended - tf.stop_gradient(self.q_gold_embedded_extended)), axis=-1)
                self._train_summaries.append(tf.summary.scalar("debug/similarities", tf.reduce_mean(tf.reduce_sum(self.similarity* self.target_weights,axis=1)/qlen_float)))
                self._train_summaries.append(tf.summary.scalar("debug/dist", tf.reduce_mean(tf.reduce_sum(self.dist* self.target_weights,axis=1)/qlen_float)))
                self.loss=tf.reduce_mean(tf.reduce_sum(tf.abs(tf.acos(self.similarity)) * self.target_weights,axis=1)/qlen_float,axis=0)
                # self.loss = tf.abs(tf.acos(self.similarity)
            else:

                self.loss = self.xe_loss + FLAGS.suppression_weight*self.suppression_loss + FLAGS.entropy_weight*self.entropy_loss

        self.shortlist_prob = tf.reduce_sum(self.q_hat[:,:,:len(self.vocab)],axis=2)*self.target_weights
        self.copy_prob = tf.reduce_sum(self.q_hat[:,:,len(self.vocab):],axis=2)*self.target_weights

        self.mean_copy_prob = tf.reduce_sum(self.copy_prob,axis=1)/qlen_float

        self._train_summaries.append(tf.summary.scalar("debug/shortlist_prob", tf.reduce_mean(tf.reduce_sum(self.shortlist_prob,axis=1)/qlen_float)))
        self._train_summaries.append(tf.summary.scalar("debug/copy_prob", tf.reduce_mean(tf.reduce_sum(self.copy_prob,axis=1)/qlen_float)))

        self._train_summaries.append(tf.summary.scalar('train_loss/xe_loss', self.xe_loss))
        self._train_summaries.append(tf.summary.scalar('train_loss/entropy_loss', self.entropy_loss))
        self._train_summaries.append(tf.summary.scalar('train_loss/suppr_loss', self.suppression_loss))





        #dont bother calculating gradients if not training
        if self.training_mode:
            # Calculate and clip gradients
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(
                gradients, 5)

            # Optimization
            if FLAGS.opt_type == "sgd":
                self.global_step = tf.train.create_global_step(self.graph)

                self.sgd_lr = 1 * tf.pow(0.5, tf.cast(tf.maximum(0, tf.cast(self.global_step, tf.int32)-8000)/1000, tf.float32))
                self._train_summaries.append(tf.summary.scalar('debug/sgd_lr', self.sgd_lr))
                self.optimizer = tf.train.GradientDescentOptimizer(self.sgd_lr).apply_gradients(
                    zip(clipped_gradients, params)) if self.training_mode else tf.no_op()
            elif FLAGS.opt_type == "sgd_mom":
                self.global_step = tf.train.create_global_step(self.graph)

                self.sgd_lr = 0.1 * tf.pow(0.5, tf.cast(tf.maximum(0, tf.cast(self.global_step, tf.int32)-16000)/2000, tf.float32))
                self._train_summaries.append(tf.summary.scalar('debug/sgd_lr', self.sgd_lr))
                momentum = tf.Variable(0.1, trainable=False)

                self.optimizer = tf.train.MomentumOptimizer(self.sgd_lr, momentum).apply_gradients(
                    zip(clipped_gradients, params)) if self.training_mode else tf.no_op()
            else:
                self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).apply_gradients(
                    zip(clipped_gradients, params)) if self.training_mode else tf.no_op()

        self.accuracy = tf.reduce_mean(tf.cast(tf.reduce_sum(self.question_onehot  * tf.contrib.seq2seq.hardmax(self.q_hat), axis=-1),tf.float32)*self.target_weights)
