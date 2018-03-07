# Implementation of
# Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation
# https://arxiv.org/pdf/1406.1078.pdf

# Follwing https://github.com/tensorflow/nmt#bibtex

import tensorflow as tf
import numpy as np

import random,os

import helpers.loader as loader

# Limit GPU usage for shared environments
os.environ["CUDA_VISIBLE_DEVICES"]="3"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)


# Load europarl
# europarl-en-50k.txt
# europarl-v7.fr-en.en
# en_sents, en_vocab = loader.load_multiline('../data/europarl/europarl-v7.fr-en.en')
# fr_sents, fr_vocab = loader.load_multiline('../data/europarl/europarl-v7.fr-en.fr')
# en_sents, en_vocab = loader.load_multiline('../data/europarl/europarl-en-50k.txt')
# fr_sents, fr_vocab = loader.load_multiline('../data/europarl/europarl-fr-50k.txt')


en_sents, fr_sents, en_vocab, fr_vocab = loader.load_multiline_aligned('../data/europarl/europarl-en-50k.txt', '../data/europarl/europarl-fr-50k.txt')

en_vocab_rev = {v:k for k,v in en_vocab.items()}
fr_vocab_rev = {v:k for k,v in fr_vocab.items()}


# kill blanks, but keep synchronised - use while loop so we can edit in place
# i=0
# while i < len(en_sents):
#     line = en_sents[i]
#     if sum(line) <= en_vocab[loader.SOS] + en_vocab[loader.EOS]:
#         en_sents = np.delete(en_sents, (i), axis=0)
#         fr_sents = np.delete(fr_sents, (i), axis=0)
#     i+=1
# i=0
# while i < len(fr_sents):
#     line = fr_sents[i]
#     if sum(line) <= fr_vocab[loader.SOS] + fr_vocab[loader.EOS]:
#         en_sents = np.delete(en_sents, (i), axis=0)
#         fr_sents = np.delete(fr_sents, (i), axis=0)
#     i+=1
print('Data loaded')

max_in_seq_len = en_sents.shape[1]
max_out_seq_len = fr_sents.shape[1]
n = en_sents.shape[0]

src_vocab_size = len(fr_vocab)
tgt_vocab_size = len(en_vocab)

assert en_sents.shape[0] == fr_sents.shape[0]

print(max_in_seq_len, n)

# model params
embedding_size = 2**4
num_units=2**5
rnn_depth = 1
dropout_prob=0.2

batch_size = 1
num_epochs=100
beam_width=5


training_mode = True

# get length of embedded sequence, assuming zero embedding for pad token
def length(sequence, time_major=False):
  used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
  length = tf.reduce_sum(used, (0 if time_major else 1))
  length = tf.cast(length, tf.int32)
  return length



def lrelu(x, alpha=0.1):
  return tf.maximum(x, alpha * x)

# define graph
# placeholders
in_sent_raw = tf.placeholder(tf.int64, [None, max_in_seq_len])
out_sent_raw = tf.placeholder(tf.int64,[None, max_out_seq_len])
use_dropout = tf.placeholder_with_default(False,(), 'use_dropout')

# do some preprocessing - get lengths, and remove padding for target output
curr_batch_size = tf.shape(in_sent_raw)[0]
out_sent_in = tf.gather(out_sent_raw, tf.range(tf.reduce_max(tf.reduce_sum(tf.cast(tf.not_equal(out_sent_raw,fr_vocab['<PAD>']),tf.int32),axis=1))),axis=1)
train_out_max_len = tf.reduce_max(tf.reduce_sum(tf.cast(tf.not_equal(out_sent_in,fr_vocab['<PAD>']),tf.int32),axis=1))

# Remove the start token from non training sequences. This is ugly, it would be better to just add when it's needed
out_sent_tgt = tf.concat([out_sent_in[:,1:], tf.zeros([curr_batch_size,1],dtype=tf.int64)],1) # remove the SoS token
in_sent = tf.concat([in_sent_raw[:,1:], tf.zeros([curr_batch_size,1],dtype=tf.int64)],1) # remove the SoS token

in_sent = tf.Print(in_sent, [tf.reduce_min(tf.reduce_sum(tf.cast(tf.not_equal(in_sent,en_vocab['<PAD>']),tf.int32),axis=1), axis=0)], 'in sent min len')


# curr_batch_size = tf.Print(curr_batch_size, [tf.shape(in_sent)], 'in size', first_n=1)

# Embedding - encoder
embedding_encoder = tf.get_variable("encoder_embed", [src_vocab_size-1, embedding_size], initializer=tf.orthogonal_initializer())
embedding_encoder = tf.concat([tf.zeros([1,embedding_size]), embedding_encoder],0)
masked_target_in = tf.where(tf.greater_equal(in_sent, len(en_vocab)),  tf.tile(tf.constant([[en_vocab['<OOV>']]],dtype=tf.int64), tf.shape(in_sent)), in_sent)


encoder_emb_inp = tf.nn.embedding_lookup(
    embedding_encoder, masked_target_in)

# Embedding - decoder
embedding_decoder = tf.get_variable("decoder_embed", [tgt_vocab_size-1, embedding_size], initializer=tf.orthogonal_initializer())
embedding_decoder = tf.concat([tf.zeros([1,embedding_size]), embedding_decoder],0)

masked_target_out = tf.where(tf.greater_equal(out_sent_in, len(fr_vocab)),  tf.tile(tf.constant([[fr_vocab['<OOV>']]],dtype=tf.int64), tf.shape(out_sent_in)), out_sent_in)
masked_target_out = tf.Print(masked_target_in, [tf.shape(out_sent_in), tf.shape(masked_target_out)], first_n=1)

decoder_emb_inp = tf.nn.embedding_lookup(
    embedding_decoder, masked_target_out)


# Build RNN cell for encoder
encoder_cell = tf.nn.rnn_cell.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(
        cell=tf.contrib.rnn.GRUCell(num_units=num_units),
        input_keep_prob=(tf.cond(use_dropout,lambda: 1.0 - dropout_prob,lambda: 1.))) for n in range(rnn_depth)])

# Unroll encoder RNN
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
    encoder_cell, encoder_emb_inp,
    sequence_length=length(encoder_emb_inp), initial_state = encoder_cell.zero_state(curr_batch_size, tf.float32))


# Build RNN cell for decoder


attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=num_units, memory=encoder_outputs,
                memory_sequence_length=length(encoder_emb_inp))
decoder_cell = tf.nn.rnn_cell.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(
        cell=tf.contrib.rnn.GRUCell(num_units=num_units),
        input_keep_prob=(tf.cond(use_dropout,lambda: 1.0 - dropout_prob,lambda: 1.))) for n in range(rnn_depth)])
decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size=num_units / 2, alignment_history=True)



# project RNN outputs into vocab space
projection_layer = tf.layers.Dense(
    tgt_vocab_size, use_bias=False)

# Helper - training
helper = tf.contrib.seq2seq.TrainingHelper(
    decoder_emb_inp, tf.fill([curr_batch_size], max_out_seq_len))
    # decoder_emb_inp, length(decoder_emb_inp)+1)

# Decoder - training
decoder = tf.contrib.seq2seq.BasicDecoder(
    decoder_cell, helper,
    initial_state=decoder_cell.zero_state(
                    dtype=tf.float32, batch_size=curr_batch_size)
    # initial_state=encoder_state
    )

# Unroll the decoder
outputs, decoder_states,out_lens = tf.contrib.seq2seq.dynamic_decode(decoder,impute_finished=True, maximum_iterations=max_out_seq_len)


attention = decoder_states.alignment_history.stack()

attention = tf.Print(attention, [tf.shape(attention)], 'attn shape', first_n=1) # out_seq x batch x in_seq
attention = tf.Print(attention, [tf.shape(encoder_emb_inp)], 'enc in', first_n=1) # batch x in_seq x embed
attention = tf.Print(attention, [tf.shape(out_sent_tgt)], 'out', first_n=1) # batch x out_seq
attention = tf.Print(attention, [tf.shape(out_sent_in)], 'out', first_n=1) # batch x out_seq
attention = tf.Print(attention, [out_lens], 'out_len', first_n=1) # batch x out_seq
attention = tf.Print(attention, [tf.shape(in_sent)], 'raw in', first_n=1) # batch x in_seq

context = tf.matmul( tf.transpose(attention,[1,0,2]), encoder_emb_inp) # should be batch x out_seq x embed

# switching network
#switch = sigmoid(concat(hidden_t, context_t))
context = tf.Print(context, [tf.shape(context)], 'context', first_n=1)
context = tf.Print(context, [tf.shape(outputs.rnn_output)], 'rnn out', first_n=1)
switch = tf.layers.dense(tf.concat([context, outputs.rnn_output],axis=2), 1, activation=tf.sigmoid)


# Project into vocab space
logits = projection_layer(outputs.rnn_output)
logits = tf.Print(logits, [tf.shape(logits)], 'logits_shape', first_n=1)

# build output from either shortlist vocab or pointer
pred = tf.concat([tf.nn.softmax(logits)*switch, tf.transpose(attention,[1,0,2])*(1-switch)], axis=2)
pred =tf.Print(pred, [tf.shape(pred), tf.shape(out_sent_tgt)], 'pred v tgt', first_n=1)

# set up training loss - mask padding
train_out_len = tf.reduce_sum(tf.cast(tf.not_equal(out_sent_in,fr_vocab['<PAD>']),tf.int32),axis=1)
target_weights = tf.sequence_mask(
        train_out_len, max_out_seq_len, dtype=logits.dtype)

# logits = tf.Print(logits, [length(encoder_emb_inp)], 'in_length')
# logits = tf.Print(logits, [length(decoder_emb_inp)], 'out_length')
# logits = tf.Print(logits, [tf.shape(in_sent)], 'insent')
# logits = tf.Print(logits, [out_lens], 'outlens')
# logits = tf.Print(logits, [tf.shape(outputs.sample_id)], 'sample_id')
# logits = tf.Print(logits, [tf.shape(outputs.rnn_output)], 'rnnout')
# logits = tf.Print(logits, [tf.shape(logits)], 'logits')
# logits = tf.Print(logits, [tf.shape(out_sent)], 'out')
# logits = tf.Print(logits, [tf.shape(target_weights)], 'w')

# crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=masked_target_out, logits=logits)

crossent = -tf.reduce_sum(tf.cast(tf.one_hot(out_sent_raw, depth=len(fr_vocab)+max_in_seq_len),tf.float32) * tf.log(pred), 2)
train_loss = (tf.reduce_sum(crossent * target_weights)/tf.to_float(curr_batch_size))

# inference Helper
inf_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
    embedding_decoder,
    tf.fill([curr_batch_size], fr_vocab[loader.SOS]), fr_vocab[loader.EOS])

# Replicate encoder state beam_width times
# tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(
#     encoder_state, multiplier=beam_width)
#
# decoder_initial_state = decoder_cell.zero_state(
#     dtype=tf.float32, batch_size=curr_batch_size * beam_width)
# decoder_initial_state = decoder_initial_state.clone(
#     cell_state=tiled_encoder_final_state)

# Define a beam-search decoder
# inf_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
#     cell=decoder_cell,
#     embedding=embedding_decoder,
#     start_tokens=tf.fill([curr_batch_size], fr_vocab[loader.SOS]),
#     end_token= fr_vocab['<PAD>'],
#     initial_state=decoder_initial_state,
#     beam_width=beam_width,
#     output_layer=projection_layer,
#     length_penalty_weight=0.0)

# Decoder
inf_decoder = tf.contrib.seq2seq.BasicDecoder(
    decoder_cell,
    inf_helper,
    decoder_cell.zero_state(dtype=tf.float32, batch_size=curr_batch_size ),
    output_layer=projection_layer)
# Dynamic decoding
inf_outputs, _,inf_out_lens = tf.contrib.seq2seq.dynamic_decode(
    inf_decoder, maximum_iterations=max_out_seq_len,impute_finished=True)
# translations = inf_outputs.predicted_ids[:,:,0]
translations = inf_outputs.sample_id

# Calculate and clip gradients
params = tf.trainable_variables()
gradients = tf.gradients(train_loss, params)
clipped_gradients, _ = tf.clip_by_global_norm(
    gradients, 5)

# Optimization
optimizer = tf.train.AdamOptimizer(0.001)
update_step = optimizer.apply_gradients(
    zip(clipped_gradients, params))

print('Graph built')


with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())

    for e in range(num_epochs):
        print('Training... epoch',e)
        for i in range(n // batch_size):
            en_batch = en_sents[i*batch_size : (i+1)*batch_size,:]
            fr_batch = fr_sents[i*batch_size : (i+1)*batch_size,:]
            _,loss = sess.run([update_step,train_loss], feed_dict={in_sent_raw:en_batch, out_sent_raw:fr_batch, use_dropout:True})

            if i % 250 == -1:
                print('Epoch ', e,' Step ',i,' -> ', loss)
                #  infer!
                num_infer=2
                rand_ix = random.randint(0, len(en_sents)-num_infer)
                en_batch = en_sents[rand_ix : rand_ix+ num_infer,:]
                fr_batch = fr_sents[rand_ix : rand_ix+num_infer,:]


                tgt_est_ids,this_in, this_out_tgt,this_out_train = sess.run([translations,in_sent,out_sent_tgt,out_sent_in], feed_dict={in_sent_raw:en_batch, out_sent_raw:fr_batch})
                # print(tgt_est_ids)
                for i in range(num_infer):
                    print('In, Tgt, Train decoder in, est')
                    # todo: actually get the string version of this sentence!
                    # sent_vocab={(i+len(en_vocab_rev)):'word'+str(i) for i in range(32)}
                    this_en_vocab = {**en_vocab_rev, **{(i+len(en_vocab_rev)):'word'+str(i) for i in range(32)}}
                    this_fr_vocab = {**fr_vocab_rev, **{(i+len(fr_vocab_rev)):'word'+str(i) for i in range(32)}}
                    print("  "," ".join([this_en_vocab[ix] for ix in en_batch[i,:]]))
                    print("  "," ".join([this_fr_vocab[ix] for ix in fr_batch[i,:]]))
                    # print("  "," ".join([fr_vocab_rev[ix] for ix in this_out_train[i,:]]))
                    print("  "," ".join([this_fr_vocab[ix] for ix in tgt_est_ids[i,:]]))


print('Done!')
