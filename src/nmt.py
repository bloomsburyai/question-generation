# Implementation of
# Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation
# https://arxiv.org/pdf/1406.1078.pdf

# Follwing https://github.com/tensorflow/nmt#bibtex

import tensorflow as tf
import numpy as np

import random,os

os.environ["CUDA_VISIBLE_DEVICES"]="3"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)


# europarl-en-50k.txt
# europarl-v7.fr-en.en
with open('../data/europarl/europarl-en-500k.txt','r') as en_file:
    en_raw = en_file.readlines()
with open('../data/europarl/europarl-fr-500k.txt','r') as fr_file:
    fr_raw = fr_file.readlines()

SOS = '<Sent>'
EOS = '<PAD>'

import re
def process_raw(raw_data, limit_length=32):
    lines = [re.sub(r'([\,\?\!\.]+)',r' \1 ', line).lower() for line in raw_data]
    # lines = re.split('[\n]+',raw_data.lower())
    vocab = {'<PAD>':0,'<OOV>':1, SOS:2}
    ids=[]
    tokenised=[]
    max_sent_len=0
    for l in lines:
        # if l[0] == '<':
        #     print(l)
        # if len(l) ==0 or l[0] == '<':
        #     continue
        # id_line=[vocab[SOS]]
        # token_line = [SOS]
        id_line=[]
        token_line = []
        for w in l.split():
            if len(id_line) > max_sent_len:
                max_sent_len = len(id_line)
            if len(id_line) >= limit_length-1:
                continue
            w = w.strip()
            if len(w) == 0:
                continue
            if w not in vocab.keys():
                vocab[w] = len(vocab)
            id_line.append(vocab[w])
            token_line.append(w)
        # if len(id_line) > 400:
        #     print(token_line)
        #     print(id_line)
        #     exit()
        id_line.append(vocab[EOS])
        token_line.append(EOS)

        # if len(token_line) >= limit_length:
        #     print(token_line)
        #     print(len(token_line))
        #     exit()
        ids.append(id_line)
        tokenised.append(token_line)

    if limit_length:
        max_sent_len = min(max_sent_len+1, limit_length)
    id_arr = np.full([len(ids), max_sent_len], vocab['<PAD>'], dtype=np.int32)

    for i, sent in enumerate(ids):
        this_len = min(len(sent), max_sent_len)
        id_arr[i,  0:this_len] = (sent if len(sent) <= max_sent_len else sent[:max_sent_len])

    return id_arr, tokenised, vocab

en_sents, en_tokenised, en_vocab = process_raw(en_raw)
fr_sents, fr_tokenised, fr_vocab = process_raw(fr_raw)

en_vocab_rev = {v:k for k,v in en_vocab.items()}
fr_vocab_rev = {v:k for k,v in fr_vocab.items()}

# kill blanks
for i,line in enumerate(en_tokenised):
    if len(line) == 0:
        en_tokenised.pop(i)
        fr_tokenised.pop(i)
        en_sents = np.delete(en_sents, (i), axis=0)
        fr_sents = np.delete(fr_sents, (i), axis=0)
for i,line in enumerate(fr_tokenised):
    if len(line) == 0:
        en_tokenised.pop(i)
        fr_tokenised.pop(i)
        en_sents = np.delete(en_sents, (i), axis=0)
        fr_sents = np.delete(fr_sents, (i), axis=0)

print('Data loaded')
# print(np.shape(en_sents))
# print(np.shape(fr_sents))

max_in_seq_len = en_sents.shape[1]
max_out_seq_len = fr_sents.shape[1]
n = en_sents.shape[0]

src_vocab_size = len(fr_vocab)
tgt_vocab_size = len(en_vocab)

assert en_sents.shape[0] == fr_sents.shape[0]

print(max_in_seq_len, n)

# model params
embedding_size = 2**6
num_units=2**7

batch_size = 32
num_epochs=20


training_mode = True


def length(sequence, time_major=False):
  used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
  length = tf.reduce_sum(used, (0 if time_major else 1))
  length = tf.cast(length, tf.int32)
  return length

def last_relevant(output, length, time_major=False):
  batch_size = tf.shape(output)[(1 if time_major else 0)]
  max_length = tf.shape(output)[(0 if time_major else 1)]
  out_size = int(output.get_shape()[2])
  index = tf.range(0, batch_size) * max_length + (length - 1)
  flat = tf.reshape(output, [-1, out_size])
  relevant = tf.gather(flat, index)
  return relevant


def lrelu(x, alpha=0.1):
  return tf.maximum(x, alpha * x)

# define graph
in_sent = tf.placeholder(tf.int64, [None, max_in_seq_len])
out_sent = tf.placeholder(tf.int64,[None, max_out_seq_len])

curr_batch_size = tf.shape(in_sent)[0]

# curr_batch_size = tf.Print(curr_batch_size, [tf.shape(in_sent)], 'in size', first_n=1)

# Embedding
oov_pad_embeddings = tf.zeros([1,embedding_size])
embedding_encoder = tf.get_variable("encoder_embed", [src_vocab_size-1, embedding_size], initializer=tf.orthogonal_initializer())
embedding_encoder = tf.concat([oov_pad_embeddings, embedding_encoder],0)
# embedding_encoder = tf.get_variable(
#     "embedding_encoder", [src_vocab_size, embedding_size], tf.float32, initializer=tf.orthogonal_initializer())
# Look up embedding:
#   encoder_inputs: [max_time, batch_size]
#   encoder_emb_inp: [max_time, batch_size, embedding_size]
encoder_emb_inp = tf.nn.embedding_lookup(
    embedding_encoder, in_sent)

# Embedding
# oov_pad_embeddings = tf.zeros([1,embedding_size])
embedding_decoder = tf.get_variable("decoder_embed", [tgt_vocab_size-1, embedding_size], initializer=tf.orthogonal_initializer())
embedding_decoder = tf.concat([oov_pad_embeddings, embedding_decoder],0)
# Look up embedding:
#   encoder_inputs: [max_time, batch_size]
#   encoder_emb_inp: [max_time, batch_size, embedding_size]
decoder_emb_inp = tf.nn.embedding_lookup(
    embedding_decoder, out_sent)

# Build RNN cell
encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

# Run Dynamic RNN
#   encoder_outpus: [max_time, batch_size, num_units]
#   encoder_state: [batch_size, num_units]
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
    encoder_cell, encoder_emb_inp,
    sequence_length=length(encoder_emb_inp), initial_state = encoder_cell.zero_state(curr_batch_size, tf.float32))


# Build RNN cell
decoder_cell = tf.nn.rnn_cell.GRUCell(num_units)


# Helper
if training_mode:
    helper = tf.contrib.seq2seq.TrainingHelper(
        # decoder_emb_inp, tf.fill([curr_batch_size], max_out_seq_len))
        decoder_emb_inp, length(decoder_emb_inp))
else:
    exit('Not implemented yet')

projection_layer = tf.layers.Dense(
    tgt_vocab_size, use_bias=False)

# Decoder
decoder = tf.contrib.seq2seq.BasicDecoder(
    decoder_cell, helper, encoder_state)
# Dynamic decoding
outputs, _,out_lens = tf.contrib.seq2seq.dynamic_decode(decoder,impute_finished=True, maximum_iterations=max_out_seq_len)
logits = projection_layer(outputs.rnn_output)

target_weights = tf.sequence_mask(
        length(decoder_emb_inp)+1, max_out_seq_len, dtype=logits.dtype)

# logits = tf.Print(logits, [length(encoder_emb_inp)], 'in_length')
# logits = tf.Print(logits, [tf.shape(in_sent)], 'insent')
# logits = tf.Print(logits, [out_lens], 'outlens')
# logits = tf.Print(logits, [tf.shape(outputs.sample_id)], 'sample_id')
# logits = tf.Print(logits, [tf.shape(outputs.rnn_output)], 'rnnout')
# logits = tf.Print(logits, [tf.shape(logits)], 'logits')
# logits = tf.Print(logits, [tf.shape(out_sent)], 'out')
# logits = tf.Print(logits, [tf.shape(target_weights)], 'w')

crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=out_sent, logits=logits)
train_loss = (tf.reduce_sum(crossent * target_weights)/tf.to_float(curr_batch_size))



# weights = tf.to_float(tf.not_equal(out_sent[:, :], fr_vocab['<PAD>']))
#
#
# train_loss = tf.contrib.seq2seq.sequence_loss(
#         logits, out_sent, weights=weights)

# inference Graph# Helper
inf_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
    embedding_decoder,
    tf.fill([curr_batch_size], fr_vocab[SOS]), fr_vocab['<PAD>'])

# Decoder
inf_decoder = tf.contrib.seq2seq.BasicDecoder(
    decoder_cell, inf_helper, encoder_state,
    output_layer=projection_layer)
# Dynamic decoding
inf_outputs, _,inf_out_lens = tf.contrib.seq2seq.dynamic_decode(
    inf_decoder, maximum_iterations=max_out_seq_len,impute_finished=True)
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
            _,loss = sess.run([update_step,train_loss], feed_dict={in_sent:fr_batch, out_sent:en_batch})
            if i % 250 == 0:
                print('Epoch ', e,' Step ',i,' -> ', loss)

                #  infer!
                num_infer=2
                rand_ix = random.randint(0, len(en_tokenised)-num_infer)
                en_batch = en_sents[rand_ix : rand_ix+ num_infer,:]
                fr_batch = fr_sents[rand_ix : rand_ix+num_infer,:]

                tgt_est_ids = sess.run(translations, feed_dict={in_sent:fr_batch, out_sent:en_batch})
                for i in range(num_infer):
                    print(" ".join([en_vocab_rev[ix] for ix in en_sents[rand_ix+i,:]]))
                    print(" ".join([fr_vocab_rev[ix] for ix in fr_sents[rand_ix+i,:]]))
                    # print(tgt_est_ids[0])
                    print(" ".join([fr_vocab_rev[ix] for ix in tgt_est_ids[i,:]]))
print('Done!')
