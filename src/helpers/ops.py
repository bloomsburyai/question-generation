import tensorflow as tf
import numpy as np
from helpers.loader import EOS,PAD,OOV

def safe_log(x):
    EPSILON = tf.cast(tf.keras.backend.epsilon(), tf.float32)
    return tf.log(tf.clip_by_value(x, EPSILON, 1-EPSILON))

def log2(x):
  numerator = safe_log(x)
  denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
  return numerator / denominator

def ids_to_string(rev_vocab, context_as_set=False):
    def _ids_to_string(ids, context):
        row_str=[]
        for i,row in enumerate(ids):
            # print(context[i])
            context_tokens = [w.decode() for w in context[i].tolist()]
            if context_as_set:
                context_set = sorted(set(context_tokens)-{EOS,PAD})
            out_str = []
            for j in row:
                if j <0:
                    print("Negative token id!")
                    print(row)
                    exit()
                elif j< len(rev_vocab):
                    out_str.append(rev_vocab[j])
                elif not context_as_set and j < len(rev_vocab)+len(context_tokens):
                    out_str.append(context_tokens[j-len(rev_vocab)])
                elif context_as_set and j < len(rev_vocab)+len(context_set):
                    out_str.append(context_set[j-len(rev_vocab)])
                else:
                    print("Token ID out of range of vocab")
                    print(j, len(rev_vocab), len(context_tokens))
                    if context_as_set:
                        print(len(context_set))

            row_str.append(out_str)

            # print(context_tokens)
            # print(out_str)
        return [row_str]
    return _ids_to_string

def id_tensor_to_string(ids, rev_vocab, context, context_as_set=False):
    return tf.py_func(ids_to_string(rev_vocab, context_as_set), [ids, context], tf.string)

# This is v similar to lookup_vocab(), but works on tensors and doesnt use a context
def string_to_ids(vocab):
    def _string_to_ids(words):
        ids =[]
        for row in words.tolist():
            ids_row=[]
            for w in row:
                w=w.decode()
                if w in vocab.keys():
                    ids_row.append(vocab[w])
                else:
                    ids_row.append(vocab[OOV])
            ids.append(ids_row)
        return np.asarray(ids, dtype=np.int32)
    return _string_to_ids

def string_tensor_to_id(words, vocab):
    return tf.py_func(string_to_ids(vocab), [words], tf.int32)

def byte_token_array_to_str(batch, lengths=None, is_array=True):
    if is_array and lengths is not None:
        lengths = lengths.tolist()
    return [" ".join([w.decode() for w in toks[:(lengths[i] if lengths is not None else len(toks[i]))]]) for i,toks in enumerate(batch.tolist() if is_array else batch)]

def get_last_from_seq(seq, lengths): # seq is batch x time  x dim
    lengths = tf.maximum(lengths, tf.zeros_like(lengths, dtype=tf.int32))

    batch_size = tf.shape(lengths)[0]
    batch_nums = tf.range(0, limit=batch_size) # shape (batch_size)
    indices = tf.stack((batch_nums, lengths), axis=1) # shape (batch_size, 2)
    result = tf.gather_nd(seq, indices)
    return result # [batch_size, dim]

def total_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("Total number of trainable parameters: {}".format(total_parameters))
