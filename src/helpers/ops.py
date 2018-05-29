import tensorflow as tf

EPSILON = tf.cast(tf.keras.backend.epsilon(), tf.float32)

def safe_log(x):
    return tf.log(tf.clip_by_value(x, EPSILON, 1-EPSILON))


def ids_to_string(rev_vocab):
    def _ids_to_string(ids, context):
        row_str=[]
        for i,row in enumerate(ids):
            # print(context[i])
            context_tokens = [w.decode() for w in context[i].tolist()]
            out_str = []
            for j in row:
                if j< len(rev_vocab):
                    out_str.append(rev_vocab[j])
                else:
                    out_str.append(context_tokens[j-len(rev_vocab)])
            row_str.append(out_str)

            # print(context_tokens)
            # print(out_str)
        return [row_str]
    return _ids_to_string

def id_tensor_to_string(ids, rev_vocab, context):

    return tf.py_func(ids_to_string(rev_vocab), [ids, context], tf.string)
