import tensorflow as tf



def safe_log(x):
    EPSILON = tf.cast(tf.keras.backend.epsilon(), tf.float32)
    return tf.log(tf.clip_by_value(x, EPSILON, 1-EPSILON))


def ids_to_string(rev_vocab):
    def _ids_to_string(ids, context):
        row_str=[]
        for i,row in enumerate(ids):
            # print(context[i])
            context_tokens = [w.decode() for w in context[i].tolist()]
            out_str = []
            for j in row:
                if j <0:
                    print("Negative token id!")
                    print(row)
                    exit()
                elif j< len(rev_vocab):
                    out_str.append(rev_vocab[j])
                elif j < len(rev_vocab)+len(context_tokens):
                    out_str.append(context_tokens[j-len(rev_vocab)])
                else:
                    print("Token ID out of range of vocab")
                    print(j, len(rev_vocab), len(context_tokens))

            row_str.append(out_str)

            # print(context_tokens)
            # print(out_str)
        return [row_str]
    return _ids_to_string

def id_tensor_to_string(ids, rev_vocab, context):

    return tf.py_func(ids_to_string(rev_vocab), [ids, context], tf.string)

def get_last_from_seq(seq, lengths): # seq is batch x time  x dim
    lengths = tf.maximum(lengths, tf.zeros_like(lengths, dtype=tf.int32))

    batch_size = tf.shape(lengths)[0]
    batch_nums = tf.range(0, limit=batch_size) # shape (batch_size)
    indices = tf.stack((batch_nums, lengths), axis=1) # shape (batch_size, 2)
    result = tf.gather_nd(seq, indices)
    return result # [batch_size, dim]
