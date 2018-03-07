# An abstract class that provides a loader and preprocessor for the SQuAD dataset (or other context/q/a triples)

import numpy as np
import tensorflow as tf

from base_model import TFModel
from helpers.loader import OOV, PAD, EOS, SOS

class SQuADModel(TFModel):
    def __init__(self, vocab, batch_size):
        self.vocab=vocab
        self.batch_size = batch_size
        super().__init__()
        

    def build_dataset(self, batch_size):
        self.context_ph = tf.placeholder(tf.string, [None])
        self.qs_ph = tf.placeholder(tf.string, [None])
        self.as_ph = tf.placeholder(tf.string, [None])

        dataset = tf.data.Dataset.from_tensor_slices( (self.context_ph, self.qs_ph, self.as_ph) )

        def lookup_vocab(words, context=None):
            ids = []
            decoded_context = [w.decode() for w in context] if context is not None else []

            for w in words:
                if w.decode() in self.vocab.keys():
                    ids.append(self.vocab[w.decode()])
                elif context is not None and w.decode() in decoded_context:
                    ids.append(decoded_context.index(w.decode()))
                else:
                    ids.append(self.vocab[OOV])
            embedded = np.asarray(ids, dtype=np.int32)

            return embedded

        # processing pipeline
        # split
        dataset = dataset.map(lambda context,q,a: (tf.string_split([context]).values,tf.string_split([q]).values,tf.string_split([a]).values))

        # map each of context, q, a to their raw, ids, len
        dataset = dataset.map(lambda context,q,a: (context,q,(a, tf.py_func(lookup_vocab, [a, context], tf.int32),tf.size(a))))
        dataset = dataset.map(lambda context,q,a: (context,(q, tf.py_func(lookup_vocab, [q, context], tf.int32),tf.size(q)),a))
        dataset = dataset.map(lambda context,q,a: ((context, tf.py_func(lookup_vocab, [context], tf.int32),tf.size(context)),q,a))

        # pad out to batches
        batched_dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=((tf.TensorShape([None]),  # source vectors of unknown size
                            tf.TensorShape([None]),  # source vectors of unknown size
                            tf.TensorShape([])),     # size(source)
                           (tf.TensorShape([None]),  # target vectors of unknown size
                            tf.TensorShape([None]),  # target vectors of unknown size
                            tf.TensorShape([])),     # size(source)
                           (tf.TensorShape([None]),  # target vectors of unknown size
                            tf.TensorShape([None]),  # target vectors of unknown size
                            tf.TensorShape([]))
                            ),    # size(target)
            padding_values=((PAD,
                            self.vocab[PAD],  # source vectors padded on the right with src_eos_id
                             0),          # size(source) -- unused
                            (PAD,
                            self.vocab[PAD],  # target vectors padded on the right with tgt_eos_id
                             0),          # size(source) -- unused
                            (PAD,
                            self.vocab[PAD],  # target vectors padded on the right with tgt_eos_id
                             0)))         # size(target) -- unused


        self.iterator = batched_dataset.make_initializable_iterator()
        self.this_context, self.this_q, self.this_a = self.iterator.get_next()
        (self.context_raw, self.context_ids, self.context_length) = self.this_context
        (self.q_raw, self.q_ids, self.q_length) = self.this_q
        (self.a_raw, self.a_ids, self.a_length) = self.this_a
