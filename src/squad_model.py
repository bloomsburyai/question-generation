# An abstract class that provides a loader and preprocessor for the SQuAD dataset (or other context/q/a triples)

import numpy as np
import tensorflow as tf

from base_model import TFModel
from helpers.loader import OOV, PAD, EOS, SOS
import helpers.preprocessing as preprocessing


class SQuADModel(TFModel):
    def __init__(self, vocab, batch_size):
        self.vocab=vocab
        self.rev_vocab = {v:k for k,v in self.vocab.items()}
        self.batch_size = batch_size
        super().__init__()


    def build_data_pipeline(self, batch_size):
        self.context_ph = tf.placeholder(tf.string, [None])
        self.qs_ph = tf.placeholder(tf.string, [None])
        self.as_ph = tf.placeholder(tf.string, [None])
        self.a_pos_ph = tf.placeholder(tf.int32, [None])

        dataset = tf.data.Dataset.from_tensor_slices( (self.context_ph, self.qs_ph, self.as_ph, self.a_pos_ph) )



        # processing pipeline
        # split
        # dataset = dataset.map(lambda context,q,a:
        #                 (tf.string_split(tf.py_func(helpers.tokenise, [context], tf.string), delimiter='\x00', skip_empty=True).values,
        #                 tf.string_split(tf.py_func(helpers.tokenise, [q], tf.string), delimiter='\x00', skip_empty=True).values,
        #                 tf.string_split(tf.py_func(helpers.tokenise, [a], tf.string), delimiter='\x00', skip_empty=True).values) )

        # map each of context, q, a to their raw, ids, len
        dataset = dataset.map(lambda context,q,a,a_pos:
                    (tuple(tf.py_func(preprocessing.process_squad_context(self.vocab), [context], [tf.string, tf.int32, tf.int32])),
                    tuple(tf.py_func(preprocessing.process_squad_question(self.vocab), [q,context], [tf.string, tf.int32, tf.int32])),
                    tuple(tf.py_func(preprocessing.process_squad_answer(self.vocab), [a,a_pos,context], [tf.string, tf.int32, tf.int32, tf.int32]))
                    # q,a
                    ))
        # dataset = dataset.map(lambda context,q,a: (context,
        #             (q,
        #              tf.py_func(helpers.lookup_vocab, [q, context], tf.int32),
        #              tf.size(q))
        #              ,a))
        # dataset = dataset.map(lambda context,q,a: (
        #             (context,
        #              tf.py_func(helpers.lookup_vocab, [context], tf.int32),
        #              tf.size(context))
        #              ,q,a))

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
                            tf.TensorShape([]),
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
                             0, # size(target) -- unused
                             0)))


        self.iterator = batched_dataset.make_initializable_iterator()
        self.this_context, self.this_question, self.this_answer = self.iterator.get_next()
        (self.context_raw, self.context_ids, self.context_length) = self.this_context
        (self.question_raw, self.question_ids, self.question_length) = self.this_question
        (self.answer_raw, self.answer_ids, self.answer_length, self.answer_pos) = self.this_answer
