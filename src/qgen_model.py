# https://arxiv.org/pdf/1705.02012.pdf

import numpy as np
import tensorflow as tf


from squad_model import SQuADModel



class QGenMaluuba(SQuADModel):
    def __init__(self, vocab, batch_size):
        super().__init__(vocab, batch_size)

    def build_model(self):

        self.build_dataset(self.batch_size)

        self.a_hat = self.a_raw
        self.loss = tf.reduce_mean(tf.cast(tf.not_equal(self.a_raw,self.a_hat),tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.a_raw,self.a_hat),tf.float32))
