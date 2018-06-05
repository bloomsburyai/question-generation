# An abstract class that provides a loader and preprocessor for the SQuAD dataset (or other context/q/a triples)

import numpy as np
import tensorflow as tf

from base_model import TFModel
from helpers.loader import OOV, PAD, EOS, SOS
import helpers.preprocessing as preprocessing


class SQuADModel(TFModel):
    def __init__(self, vocab, batch_size, training_mode=False):
        self.vocab=vocab
        self.rev_vocab = {v:k for k,v in self.vocab.items()}
        self.batch_size = batch_size

        self.training_mode = training_mode
        super().__init__()
