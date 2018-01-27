# https://arxiv.org/pdf/1705.02012.pdf

import numpy as np
import tensorflow as tf

import utils.loader

# config
train = True


# load dataset
train_data = loader.load_squad_dataset(False)
dev_data = loader.load_squad_dataset(True)
