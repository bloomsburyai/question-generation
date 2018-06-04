import tensorflow as tf


# config
tf.app.flags.DEFINE_boolean("train", True, "Training mode?")
tf.app.flags.DEFINE_integer("eval_freq", 100, "Evaluate the model after this many steps")
tf.app.flags.DEFINE_integer("num_epochs", 20, "Train the model for this many epochs")
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size")
tf.app.flags.DEFINE_string("data_path", './data/', "Path to dataset")
tf.app.flags.DEFINE_string("log_dir", './logs/', "Path to logs")
tf.app.flags.DEFINE_string("model_dir", './models/', "Path to checkpoints")

tf.app.flags.DEFINE_boolean("use_gpu", False, "Is a GPU available on this system?")

# hyperparams - these should probably be within the model?
tf.app.flags.DEFINE_integer("embedding_size", 200, "Dimensionality to use for learned word embeddings")
tf.app.flags.DEFINE_integer("context_encoder_units", 768, "Number of hidden units for context encoder (ie 1st stage)")
tf.app.flags.DEFINE_integer("answer_encoder_units", 768, "Number of hidden units for answer encoder (ie 2nd stage)")
tf.app.flags.DEFINE_integer("decoder_units", 768, "Number of hidden units for decoder")
tf.app.flags.DEFINE_integer("vocab_size", 2000, "Shortlist vocab size")
tf.app.flags.DEFINE_float("learning_rate", 2e-4, "Optimizer learning rate")
tf.app.flags.DEFINE_float("dropout_rate", 0.3, "Dropout probability")

# QA - MPCM hparams
tf.app.flags.DEFINE_integer("qa_vocab_size", 10000, "QA system vocab size")
tf.app.flags.DEFINE_integer("qa_encoder_units", 100, "QA system - num units in encoder LSTM")
tf.app.flags.DEFINE_integer("qa_match_units", 100, "QA system - num units in match LSTM")

# LM hparams
tf.app.flags.DEFINE_integer("lm_vocab_size", 10000, "QA system vocab size")
tf.app.flags.DEFINE_integer("lm_units", 512, "QA system vocab size")


# eval params
tf.app.flags.DEFINE_integer("beam_width", 32, "Beam width for decoding")
