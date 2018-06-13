import tensorflow as tf


# config
tf.app.flags.DEFINE_boolean("testing", False, "Reduce model size for local testing")

tf.app.flags.DEFINE_boolean("train", True, "Training mode?")
tf.app.flags.DEFINE_integer("eval_freq", 100, "Evaluate the model after this many steps")
tf.app.flags.DEFINE_integer("num_epochs", 30, "Train the model for this many epochs")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size")
tf.app.flags.DEFINE_integer("eval_batch_size", 16, "Batch size")
tf.app.flags.DEFINE_string("data_path", './data/', "Path to dataset")
tf.app.flags.DEFINE_string("log_dir", './logs/', "Path to logs")
tf.app.flags.DEFINE_string("model_dir", './models/', "Path to checkpoints")

# hyperparams - these should probably be within the model?
tf.app.flags.DEFINE_integer("max_copy_size", 768, "Max context length to limit output distribution")

tf.app.flags.DEFINE_integer("embedding_size", 200, "Dimensionality to use for learned word embeddings")
tf.app.flags.DEFINE_integer("context_encoder_units", 768, "Number of hidden units for context encoder (ie 1st stage)")
tf.app.flags.DEFINE_integer("answer_encoder_units", 768, "Number of hidden units for answer encoder (ie 2nd stage)")
tf.app.flags.DEFINE_integer("decoder_units", 768, "Number of hidden units for decoder")
tf.app.flags.DEFINE_integer("vocab_size", 2000, "Shortlist vocab size")
tf.app.flags.DEFINE_float("learning_rate", 5e-4, "Optimizer learning rate")
tf.app.flags.DEFINE_float("dropout_rate", 0.3, "Dropout probability")

tf.app.flags.DEFINE_float("lm_weight", 0.25, "Loss multiplier for LM in Maluuba model. Paper gives 0.1 alone or 0.25 joint")
tf.app.flags.DEFINE_float("qa_weight", 0.5, "Loss multiplier for QA in Maluuba model. Paper gives 1.0 alone or 0.5 joint")

# QA - MPCM hparams
tf.app.flags.DEFINE_integer("qa_vocab_size", 20000, "QA system vocab size")
tf.app.flags.DEFINE_integer("qa_encoder_units", 100, "QA system - num units in encoder LSTM")
tf.app.flags.DEFINE_integer("qa_match_units", 100, "QA system - num units in match LSTM")
tf.app.flags.DEFINE_integer("qa_num_epochs", 20, "QA num epochs")
tf.app.flags.DEFINE_float("qa_learning_rate", 5e-4, "QA LR")

# LM hparams
tf.app.flags.DEFINE_integer("lm_vocab_size", 10000, "LM vocab size")
tf.app.flags.DEFINE_integer("lm_units", 512, "LM num units")
tf.app.flags.DEFINE_integer("lm_num_epochs", 5, "LM num epochs")


# eval params
tf.app.flags.DEFINE_integer("beam_width", 32, "Beam width for decoding")
