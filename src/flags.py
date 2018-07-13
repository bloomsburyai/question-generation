import tensorflow as tf


# config
tf.app.flags.DEFINE_boolean("testing", False, "Reduce model size for local testing")

tf.app.flags.DEFINE_string("model_type", 'MALUUBA', "Model type code")

tf.app.flags.DEFINE_boolean("restore", False, "Restore from existing chkpt?")
tf.app.flags.DEFINE_boolean("policy_gradient", False, "Train using policy gradient?")
tf.app.flags.DEFINE_boolean("glove_vocab", False, "Use glove to determine the top n words? Set false to use corpus")
tf.app.flags.DEFINE_boolean("embedding_loss", False, "Use a loss based on similarity between embeddings instead of XE")

tf.app.flags.DEFINE_integer("eval_freq", 100, "Evaluate the model after this many steps")
tf.app.flags.DEFINE_integer("num_epochs", 30, "Train the model for this many epochs")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size")
tf.app.flags.DEFINE_integer("eval_batch_size", 16, "Batch size")
tf.app.flags.DEFINE_string("data_path", './data/', "Path to dataset")
tf.app.flags.DEFINE_string("log_dir", './logs/', "Path to logs")
tf.app.flags.DEFINE_string("model_dir", './models/', "Path to checkpoints")

# hyperparams
tf.app.flags.DEFINE_integer("filter_window_size", 1, "Filter contexts down to the sentences around the answer. Set -1 to disable filtering")
tf.app.flags.DEFINE_integer("filter_max_tokens", 100, "Filter contexts down to at most this many tokens around the answer. Set -1 to disable filtering")

tf.app.flags.DEFINE_integer("max_context_len", 203, "Max context length. 768 for squad, 384 if filtered, 442 if filtered with window=1")
tf.app.flags.DEFINE_integer("max_copy_size", 203, "Max size of copy vocab to limit output distribution. 768 or 384/442 for unfilt, 265 or 132 if filtered")

tf.app.flags.DEFINE_integer("embedding_size", 200, "Dimensionality to use for learned word embeddings")
tf.app.flags.DEFINE_integer("context_encoder_units", 768, "Number of hidden units for context encoder (ie 1st stage)")
tf.app.flags.DEFINE_integer("answer_encoder_units", 768, "Number of hidden units for answer encoder (ie 2nd stage)")
tf.app.flags.DEFINE_integer("decoder_units", 768, "Number of hidden units for decoder")
tf.app.flags.DEFINE_integer("switch_units", 64, "Number of hidden units for switch network. NOTE this should be 384 according to Eric")
tf.app.flags.DEFINE_integer("vocab_size", 2000, "Shortlist vocab size")
tf.app.flags.DEFINE_float("learning_rate", 2e-4, "Optimizer learning rate")
tf.app.flags.DEFINE_float("dropout_rate", 0.3, "Dropout probability")
tf.app.flags.DEFINE_boolean("context_as_set", False, "Convert context into a set of tokens rather than list for use by copy mech. Must use copy priority if this is enabled!!")
tf.app.flags.DEFINE_boolean("copy_priority", False, "Preferentially encode q using copy priority")
tf.app.flags.DEFINE_boolean("smart_copy", True, "Use smarter heuristics to determine copy location if there are multiple choices")

tf.app.flags.DEFINE_float("length_penalty", 0.15, "TF beam search length penalty hparam")

tf.app.flags.DEFINE_integer("pg_burnin", 100, "Num steps to burn in reward whitening before updating")

tf.app.flags.DEFINE_float("lm_weight", 0.25, "Loss multiplier for LM in Maluuba model. Paper gives 0.1 alone or 0.25 joint")
tf.app.flags.DEFINE_float("qa_weight", 0.5, "Loss multiplier for QA in Maluuba model. Paper gives 1.0 alone or 0.5 joint")

# QA - MPCM hparams
tf.app.flags.DEFINE_integer("qa_vocab_size", 20000, "QA system vocab size")
tf.app.flags.DEFINE_integer("qa_encoder_units", 100, "QA system - num units in encoder LSTM")
tf.app.flags.DEFINE_integer("qa_match_units", 100, "QA system - num units in match LSTM")
tf.app.flags.DEFINE_integer("qa_num_epochs", 20, "QA num epochs")
tf.app.flags.DEFINE_integer("qa_batch_size", 32, "QA batch size")
tf.app.flags.DEFINE_float("qa_learning_rate", 1e-4, "QA LR")

# LM hparams
tf.app.flags.DEFINE_integer("lm_vocab_size", 20000, "LM vocab size")
tf.app.flags.DEFINE_integer("lm_units", 384, "LM num units")
tf.app.flags.DEFINE_integer("lm_num_epochs", 25, "LM num epochs")
tf.app.flags.DEFINE_float("lm_dropout", 0.3, "LM num epochs")


# eval params
tf.app.flags.DEFINE_integer("beam_width", 32, "Beam width for decoding")
tf.app.flags.DEFINE_integer("num_dev_samples", 5000, "How many examples to use for OOS evaluations")
