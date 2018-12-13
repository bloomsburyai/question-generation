import tensorflow as tf


# config
tf.app.flags.DEFINE_boolean("testing", False, "Reduce model size for local testing")

tf.app.flags.DEFINE_string("model_type", 'RL-S2S', "Model type code")

tf.app.flags.DEFINE_boolean("restore", False, "Restore from existing chkpt?")
tf.app.flags.DEFINE_string("restore_path", None, "Restore from existing chkpt?")


tf.app.flags.DEFINE_boolean("policy_gradient", False, "Train using policy gradient?")
tf.app.flags.DEFINE_boolean("glove_vocab", False, "Use glove to determine the top n words? Set false to use corpus")
tf.app.flags.DEFINE_boolean("embedding_loss", False, "Use a loss based on similarity between embeddings instead of XE")
tf.app.flags.DEFINE_boolean("latent_switch", False, "When encoding the gold questions, use a many-hot representation to allow for full freedom in the switch variable")
tf.app.flags.DEFINE_boolean("combine_vocab", False, "Combine pointer and shortlist vocabs in copy layer")
tf.app.flags.DEFINE_boolean("lr_schedule", False, "Adjust LR according to a (hard coded) schedule")

tf.app.flags.DEFINE_integer("eval_freq", 1000, "Evaluate the model after this many steps")
tf.app.flags.DEFINE_integer("num_epochs", 25, "Train the model for this many epochs")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size")
tf.app.flags.DEFINE_integer("eval_batch_size", 16, "Batch size")
tf.app.flags.DEFINE_string("data_path", './data/', "Path to dataset")
tf.app.flags.DEFINE_string("log_dir", './logs/', "Path to logs")
tf.app.flags.DEFINE_string("model_dir", './models/', "Path to checkpoints")

# hyperparams
tf.app.flags.DEFINE_integer("filter_window_size_before", 1, "Filter contexts down to the sentences around the answer. Set -1 to disable filtering")
tf.app.flags.DEFINE_integer("filter_window_size_after", 1, "Filter contexts down to the sentences around the answer. Set -1 to disable filtering")
tf.app.flags.DEFINE_integer("filter_max_tokens", 100, "Filter contexts down to at most this many tokens around the answer. Set -1 to disable filtering")

tf.app.flags.DEFINE_integer("max_context_len", 203, "Max context length. 768 for squad, 384 if filtered, 442 if filtered with window=1")
tf.app.flags.DEFINE_integer("max_copy_size", 203, "Max size of copy vocab to limit output distribution. 768 or 384/442 for unfilt, 265 or 132 if filtered")

tf.app.flags.DEFINE_integer("embedding_size", 200, "Dimensionality to use for learned word embeddings")
tf.app.flags.DEFINE_integer("context_encoder_units", 768, "Number of hidden units for context encoder (ie 1st stage)")
tf.app.flags.DEFINE_integer("answer_encoder_units", 768, "Number of hidden units for answer encoder (ie 2nd stage)")
tf.app.flags.DEFINE_boolean("full_context_encoding", True, "Concat the context encoding with the answer encoding to give the encoder output")
tf.app.flags.DEFINE_integer("decoder_units", 768, "Number of hidden units for decoder")
tf.app.flags.DEFINE_integer("switch_units", 128, "Number of hidden units for switch network. NOTE this should be 384 according to Eric")
tf.app.flags.DEFINE_integer("ctxt_encoder_depth", 1, "Number of RNN layers in context encoder")
tf.app.flags.DEFINE_integer("ans_encoder_depth", 1, "Number of RNN layers in answer encoder (ie the Maluuba widget)")


tf.app.flags.DEFINE_integer("vocab_size", 2000, "Shortlist vocab size")
tf.app.flags.DEFINE_float("learning_rate", 2e-4, "Optimizer learning rate")
tf.app.flags.DEFINE_string("opt_type", "adam", "Optimizer")
tf.app.flags.DEFINE_float("entropy_weight", 0.01, "Weight for aux entropy loss")
tf.app.flags.DEFINE_float("suppression_weight", 0.01, "Weight for suppression loss")
tf.app.flags.DEFINE_float("dropout_rate", 0.3, "Dropout probability")
tf.app.flags.DEFINE_boolean("context_as_set", True, "Convert context into a set of tokens rather than list for use by copy mech. Must use copy priority if this is enabled!!")
tf.app.flags.DEFINE_boolean("copy_priority", False, "Preferentially encode q using copy priority")
tf.app.flags.DEFINE_boolean("smart_copy", True, "Use smarter heuristics to determine copy location if there are multiple choices")
tf.app.flags.DEFINE_boolean("separate_copy_mech", False, "Use a separate set of weights for the copy mech and attention mech")
tf.app.flags.DEFINE_boolean("begin_ans_feat", False, "Include a feature denoting the first token of the answer span")
tf.app.flags.DEFINE_boolean("maxout_pointer", False, "Use a maxout pointer network (see http://aclweb.org/anthology/D18-1424)")
tf.app.flags.DEFINE_boolean("loc_embeddings", False, "Use different embeddings for each copy position? Otherwise use OOV embedding")
tf.app.flags.DEFINE_boolean("out_vocab_cpu", False, "Place output projection on CPU - enable for big vocabs")

tf.app.flags.DEFINE_boolean("advanced_condition_encoding", False, "Use the extra layers from the Maluuba paper")


tf.app.flags.DEFINE_boolean("disable_copy", False, "Prevent the model from generating using the copy vocab")
tf.app.flags.DEFINE_boolean("disable_shortlist", False, "Prevent the model from generating using the shortlist vocab")



tf.app.flags.DEFINE_float("length_penalty", 0.05, "TF beam search length penalty hparam")

tf.app.flags.DEFINE_integer("pg_burnin", 200, "Num steps to burn in reward whitening before updating")
tf.app.flags.DEFINE_boolean("pg_dropout", False, "Use dropout when generating the examples for policy gradient")

tf.app.flags.DEFINE_float("lm_weight", 0.25, "Loss multiplier for LM in Maluuba model. Paper gives 0.1 alone or 0.25 joint")
tf.app.flags.DEFINE_float("qa_weight", 0.5, "Loss multiplier for QA in Maluuba model. Paper gives 1.0 alone or 0.5 joint")
tf.app.flags.DEFINE_float("bleu_weight", 0.0, "Loss multiplier for BLEU reward")
tf.app.flags.DEFINE_float("pg_ml_weight", 1, "Loss multiplier for maximum likelihood when doing PG")


tf.app.flags.DEFINE_float("disc_weight", 0.0, "Loss multiplier for discriminator")
tf.app.flags.DEFINE_boolean("disc_train", False, "Jointly train the discriminator along with the generator")

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
tf.app.flags.DEFINE_integer("beam_width", 16, "Beam width for decoding")
# tf.app.flags.DEFINE_integer("num_dev_samples", 4691, "How many examples to use for OOS evaluations")
tf.app.flags.DEFINE_integer("num_dev_samples", 10570, "How many examples to use for OOS evaluations")
# tf.app.flags.DEFINE_integer("num_eval_samples", 5609, "How many examples to use for evaluations")
tf.app.flags.DEFINE_integer("num_eval_samples", 11877, "How many examples to use for evaluations")
tf.app.flags.DEFINE_boolean("eval_on_dev", True, "Should the eval script use the dev set?")
tf.app.flags.DEFINE_boolean("eval_on_test", False, "Should the eval script use the test set?")
tf.app.flags.DEFINE_string("eval_model_id", "", "Run ID of the saved model to be evaluated")
tf.app.flags.DEFINE_boolean("eval_metrics", True, "Calculate metrics when evaling - disable to speed up results generation")

tf.app.flags.DEFINE_boolean("diverse_bs", False, "Use diverse BS to generate mutliple outputs")
tf.app.flags.DEFINE_integer("beam_groups", 1, "How many groups to split the beam into for DBS?")
tf.app.flags.DEFINE_float("beam_diversity", 0.5, "Diversity parameter used for rescoring paths in each group")
