class FlagsObject(object):
    pass

FLAGS = FlagsObject()


# config
FLAGS.testing = False

FLAGS.model_type = 'RL-S2S'

FLAGS.restore = False
FLAGS.restore_path = None


FLAGS.policy_gradient = False
FLAGS.glove_vocab = False
FLAGS.embedding_loss = False
FLAGS.latent_switch = False
FLAGS.combine_vocab = False
FLAGS.lr_schedule = False

FLAGS.eval_freq = 1000
FLAGS.num_epochs = 25
FLAGS.batch_size = 64
FLAGS.eval_batch_size = 16
FLAGS.data_path = '../data/'
FLAGS.log_dir = '../logs/'
FLAGS.model_dir = '../models/'

# hyperparams
FLAGS.filter_window_size_before = 1
FLAGS.filter_window_size_after = 1
FLAGS.filter_max_tokens = 100

FLAGS.max_context_len = 203
FLAGS.max_copy_size = 203

FLAGS.embedding_size = 200
FLAGS.context_encoder_units = 768
FLAGS.answer_encoder_units = 768
FLAGS.full_context_encoding = True
FLAGS.decoder_units = 768
FLAGS.switch_units = 128
FLAGS.ctxt_encoder_depth = 1
FLAGS.ans_encoder_depth = 1


FLAGS.vocab_size = 2000
FLAGS.learning_rate = 2e-4
FLAGS.opt_type = "adam"
FLAGS.entropy_weight = 0.01
FLAGS.suppression_weight = 0.01
FLAGS.dropout_rate = 0.3
FLAGS.context_as_set = True
FLAGS.copy_priority = False
FLAGS.smart_copy = True
FLAGS.separate_copy_mech = False
FLAGS.begin_ans_feat = False
FLAGS.maxout_pointer = False
FLAGS.loc_embeddings = False
FLAGS.out_vocab_cpu = False

FLAGS.advanced_condition_encoding = False


FLAGS.disable_copy = False
FLAGS.disable_shortlist = False



FLAGS.length_penalty = 0.05

FLAGS.pg_burnin = 200
FLAGS.pg_dropout = False

FLAGS.lm_weight = 0.25
FLAGS.qa_weight = 0.5
FLAGS.bleu_weight = 0.0
FLAGS.pg_ml_weight = 1


FLAGS.disc_weight = 0.0
FLAGS.disc_train = False

# QA - MPCM hparams
FLAGS.qa_vocab_size = 20000
FLAGS.qa_encoder_units = 100
FLAGS.qa_match_units = 100
FLAGS.qa_num_epochs = 20
FLAGS.qa_batch_size = 32
FLAGS.qa_learning_rate = 1e-4

# LM hparams
FLAGS.lm_vocab_size = 20000
FLAGS.lm_units = 384
FLAGS.lm_num_epochs = 25
FLAGS.lm_dropout = 0.3


# eval params
FLAGS.beam_width = 16
# FLAGS.num_dev_samples = 4691
FLAGS.num_dev_samples = 10570
# FLAGS.num_eval_samples = 5609
FLAGS.num_eval_samples = 11877
FLAGS.eval_on_dev = True
FLAGS.eval_on_test = False
FLAGS.eval_model_id = ""
FLAGS.eval_metrics = True

FLAGS.diverse_bs = False
FLAGS.beam_groups = 1
FLAGS.beam_diversity = 0.5
