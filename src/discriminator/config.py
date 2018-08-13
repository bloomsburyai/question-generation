import os
import tensorflow as tf

'''
This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''

from qa.qanet.prepro import prepro

flags = tf.flags

home = os.getcwd()
train_file = os.path.join(home, "datasets", "squad", "train-v1.1.json")
dev_file = os.path.join(home, "datasets", "squad", "dev-v1.1.json")
test_file = os.path.join(home, "datasets", "squad", "dev-v1.1.json")
glove_word_file = os.path.join(home, "datasets", "glove", "glove.840B.300d.txt")

train_dir = "train"
model_name = "FRC"
dir_name = os.path.join(train_dir, model_name)
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(os.path.join(os.getcwd(),dir_name)):
    os.mkdir(os.path.join(os.getcwd(),dir_name))

# TODO: This needs to be passed in from the cmd line somehow - but flags doesnt exist yet because all the logic happens when the defaults are being built, grr
if "CLUSTER" in os.environ.keys():
    target_dir = "/home/thosking/msc-project/models/saved/discriminator"
else:
    target_dir = "./models/saved/discriminator"

log_dir = os.path.join(dir_name, "event")
save_dir = os.path.join(dir_name, "model")
answer_dir = os.path.join(dir_name, "answer")
train_record_file = os.path.join(target_dir, "train.tfrecords")
dev_record_file = os.path.join(target_dir, "dev.tfrecords")
test_record_file = os.path.join(target_dir, "test.tfrecords")
word_emb_file = os.path.join(target_dir, "word_emb.json")
char_emb_file = os.path.join(target_dir, "char_emb.json")
train_eval = os.path.join(target_dir, "train_eval.json")
dev_eval = os.path.join(target_dir, "dev_eval.json")
test_eval = os.path.join(target_dir, "test_eval.json")
dev_meta = os.path.join(target_dir, "dev_meta.json")
test_meta = os.path.join(target_dir, "test_meta.json")
word_dictionary = os.path.join(target_dir, "word_dictionary.json")
char_dictionary = os.path.join(target_dir, "char_dictionary.json")
answer_file = os.path.join(answer_dir, "answer.json")

# if not os.path.exists(target_dir):
#     os.makedirs(target_dir)
# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# if not os.path.exists(answer_dir):
#     os.makedirs(answer_dir)

flags.DEFINE_string("disc_mode", "train", "Running mode train/debug/test")

flags.DEFINE_string("disc_target_dir", target_dir, "Target directory for out data")
# flags.DEFINE_string("disc_log_dir", log_dir, "Directory for tf event")
flags.DEFINE_string("disc_save_dir", save_dir, "Directory for saving model")
flags.DEFINE_string("disc_train_file", train_file, "Train source file")
flags.DEFINE_string("disc_dev_file", dev_file, "Dev source file")
flags.DEFINE_string("disc_test_file", test_file, "Test source file")
flags.DEFINE_string("disc_glove_word_file", glove_word_file, "Glove word embedding source file")

flags.DEFINE_string("disc_train_record_file", train_record_file, "Out file for train data")
flags.DEFINE_string("disc_dev_record_file", dev_record_file, "Out file for dev data")
flags.DEFINE_string("disc_test_record_file", test_record_file, "Out file for test data")
flags.DEFINE_string("disc_word_emb_file", word_emb_file, "Out file for word embedding")
flags.DEFINE_string("disc_char_emb_file", char_emb_file, "Out file for char embedding")
flags.DEFINE_string("disc_train_eval_file", train_eval, "Out file for train eval")
flags.DEFINE_string("disc_dev_eval_file", dev_eval, "Out file for dev eval")
flags.DEFINE_string("disc_test_eval_file", test_eval, "Out file for test eval")
flags.DEFINE_string("disc_dev_meta", dev_meta, "Out file for dev meta")
flags.DEFINE_string("disc_test_meta", test_meta, "Out file for test meta")
flags.DEFINE_string("disc_answer_file", answer_file, "Out file for answer")
flags.DEFINE_string("disc_word_dictionary", word_dictionary, "Word dictionary")
flags.DEFINE_string("disc_char_dictionary", char_dictionary, "Character dictionary")

flags.DEFINE_boolean("disc_trainongenerated", True, "Train on generated Qs")
flags.DEFINE_boolean("disc_trainonsquad", False, "Train on squad v2")
flags.DEFINE_string("disc_modelslug", "MALUUBA_CROP_GLOVE_SMART_train", "Model ID slug to use for pre-generated Qs")
flags.DEFINE_boolean("disc_dev_set", False, "Use dev set")
flags.DEFINE_boolean("disc_init_qanet", False, "Init weights using trained qanet instance")

flags.DEFINE_integer("disc_glove_char_size", 94, "Corpus size for Glove")
flags.DEFINE_integer("disc_glove_word_size", int(2.2e6), "Corpus size for Glove")
flags.DEFINE_integer("disc_glove_dim", 300, "Embedding dimension for Glove")
flags.DEFINE_integer("disc_char_dim", 64, "Embedding dimension for char")

flags.DEFINE_integer("disc_para_limit", 900, "Limit length for paragraph")
flags.DEFINE_integer("disc_ques_limit", 260, "Limit length for question - NOTE QANet uses a different tokenizer so this can exceed the max generated len")
flags.DEFINE_integer("disc_ans_limit", 30, "Limit length for answers")
flags.DEFINE_integer("disc_test_para_limit", 1000, "Limit length for paragraph in test file")
flags.DEFINE_integer("disc_test_ques_limit", 150, "Limit length for question in test file")
flags.DEFINE_integer("disc_char_limit", 16, "Limit length for character")
flags.DEFINE_integer("disc_word_count_limit", -1, "Min count for word")
flags.DEFINE_integer("disc_char_count_limit", -1, "Min count for char")

flags.DEFINE_integer("disc_capacity", 15000, "Batch size of dataset shuffle")
flags.DEFINE_integer("disc_num_threads", 4, "Number of threads in input pipeline")
flags.DEFINE_boolean("disc_is_bucket", False, "build bucket batch iterator or not")
# flags.DEFINE_integer("disc_bucket_range", [40, 401, 40], "the range of bucket")

# flags.DEFINE_integer("disc_batch_size", 32, "Batch size")
# flags.DEFINE_integer("disc_num_steps", 60000, "Number of steps")
flags.DEFINE_integer("disc_num_epochs", 20, "Number of steps")
flags.DEFINE_integer("disc_checkpoint", 1000, "checkpoint to save and evaluate the model")
flags.DEFINE_integer("disc_period", 100, "period to save batch loss")
flags.DEFINE_integer("disc_val_num_batches", 150, "Number of batches to evaluate the model")
flags.DEFINE_float("disc_dropout", 0.3, "Dropout prob across the layers")
flags.DEFINE_float("disc_grad_clip", 5.0, "Global Norm gradient clipping rate")
flags.DEFINE_float("disc_learning_rate", 0.0001, "Learning rate")
flags.DEFINE_float("disc_decay", 0.9999, "Exponential moving average decay")
flags.DEFINE_float("disc_l2_norm", 3e-8, "L2 norm scale")
flags.DEFINE_integer("disc_hidden", 96, "Hidden size")
flags.DEFINE_integer("disc_num_heads", 1, "Number of heads in self attention")
flags.DEFINE_integer("disc_early_stop", 10, "Checkpoints for early stop")

# Extensions (Uncomment corresponding code in download.sh to download the required data)
glove_char_file = os.path.join(home, "data", "glove", "glove.840B.300d-char.txt")
flags.DEFINE_string("disc_glove_char_file", glove_char_file, "Glove character embedding source file")
flags.DEFINE_boolean("disc_pretrained_char", False, "Whether to use pretrained character embedding")

fasttext_file = os.path.join(home, "data", "fasttext", "wiki-news-300d-1M.vec")
flags.DEFINE_string("disc_fasttext_file", fasttext_file, "Fasttext word embedding source file")
flags.DEFINE_boolean("disc_fasttext", False, "Whether to use fasttext")
