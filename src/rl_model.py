import tensorflow as tf
from helpers.ops import safe_log, total_params
from seq2seq_model import Seq2SeqModel

from langmodel.lm import LstmLmInstance
# from qa.mpcm import MpcmQaInstance
from qa.qanet.instance import QANetInstance

# from helpers.misc_utils import debug_shape


FLAGS = tf.app.flags.FLAGS



class RLModel(Seq2SeqModel):
    def __init__(self, vocab, training_mode=False, use_embedding_loss=False):


        super().__init__(vocab, training_mode=training_mode, use_embedding_loss=use_embedding_loss)
        self.modify_seq2seq_model()

    def modify_seq2seq_model(self):
        print('Modifying Seq2Seq model to incorporate RL rewards')

        if FLAGS.policy_gradient:
            print('Building and loading LM')
            self.lm = LstmLmInstance()
            self.lm.load_from_chkpt(FLAGS.model_dir+'saved/lmtest')

            print('Building and loading QA model')
            # self.qa = MpcmQaInstance()
            # self.qa.load_from_chkpt(FLAGS.model_dir+'saved/qatest')
            self.qa = QANetInstance()
            self.qa.load_from_chkpt(FLAGS.model_dir+'saved/qanet2')

        with self.graph.as_default():

            self.lm_score = tf.placeholder(tf.float32, [None], "lm_score")
            self.qa_score = tf.placeholder(tf.float32, [None],"qa_score")
            self.disc_score = tf.placeholder(tf.float32, [None],"disc_score")
            self.bleu_score = tf.placeholder(tf.float32, [None],"bleu_score")
            self.rl_lm_enabled = tf.placeholder_with_default(False,(), "rl_lm_enabled")
            self.rl_qa_enabled = tf.placeholder_with_default(False,(), "rl_qa_enabled")
            self.rl_disc_enabled = tf.placeholder_with_default(False,(), "rl_disc_enabled")
            self.rl_bleu_enabled = tf.placeholder_with_default(False,(), "rl_bleu_enabled")
            self.step = tf.placeholder(tf.int32, (), "step")

            with tf.variable_scope('rl_rewards'):
                # NOTE: This isnt obvious! If we feed in the generated Qs as the gold with a reward,
                # we get REINFORCE. If we feed in a reward of 1.0 with an actual gold Q, we get cross entropy.
                # So we can combine both in the same set of ops, but need to construct batches appropriately
                mask = tf.one_hot(self.question_ids, depth=len(self.vocab) +FLAGS.max_copy_size)

                self.lm_loss = -1.0*self.lm_score * tf.reduce_sum(tf.reduce_sum(safe_log(self.q_hat) * mask, axis=[2])* self.target_weights,axis=1)/tf.cast(self.question_length, tf.float32)
                self.qa_loss = -1.0*self.qa_score * tf.reduce_sum(tf.reduce_sum(safe_log(self.q_hat) * mask, axis=[2])* self.target_weights,axis=1)/tf.cast(self.question_length, tf.float32)
                self.disc_loss = -1.0*self.disc_score * tf.reduce_sum(tf.reduce_sum(safe_log(self.q_hat) * mask, axis=[2])* self.target_weights,axis=1)/tf.cast(self.question_length, tf.float32)
                self.bleu_loss = -1.0*self.bleu_score * tf.reduce_sum(tf.reduce_sum(safe_log(self.q_hat) * mask, axis=[2])* self.target_weights,axis=1)/tf.cast(self.question_length, tf.float32)

            pg_loss = tf.cond(self.rl_lm_enabled, lambda: self.lm_loss, lambda: tf.constant([0.0])) + \
                tf.cond(self.rl_qa_enabled, lambda: self.qa_loss, lambda: tf.constant([0.0])) + \
                tf.cond(self.rl_disc_enabled, lambda: self.disc_loss, lambda: tf.constant([0.0])) + \
                tf.cond(self.rl_bleu_enabled, lambda: self.bleu_loss, lambda: tf.constant([0.0]))

            curr_batch_size_pg = tf.shape(self.answer_ids)[0]//2

            # log the first half of the batch - this is the RL part
            self._train_summaries.append(tf.summary.scalar("train_loss/pg_loss_rl", tf.reduce_mean(pg_loss[:curr_batch_size_pg])))
            self._train_summaries.append(tf.summary.scalar("train_loss/pg_loss_ml", tf.reduce_mean(pg_loss[curr_batch_size_pg:])))

            self.pg_loss=tf.reduce_mean(pg_loss,axis=[0])

            self._train_summaries.append(tf.summary.scalar("train_loss/pg_loss", self.pg_loss))

            # this needs rebuilding again
            self.train_summary = tf.summary.merge(self._train_summaries)

            # dont bother calculating gradients if not training
            if self.training_mode:
                # these need to be redefined with the correct inputs
                # Calculate and clip gradients
                params = tf.trainable_variables()
                gradients = tf.gradients(self.pg_loss, params)
                clipped_gradients, _ = tf.clip_by_global_norm(
                    gradients, 5)

                # Optimization
                lr = FLAGS.learning_rate if not FLAGS.lr_schedule else tf.minimum(1.0, tf.cast(self.step, tf.float32)*0.001)*FLAGS.learning_rate
                self.pg_optimizer = tf.train.AdamOptimizer(lr).apply_gradients(
                    zip(clipped_gradients, params)) if self.training_mode else tf.no_op()

            total_params()
