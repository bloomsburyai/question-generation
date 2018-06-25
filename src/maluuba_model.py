import tensorflow as tf
from helpers.ops import safe_log
from seq2seq_model import Seq2SeqModel

from langmodel.lm import LstmLmInstance
from qa.mpcm import MpcmQaInstance

from helpers.misc_utils import debug_shape


FLAGS = tf.app.flags.FLAGS



class MaluubaModel(Seq2SeqModel):
    def __init__(self, vocab, training_mode=False, lm_weight=0, qa_weight=0):
        self.lm_weight = lm_weight
        self.qa_weight = qa_weight

        super().__init__(vocab, advanced_condition_encoding=True, training_mode=training_mode)
        self.modify_seq2seq_model()

    def modify_seq2seq_model(self):
        print('Modifying Seq2Seq model to incorporate RL rewards')

        print('Building and loading LM')
        self.lm = LstmLmInstance()
        self.lm.load_from_chkpt(FLAGS.model_dir+'saved/lmtest')

        print('Building and loading QA model')
        self.qa = MpcmQaInstance()
        self.qa.load_from_chkpt(FLAGS.model_dir+'saved/qatest')

        with self.graph.as_default():

            self.lm_score = tf.placeholder(tf.float32, [None], "lm_score")
            self.qa_score = tf.placeholder(tf.float32, [None],"qa_score")
            self.rl_lm_enabled = tf.placeholder_with_default(False,(), "rl_lm_enabled")
            self.rl_qa_enabled = tf.placeholder_with_default(False,(), "rl_qa_enabled")

            with tf.variable_scope('rl_rewards'):
                # NOTE: This isnt obvious! If we feed in the generated Qs as the gold with a reward,
                # we get REINFORCE. If we feed in a reward of 1.0 with an actual gold Q, we get cross entropy.
                # So we can combine both in the same set of ops, but need to construct batches appropriately
                mask = tf.one_hot(self.question_ids, depth=len(self.vocab) +FLAGS.max_copy_size)

                lm_loss = tf.reduce_mean(-1.0*self.lm_score * tf.reduce_sum(tf.reduce_sum(safe_log(self.q_hat) * mask, axis=[2])* self.target_weights,axis=1)/tf.cast(self.question_length, tf.float32),axis=[0])
                qa_loss = tf.reduce_mean(-1.0*self.qa_score * tf.reduce_sum(tf.reduce_sum(safe_log(self.q_hat) * mask, axis=[2])* self.target_weights,axis=1)/tf.cast(self.question_length, tf.float32),axis=[0])

            self._train_summaries.append(tf.summary.scalar("rl_rewards/lm", tf.reduce_mean(self.lm_score)))
            self._train_summaries.append(tf.summary.scalar("rl_rewards/qa", tf.reduce_mean(self.qa_score)))

            self._train_summaries.append(tf.summary.scalar("train_loss/lm", lm_loss))
            self._train_summaries.append(tf.summary.scalar("train_loss/qa", qa_loss))

            self.pg_loss = tf.cond(self.rl_lm_enabled, lambda: lm_loss, lambda: tf.constant(0.0)) + \
                tf.cond(self.rl_qa_enabled, lambda: qa_loss, lambda: tf.constant(0.0))

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
                self.pg_optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).apply_gradients(
                    zip(clipped_gradients, params)) if self.training_mode else tf.no_op()
