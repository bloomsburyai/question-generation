import tensorflow as tf
from helpers.ops import safe_log
from seq2seq_model import Seq2SeqModel

from langmodel.lm import LstmLmInstance
from qa.mpcm import MpcmQaInstance

from helpers.misc_utils import debug_shape


FLAGS = tf.app.flags.FLAGS



class MaluubaModel(Seq2SeqModel):
    def __init__(self, vocab, lm_vocab, qa_vocab, batch_size, training_mode=False):
        super().__init__(vocab, batch_size, advanced_condition_encoding=True, training_mode=training_mode)
        self.modify_seq2seq_model(lm_vocab, qa_vocab)

    def modify_seq2seq_model(self, lm_vocab, qa_vocab):
        print('Modifying Seq2Seq model to incorporate RL rewards')

        self.lm = LstmLmInstance(lm_vocab)
        self.lm.load_from_chkpt(FLAGS.model_dir+'saved/lmtest')

        # self.qa = MpcmQaInstance(qa_vocab)
        # self.qa.load_from_chkpt(FLAGS.model_dir+'saved/qatest')

        with self.graph.as_default():

            self.lm_score = tf.placeholder(tf.float32, [None], "lm_score")
            self.qa_score = tf.placeholder(tf.float32, [None],"qa_score")
            self.rl_lm_enabled = tf.placeholder_with_default(False,(), "rl_lm_enabled")
            self.rl_qa_enabled = tf.placeholder_with_default(False,(), "rl_qa_enabled")

            with tf.variable_scope('rl_rewards'):
                # TODO: Fluency reward from LM
                # TODO: Answerability reward from QA model
                # TODO: Correct REINFORCE loss
                # TODO: Check teacher forcing method for learning using beam search
                # TODO: whiten rewards (popart)
                curr_batch_size = tf.shape(self.lm_score)[0]
                lm_score_whitened = (self.lm_score-tf.tile(tf.reduce_mean(self.lm_score, keep_dims=True),[curr_batch_size]))/tf.tile(tf.nn.moments(self.lm_score,axes=0,keep_dims=True)[1]+1e-6,[curr_batch_size])
                qa_score_whitened = (self.qa_score-tf.tile(tf.reduce_mean(self.qa_score, keep_dims=True),[curr_batch_size]))/tf.tile(tf.nn.moments(self.qa_score,axes=0,keep_dims=True)[1]+1e-6,[curr_batch_size])

                mask = tf.one_hot(self.q_hat_ids, depth=len(self.vocab) +FLAGS.max_copy_size)

                lm_loss = tf.reduce_mean(-1.0*lm_score_whitened * tf.reduce_sum(safe_log(self.q_hat * mask), axis=[1,2]),axis=0)
                qa_loss = tf.reduce_mean(-1.0*qa_score_whitened * tf.reduce_sum(safe_log(self.q_hat * mask), axis=[1,2]),axis=0)

            self._train_summaries.append(tf.summary.scalar("rl_rewards/lm", tf.reduce_mean(self.lm_score)))
            self._train_summaries.append(tf.summary.scalar("rl_rewards/qa", tf.reduce_mean(self.qa_score)))

            self._train_summaries.append(tf.summary.scalar("train_loss/lm", lm_loss))
            self._train_summaries.append(tf.summary.scalar("train_loss/qa", qa_loss))

            self.loss = self.loss + \
                tf.cond(self.rl_lm_enabled, lambda: lm_loss*0.1, lambda: tf.constant(0.0)) + \
                tf.cond(self.rl_qa_enabled, lambda: qa_loss*1.0, lambda: tf.constant(0.0))

            self._train_summaries.append(tf.summary.scalar("train_loss/loss_incrl", self.loss))

            # this needs rebuilding again
            self.train_summary = tf.summary.merge(self._train_summaries)


            # these need to be redefined with the correct inputs
            # Calculate and clip gradients
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(
                gradients, 2)

            # Optimization
            self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).apply_gradients(
                zip(clipped_gradients, params)) if self.training_mode else tf.no_op()
