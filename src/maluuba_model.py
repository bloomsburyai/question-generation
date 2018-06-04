import tensorflow as tf
from helpers.ops import safe_log
from seq2seq_model import Seq2SeqModel


FLAGS = tf.app.flags.FLAGS



class MaluubaModel(Seq2SeqModel):
    def __init__(self, vocab, batch_size, training_mode=False):
        super().__init__(vocab, batch_size, advanced_condition_encoding=True, training_mode=training_mode)
        self.modify_seq2seq_model()

    def modify_seq2seq_model(self):
        print('Modifying Seq2Seq model to incorporate RL rewards')

        rewards = []
        rl_reward_loss = tf.constant(0.)
        with tf.variable_scope('rl_rewards'):
            # TODO: Fluency reward from LM
            # TODO: Answerability reward from QA model
            # TODO: Correct REINFORCE loss
            # TODO: Check teacher forcing method for learning using beam search
            # TODO: whiten rewards (popart)

            mask = tf.one_hot(self.q_hat_ids, depth=len(self.vocab) +FLAGS.max_copy_size)
            for reward in rewards:
                # REINFORCE "loss"
                rl_reward_loss += reward * tf.reduce_sum(safe_log(self.q_hat * mask), axis=[1,2])
