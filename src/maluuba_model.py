
class MaluubaModel(Seq2SeqModel):
    def __init__(self, vocab, batch_size, training_mode=False):
        super().__init__(vocab, batch_size, advanced_condition_encoding=True, training_mode)

    def build_model(self):
        print('Modifying Seq2Seq model to incorporate RL rewards')
        rewards = []
        rl_reward_loss = tf.constant(0.)
        with tf.variable_scope('rl_rewards'):
            # TODO: Fluency reward from LM
            # TODO: Answerability reward from QA model
            # TODO: Correct REINFORCE loss
            # TODO: Check teacher forcing method for learning using beam search
            # TODO: whiten rewards (popart)

            for reward in rewards:
                rl_reward_loss += 0.5* tf.square(reward)
