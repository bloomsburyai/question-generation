
class MaluubaModel(Seq2SeqModel):
    def __init__(self, vocab, batch_size, training_mode=False):
        super().__init__(vocab, batch_size, training_mode)

    def build_model(self):
        print('Modifying Seq2Seq model to incorporate RL rewards')
