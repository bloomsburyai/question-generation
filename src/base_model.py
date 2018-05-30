import tensorflow as tf

class TFModel():
    def __init__(self):
        self.is_training = tf.placeholder_with_default(False,(), "is_training")
        self.dropout_prob = tf.app.flags.FLAGS.dropout_rate # TODO: add this to FLAGS

        self._train_summaries=[]
        self._eval_summaries=[]
        self._output_summaries=[]

        self.build_model()

        self._train_summaries.extend(
            [tf.summary.scalar("train_loss/loss", self.loss),
             tf.summary.scalar("train_loss/accuracy", self.accuracy)])
        self._eval_summaries.extend(
            [tf.summary.scalar("eval_loss/loss", self.loss),
             tf.summary.scalar("eval_loss/accuracy", self.accuracy)])
        self.train_summary = tf.summary.merge(self._train_summaries)
        self.output_summary = tf.summary.merge(self._output_summaries)
        self.eval_summary = tf.summary.merge(self._eval_summaries)

    # Implement this method!
    def build_model(self):
        pass




    def placeholders(self):
        return self.x, self.y, self.is_training


    def predict(self):
        pass
