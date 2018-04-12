import tensorflow as tf

EPSILON = tf.cast(tf.keras.backend.epsilon(), tf.float32)

def safe_log(x):
    return tf.log(tf.clip_by_value(x, EPSILON, 1-EPSILON))
