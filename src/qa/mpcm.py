import sys,json
sys.path.insert(0, "/Users/tom/Dropbox/msc-ml/project/src/")


import tensorflow as tf
import numpy as np

from base_model import TFModel

import helpers.loader as loader
from helpers.preprocessing import tokenise

from helpers.misc_utils import debug_shape


import flags
FLAGS = tf.app.flags.FLAGS

mem_limit=0.95


# This should handle the mechanics of the model - basically it's a wrapper around the TF graph
class MpcmQa(TFModel):
    def __init__(self, vocab, training_mode=True):
        self.embedding_size = tf.app.flags.FLAGS.embedding_size
        self.vocab = vocab
        self.training_mode = training_mode
        super().__init__()


    def build_model(self):

        self.dropout_prob=0.2

        self.context_in = tf.placeholder(tf.int32, [None, None])
        self.question_in = tf.placeholder(tf.int32, [None, None])
        self.context_len = tf.reduce_sum(tf.cast(tf.not_equal(self.context_in, self.vocab[loader.PAD]), tf.int32), axis=1)
        self.question_len = tf.reduce_sum(tf.cast(tf.not_equal(self.question_in, self.vocab[loader.PAD]), tf.int32), axis=1)

        self.answer_spans_in = tf.placeholder(tf.int32, [None, 2])

        with tf.device('/cpu:*'):
            # Load glove embeddings
            glove_embeddings = loader.load_glove(FLAGS.data_path, d=FLAGS.embedding_size)
            embeddings_init = tf.constant(loader.get_embeddings(self.vocab, glove_embeddings, D=FLAGS.embedding_size))
            self.embeddings = tf.get_variable('word_embeddings', initializer=embeddings_init, dtype=tf.float32, trainable=False)
            assert self.embeddings.shape == [len(self.vocab), self.embedding_size]

        # Layer 1: representation layer
        self.context_embedded = tf.layers.dropout(tf.nn.embedding_lookup(self.embeddings, self.context_in), rate=self.dropout_prob, training=self.is_training)
        self.question_embedded = tf.layers.dropout(tf.nn.embedding_lookup(self.embeddings, self.question_in), rate=self.dropout_prob, training=self.is_training)

        # Layer 2: Filter. r is batch x con_len x q_len
        self.r_norm = (tf.expand_dims(tf.norm(self.context_embedded, ord=2, axis=2),-1) * tf.expand_dims(tf.norm(self.question_embedded, ord=2, axis=2),-2))
        self.r = tf.matmul(self.context_embedded, tf.transpose(self.question_embedded,[0,2,1]))/self.r_norm
        self.r_context = tf.reduce_max(self.r, axis=2, keep_dims=True)
        # r_question = tf.reduce_max(r, axis=1, keep_dims=True)

        self.context_filtered = self.r_context * self.context_embedded
        self.question_filtered = self.question_embedded#tf.layers.dropout(tf.tile(tf.transpose(r_question,[0,2,1]), [1,1,self.embedding_size]) * self.question_embedded, rate=0.2, training=self.is_training)

        # print(self.context_filtered)
        # print(self.question_filtered)

        # Layer 3: Context representation (BiLSTM encoder)
        num_units_encoder=FLAGS.qa_encoder_units
        with tf.variable_scope('layer3_fwd_cell'):
            cell_fw = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(num_units=num_units_encoder),
                input_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.)),
                state_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.)),
                output_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.)),
                input_size=self.embedding_size,
                variational_recurrent=True,
                dtype=tf.float32)
        with tf.variable_scope('layer3_bwd_cell'):
            cell_bw = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(num_units=num_units_encoder),
                input_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.)),
                state_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.)),
                output_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.)),
                input_size=self.embedding_size,
                variational_recurrent=True,
                dtype=tf.float32)
        with tf.variable_scope('context_rnn'):
            self.context_encodings,_ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.context_filtered, dtype=tf.float32)
        with tf.variable_scope('q_rnn'):
            self.question_encodings,_ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.question_filtered, dtype=tf.float32)

        # print(self.context_encodings)
        # print(self.question_encodings)

        # Layer 4: context matching layer
        eps = 1e-6
        def similarity(v1, v2, W): #v1,v2 are batch x seq x d, W is lxd
            #"btd,ld->btld"

            # W_tiled = tf.tile(tf.expand_dims(W,axis=-1), [1,1,tf.shape(W)[1]])
            # v1_weighted =tf.tensordot(v1, W_tiled, [[-1],[-1]])
            # v2_weighted =tf.tensordot(v2, W_tiled, [[-1],[-1]])

            v1_weighted = tf.expand_dims(v1,2) * tf.expand_dims(tf.expand_dims(W, axis=0),axis=0)
            v2_weighted = tf.expand_dims(v2,2) * tf.expand_dims(tf.expand_dims(W, axis=0),axis=0)

            # v1_weighted = tf.einsum("btd,ld->btld", v1, W)
            # v2_weighted = tf.einsum("btd,ld->btld", v2, W)


            # similarity = tf.einsum("bild,bjld->bijl", v1_weighted, v2_weighted)
            similarity = tf.matmul(tf.transpose(v1_weighted,[0,2,1,3]), tf.transpose(v2_weighted, [0,2,3,1]))
            similarity = tf.transpose(similarity, [0,2,3,1])

            v1_norm = tf.expand_dims(tf.norm(v1_weighted, ord=2,axis=-1),axis=-2)
            v2_norm = tf.expand_dims(tf.norm(v2_weighted, ord=2,axis=-1),axis=-3)

            # print(similarity)
            return similarity/v1_norm/v2_norm

        m_fwd = similarity(self.context_encodings[0], self.question_encodings[0], tf.get_variable("W1", (50, num_units_encoder), tf.float32))
        m_bwd = similarity(self.context_encodings[1], self.question_encodings[1], tf.get_variable("W2", (50, num_units_encoder), tf.float32))
        m_fwd2 = similarity(self.context_encodings[0], self.question_encodings[0], tf.get_variable("W3", (50, num_units_encoder), tf.float32))
        m_bwd2 = similarity(self.context_encodings[1], self.question_encodings[1], tf.get_variable("W4", (50, num_units_encoder), tf.float32))
        m_fwd3 = similarity(self.context_encodings[0], self.question_encodings[0], tf.get_variable("W5", (50, num_units_encoder), tf.float32))
        m_bwd3 = similarity(self.context_encodings[1], self.question_encodings[1], tf.get_variable("W6", (50, num_units_encoder), tf.float32))

        def get_last_seq(seq, lengths): # seq is batch x dim1 x time  x dim2
            seq = tf.transpose(seq, [0,2,1,3]) # batch x time x dim1 x dim2
            lengths = tf.maximum(lengths, tf.zeros_like(lengths, dtype=tf.int32))

            batch_size = tf.shape(lengths)[0]
            batch_nums = tf.range(0, limit=batch_size) # shape (batch_size)
            indices = tf.stack((batch_nums, lengths), axis=1) # shape (batch_size, 2)
            result = tf.gather_nd(seq, indices)
            return result # [batch_size, dim1, dim 2]

        # -1 should actually be the question length
        mask = tf.expand_dims(tf.expand_dims(tf.sequence_mask(lengths=self.question_len, maxlen=tf.reduce_max(self.question_len), dtype=tf.float32), 1),-1)
        m_full_fwd = get_last_seq(m_fwd, self.question_len-1)
        m_full_bwd = m_bwd[:,:,0,:]
        m_max_fwd  = tf.reduce_max(m_fwd2*mask, axis=2)
        m_max_bwd  = tf.reduce_max(m_bwd2*mask, axis=2)
        m_mean_fwd  = tf.reduce_sum(m_fwd3*mask, axis=2)/tf.expand_dims(tf.expand_dims(tf.cast(self.question_len, tf.float32),-1),-1)
        m_mean_bwd  = tf.reduce_sum(m_bwd3*mask, axis=2)/tf.expand_dims(tf.expand_dims(tf.cast(self.question_len, tf.float32),-1),-1)
        self.matches = tf.concat([m_full_fwd, m_full_bwd, m_max_fwd, m_max_bwd, m_mean_fwd, m_mean_bwd], axis=2)

        # print(m_full_bwd)
        # print(self.matches)

        # Layer 5: aggregate with BiLSTM
        with tf.variable_scope('layer5_fwd_cell'):
            cell_fw2 = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(num_units=FLAGS.qa_match_units),
                input_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.)),
                state_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.)),
                output_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.)),
                input_size=50*6,
                variational_recurrent=True,
                dtype=tf.float32)
        with tf.variable_scope('layer5_bwd_cell'):
            cell_bw2 = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(num_units=FLAGS.qa_match_units),
                input_keep_prob=1.0,
                state_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.)),
                output_keep_prob=(tf.cond(self.is_training,lambda: 1.0 - self.dropout_prob,lambda: 1.)),
                input_size=50*6,
                variational_recurrent=True,
                dtype=tf.float32)
        with tf.variable_scope('match_rnn'):
            self.aggregated_matches,_ = tf.nn.bidirectional_dynamic_rnn(cell_fw2, cell_bw2, self.matches, dtype=tf.float32)
        self.aggregated_matches = tf.concat(self.aggregated_matches, axis=2)

        # Layer 6: Fully connected to get logits
        self.logits_start = tf.squeeze(tf.layers.dense(self.aggregated_matches, 1, activation=None),-1)
        self.logits_end = tf.squeeze(tf.layers.dense(self.aggregated_matches, 1, activation=None),-1)

        self.prob_start = tf.nn.softmax(self.logits_start)
        self.prob_end = tf.nn.softmax(self.logits_end)

        # training loss
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.answer_spans_in[:,0], logits=self.logits_start)*0.5+tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.answer_spans_in[:,1], logits=self.logits_end)*0.5)
        self.nll = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.answer_spans_in[:,0], logits=self.logits_start)*0.5+tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.answer_spans_in[:,1], logits=self.logits_end)*0.5

        if self.training_mode:
            # Calculate and clip gradients
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(
                gradients, 5)

            # Optimization
            self.optimizer = tf.train.AdamOptimizer(FLAGS.qa_learning_rate).apply_gradients(
                zip(clipped_gradients, params))

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.stack([tf.argmax(self.prob_start, axis=-1, output_type=tf.int32),tf.argmax(self.prob_end, axis=-1, output_type=tf.int32)],axis=1), self.answer_spans_in), tf.float32))

        # predictions: coerce start<end
        self.probs_coerced = tf.matrix_band_part(tf.matmul(tf.expand_dims(self.prob_start, 2), tf.expand_dims(self.prob_end,1)), 0, -1)

        self.pred_ix = tf.argmax(tf.reshape(self.probs_coerced, [-1, tf.shape(self.context_in)[1]*tf.shape(self.context_in)[1]]),axis=1)
        self.pred_start = tf.cast(tf.floor(tf.cast(self.pred_ix,tf.float32)/tf.cast(tf.shape(self.context_in)[1],tf.float32)), tf.int32)
        self.pred_end = tf.cast(tf.mod(tf.cast(self.pred_ix,tf.int32), tf.shape(self.context_in)[1]), tf.int32)
        self.pred_span = tf.concat([tf.expand_dims(self.pred_start,1), tf.expand_dims(self.pred_end,1)], axis=1)

class MpcmQaInstance():
    def __del__(self):
        self.sess.close()

    def get_padded_batch(self, seq_batch):
        seq_batch_ids = [[self.vocab[loader.SOS]]+[self.vocab[tok if tok in self.vocab.keys() else loader.OOV] for tok in tokenise(sent, asbytes=False)]+[self.vocab[loader.EOS]] for sent in seq_batch]
        max_seq_len = max([len(seq) for seq in seq_batch_ids])
        padded_batch = np.asarray([seq + [self.vocab[loader.PAD] for i in range(max_seq_len-len(seq))] for seq in seq_batch_ids])
        return padded_batch


    def load_from_chkpt(self, path):
        with open(path+'/vocab.json') as f:
            self.vocab = json.load(f)

        self.model = MpcmQa(self.vocab, training_mode=False)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_limit,allow_growth = True,visible_device_list='0')
        self.sess = tf.Session(graph=self.model.graph, config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True))

        with self.model.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, path+ '/model.checkpoint')

    def get_ans(self, contexts, questions):
        toks=[tokenise(ctxt, asbytes=False) for ctxt in contexts]
        padded_batch_cs = self.get_padded_batch(contexts)
        padded_batch_qs = self.get_padded_batch(questions)
        spans = self.sess.run(self.model.pred_span, feed_dict={self.model.context_in: padded_batch_cs, self.model.question_in: padded_batch_qs})
        return [" ".join(toks[i][span[0]:span[1]+1]) for i,span in enumerate(spans)]


def main(_):
    import helpers.metrics as metrics
    from tqdm import tqdm

    # train_data = loader.load_squad_triples("./data/", False)
    dev_data = loader.load_squad_triples("./data/", test=True, ans_list=True)



    # print('Loaded SQuAD with ',len(train_data),' triples')
    # train_contexts, train_qs, train_as,train_a_pos = zip(*train_data)

    qa = MpcmQaInstance()
    qa.load_from_chkpt(FLAGS.model_dir+'saved/qamaybe')
    vocab = qa.vocab

    questions = ["What colour is the car?","When was the car made?","Where was the date?", "What was the dog called?","Who was the oldest cat?"]
    contexts=["The car is green, and was built in 1985. This sentence should make it less likely to return the date, when asked about a cat. The oldest cat was called creme puff and lived for many years!" for i in range(len(questions))]



    # print(contexts[0])


    f1s=[]
    ems=[]
    for x in tqdm(dev_data):
        ans_pred = qa.get_ans([x[0]], [x[1]])[0]


        this_f1s=[]
        this_ems=[]
        for a in range(len(x[2])):
            this_ems.append(1.0*(metrics.normalize_answer(ans_pred) == metrics.normalize_answer(x[2][a])))
            this_f1s.append(metrics.f1(metrics.normalize_answer(ans_pred), metrics.normalize_answer(x[2][a])))
        ems.append(max(this_ems))
        f1s.append(max(this_f1s))
    print("EM: ",np.mean(ems), " F1: ", np.mean(f1s))
if __name__ == "__main__":
    tf.app.run()
