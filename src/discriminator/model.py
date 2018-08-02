import tensorflow as tf
from discriminator.layers import initializer, regularizer, residual_block, highway, conv, mask_logits, trilinear, total_params, optimized_trilinear_for_attention

class Model(object):
    def __init__(self, config, batch, word_mat=None, char_mat=None, trainable=True, opt=True, demo = False, graph = None):
        self.config = config
        self.demo = demo
        self.graph = graph if graph is not None else tf.Graph()
        self._train_summaries=[]
        with self.graph.as_default():

            self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                               initializer=tf.constant_initializer(0), trainable=False)
            self.dropout = tf.placeholder_with_default(0.0, (), name="dropout")
            if self.demo:
                self.c = tf.placeholder(tf.int32, [None, config.disc_para_limit],"context")
                self.q = tf.placeholder(tf.int32, [None, config.disc_ques_limit],"question")
                self.ch = tf.placeholder(tf.int32, [None, config.disc_para_limit, config.disc_char_limit],"context_char")
                self.qh = tf.placeholder(tf.int32, [None, config.disc_ques_limit, config.disc_char_limit],"question_char")
                self.ans_start = tf.placeholder(tf.int32, [None],"answer_index1")
                self.ans_end = tf.placeholder(tf.int32, [None],"answer_index2")
                self.gold_class = tf.placeholder(tf.int32, [None],"gold_class")
            else:
                self.c, self.q, self.ch, self.qh, self.ans_start, self.ans_end, self.qa_id = batch.get_next()

            # self.word_unk = tf.get_variable("word_unk", shape = [config.disc_glove_dim], initializer=initializer())
            with tf.device('/cpu:*'):
                self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
                    word_mat, dtype=tf.float32), trainable=False)
                self.char_mat = tf.get_variable(
                    "char_mat", initializer=tf.constant(char_mat, dtype=tf.float32))

            self.c_mask = tf.cast(self.c, tf.bool)
            self.q_mask = tf.cast(self.q, tf.bool)
            self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
            self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

            if opt:
                # we have to hardcode the max batch size here! use the batch size from the generator as this will be used for PG
                N, CL = config.batch_size if not self.demo else config.batch_size, config.disc_char_limit
                self.c_maxlen = tf.reduce_max(self.c_len)
                self.q_maxlen = tf.reduce_max(self.q_len)
                self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen])
                self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen])
                self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
                self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
                self.ch = tf.slice(self.ch, [0, 0, 0], [N, self.c_maxlen, CL])
                self.qh = tf.slice(self.qh, [0, 0, 0], [N, self.q_maxlen, CL])
                self.ans_start = tf.slice(self.ans_start, [0], [N])
                self.ans_end = tf.slice(self.ans_end, [0], [N])
                self.gold_class= tf.slice(self.gold_class, [0], [N]) # batch x 1
            else:
                self.c_maxlen, self.q_maxlen = config.disc_para_limit, config.disc_ques_limit

            self.ch_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])
            self.qh_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])

            self.forward()
            total_params()

            if trainable:
                self.lr = tf.minimum(config.disc_learning_rate, 0.001 / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1))
                self.opt = tf.train.AdamOptimizer(learning_rate = self.lr, beta1 = 0.8, beta2 = 0.999, epsilon = 1e-7)
                grads = self.opt.compute_gradients(self.loss)
                gradients, variables = zip(*grads)
                capped_grads, _ = tf.clip_by_global_norm(
                    gradients, config.disc_grad_clip)
                self.train_op = self.opt.apply_gradients(
                    zip(capped_grads, variables), global_step=self.global_step)

            self.train_summary = tf.summary.merge(self._train_summaries)

    def forward(self):
        config = self.config
        N, PL, QL, CL, d, dc, nh = config.batch_size if not self.demo else tf.shape(self.c)[0], self.c_maxlen, self.q_maxlen, config.disc_char_limit, config.disc_hidden, config.disc_char_dim, config.disc_num_heads

        with tf.variable_scope("Input_Embedding_Layer"):
            ch_emb = tf.reshape(tf.nn.embedding_lookup(
                self.char_mat, self.ch), [N * PL, CL, dc])
            qh_emb = tf.reshape(tf.nn.embedding_lookup(
                self.char_mat, self.qh), [N * QL, CL, dc])


            ch_emb = tf.nn.dropout(ch_emb, 1.0 - 0.5 * self.dropout)
            qh_emb = tf.nn.dropout(qh_emb, 1.0 - 0.5 * self.dropout)

			# Bidaf style conv-highway encoder
            ch_emb = conv(ch_emb, d,
                bias = True, activation = tf.nn.relu, kernel_size = 5, name = "char_conv", reuse = None)
            qh_emb = conv(qh_emb, d,
                bias = True, activation = tf.nn.relu, kernel_size = 5, name = "char_conv", reuse = True)

            ch_emb = tf.reduce_max(ch_emb, axis = 1)
            qh_emb = tf.reduce_max(qh_emb, axis = 1)

            ch_emb = tf.reshape(ch_emb, [N, PL, ch_emb.shape[-1]])
            qh_emb = tf.reshape(qh_emb, [N, QL, ch_emb.shape[-1]])

            c_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.c), 1.0 - self.dropout)
            q_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.q), 1.0 - self.dropout)

            # Is context token in answer?

            context_ix = tf.tile(tf.expand_dims(tf.range(PL, dtype=tf.int32),axis=0), [N,1])
            gt_start = tf.greater_equal(context_ix, tf.tile(tf.expand_dims(self.ans_start,axis=-1), [1, PL]))
            lt_end = tf.less_equal(context_ix, tf.tile(tf.expand_dims(self.ans_end,axis=-1), [1, PL]))
            self.in_answer_feature = tf.expand_dims(tf.cast(tf.logical_and(gt_start, lt_end), tf.float32),axis=2)



            # c_emb = tf.concat([c_emb, self.in_answer_feature, ch_emb], axis=2)
            # q_emb = tf.concat([q_emb, tf.zeros([N, QL, 1]), qh_emb], axis=2)

            c_emb = tf.concat([c_emb, ch_emb], axis=2)
            q_emb = tf.concat([q_emb, qh_emb], axis=2)

            c_emb = highway(c_emb, size = d, scope = "highway", dropout = self.dropout, reuse = None)
            q_emb = highway(q_emb, size = d, scope = "highway", dropout = self.dropout, reuse = True)

        with tf.variable_scope("Embedding_Encoder_Layer"):
            c = residual_block(c_emb,
                num_blocks = 1,
                num_conv_layers = 4,
                kernel_size = 7,
                mask = self.c_mask,
                num_filters = d,
                num_heads = nh,
                seq_len = self.c_len,
                scope = "Encoder_Residual_Block",
                bias = False,
                dropout = self.dropout)
            q = residual_block(q_emb,
                num_blocks = 1,
                num_conv_layers = 4,
                kernel_size = 7,
                mask = self.q_mask,
                num_filters = d,
                num_heads = nh,
                seq_len = self.q_len,
                scope = "Encoder_Residual_Block",
                reuse = True, # Share the weights between passage and question
                bias = False,
                dropout = self.dropout)

        with tf.variable_scope("Context_to_Query_Attention_Layer"):
            # C = tf.tile(tf.expand_dims(c,2),[1,1,self.q_maxlen,1])
            # Q = tf.tile(tf.expand_dims(q,1),[1,self.c_maxlen,1,1])
            # S = trilinear([C, Q, C*Q], input_keep_prob = 1.0 - self.dropout)
            S = optimized_trilinear_for_attention([c, q], self.c_maxlen, self.q_maxlen, input_keep_prob = 1.0 - self.dropout)
            mask_q = tf.expand_dims(self.q_mask, 1)
            S_ = tf.nn.softmax(mask_logits(S, mask = mask_q))
            mask_c = tf.expand_dims(self.c_mask, 2)
            S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask = mask_c), dim = 1),(0,2,1))
            self.c2q = tf.matmul(S_, q)
            self.q2c = tf.matmul(tf.matmul(S_, S_T), c)
            attention_outputs = [c, self.c2q, c * self.c2q, c * self.q2c]

        with tf.variable_scope("Model_Encoder_Layer"):
            inputs = tf.concat(attention_outputs, axis = -1)
            self.enc = [conv(inputs, d, name = "input_projection")]
            for i in range(3):
                if i % 2 == 0: # dropout every 2 blocks
                    self.enc[i] = tf.nn.dropout(self.enc[i], 1.0 - self.dropout)
                self.enc.append(
                    residual_block(self.enc[i],
                        num_blocks = 7,
                        num_conv_layers = 2,
                        kernel_size = 5,
                        mask = self.c_mask,
                        num_filters = d,
                        num_heads = nh,
                        seq_len = self.c_len,
                        scope = "Model_Encoder",
                        bias = False,
                        reuse = True if i > 0 else None,
                        dropout = self.dropout)
                    )

        # with tf.variable_scope("Output_Layer"):
        #     start_logits = tf.squeeze(conv(tf.concat([self.enc[1], self.enc[2]],axis = -1),1, bias = False, name = "start_pointer"),-1)
        #     end_logits = tf.squeeze(conv(tf.concat([self.enc[1], self.enc[3]],axis = -1),1, bias = False, name = "end_pointer"), -1)
        #     self.logits = [mask_logits(start_logits, mask = self.c_mask),
        #                    mask_logits(end_logits, mask = self.c_mask)]
        #
        #     logits1, logits2 = [l for l in self.logits]
        #
        #     outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
        #                       tf.expand_dims(tf.nn.softmax(logits2), axis=1))
        #     outer = tf.matrix_band_part(outer, 0, config.disc_ans_limit)
        #     self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
        #     self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
        #     losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #         logits=logits1, labels=self.ans_start)
        #     losses2 = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #         logits=logits2, labels=self.ans_end)
        #     self.loss = tf.reduce_mean(losses + losses2)
        with tf.variable_scope("Output_Layer"):
            pooled = [tf.reduce_max(enc, axis=1, keep_dims=True) for enc in self.enc]
            ans_enc = tf.reduce_sum(self.enc[2]*self.in_answer_feature, axis=1, keep_dims=True)/(1e-6+tf.reduce_sum(self.in_answer_feature, axis=1, keep_dims=True))
            pooled = tf.squeeze(tf.concat(pooled[0:2]+[ans_enc], axis=2), 1)
            hidden = tf.layers.dense(pooled,64,activation=tf.nn.relu, use_bias = False, name = "hidden")
            logits = tf.squeeze(tf.layers.dense(hidden,1, activation=None, use_bias = False, name = "logits"),-1)

            self.nll = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.gold_class, tf.float32), logits=logits)
            self.loss = tf.reduce_mean(self.nll)
            self.probs = tf.nn.sigmoid(logits)
            self.pred = tf.cast(tf.round(self.probs),tf.int32)
            self.equality = tf.equal(self.pred, self.gold_class)
            self.accuracy = tf.reduce_mean(tf.cast(self.equality, tf.float32))

        if config.disc_l2_norm is not None:
            variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            l2_loss = tf.contrib.layers.apply_regularization(regularizer, variables)
            self.loss += l2_loss

        self._train_summaries.append(tf.summary.scalar('train_loss/loss', self.loss))
        self._train_summaries.append(tf.summary.scalar('train_loss/accuracy', self.accuracy))


        if config.disc_decay is not None:
            self.var_ema = tf.train.ExponentialMovingAverage(config.disc_decay)
            ema_op = self.var_ema.apply(tf.trainable_variables())
            with tf.control_dependencies([ema_op]):
                self.loss = tf.identity(self.loss)

                self.assign_vars = []
                for var in tf.global_variables():
                    v = self.var_ema.average(var)
                    if v:
                        self.assign_vars.append(tf.assign(var,v))

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step
