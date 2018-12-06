from typing import Callable

from tensorflow.python.layers import base

from tensorflow.python.eager import context
from tensorflow.python.estimator import util as estimator_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import utils as layers_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops

from tensorflow.contrib.layers import fully_connected

import tensorflow as tf
import sys

from helpers.misc_utils import debug_tensor, debug_shape
from helpers.ops import safe_log

FLAGS = tf.app.flags.FLAGS


class CopyLayer(base.Layer):
    """Densely-connected layer class.

  This layer implements the operation:
  `outputs = activation(inputs * kernel + bias)`
  Where `activation` is the activation function passed as the `activation`
  argument (if not `None`), `kernel` is a weights matrix created by the layer,
  and `bias` is a bias vector created by the layer
  (only if `use_bias` is `True`).

  Note: if the input to the layer has a rank greater than 2, then it is
  flattened prior to the initial matrix multiply by `kernel`.

  Arguments:
    units: Integer or Long, dimensionality of the output space.
    activation: Activation function (callable). Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: Initializer function for the weight matrix.
      If `None` (default), weights are initialized using the default
      initializer used by `tf.get_variable`.
    bias_initializer: Initializer function for the bias.
    kernel_regularizer: Regularizer function for the weight matrix.
    bias_regularizer: Regularizer function for the bias.
    activity_regularizer: Regularizer function for the output.
    kernel_constraint: An optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: An optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such cases.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

  Properties:
    units: Python integer, dimensionality of the output space.
    activation: Activation function (callable).
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: Initializer instance (or name) for the kernel matrix.
    bias_initializer: Initializer instance (or name) for the bias.
    kernel_regularizer: Regularizer instance for the kernel matrix (callable)
    bias_regularizer: Regularizer instance for the bias (callable).
    activity_regularizer: Regularizer instance for the output (callable)
    kernel_constraint: Constraint function for the kernel matrix.
    bias_constraint: Constraint function for the bias.
    kernel: Weight matrix (TensorFlow variable or tensor).
    bias: Bias vector, if applicable (TensorFlow variable or tensor).
  """

    def __init__(self, embedding_dim,
                 units,
                 switch_units=64,
                 activation=None,
                 use_bias=False,
                 kernel_initializer=None,
                 bias_initializer=init_ops.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 source_provider: Callable[[], tf.Tensor] = None,
                 source_provider_sl: Callable[[], tf.Tensor] = None,
                 condition_encoding: Callable[[], tf.Tensor] = None,
                 output_mask: Callable[[], tf.Tensor] = None,
                 training_mode=False,
                 vocab_size=None,
                 context_as_set=False,
                 max_copy_size=None,
                 mask_oovs=False,
                 **kwargs):
        super(CopyLayer, self).__init__(trainable=trainable, name=name,
                                        activity_regularizer=activity_regularizer,
                                        **kwargs)
        self.vocab_size = vocab_size
        self.source_provider = source_provider
        self.source_provider_sl = source_provider_sl
        self.embedding_dim = embedding_dim
        self.units = units
        self.switch_units = switch_units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.input_spec = base.InputSpec(min_ndim=2)
        self.training_mode=training_mode
        # self.output_mask=output_mask
        self.max_copy_size=max_copy_size
        self.mask_oovs = mask_oovs
        self.context_as_set=context_as_set
        self.condition_encoding = condition_encoding

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        # print("building copy layer")
        # print(input_shape)
        self.built = True

    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)  # batch x len_source+emb_dim
        # inputs = debug_shape(inputs, "inputs")
        # print(inputs)
        #  [batch_size, emb_dim + len_source] in eval,
        #  [len_target, batch_size,emb_dim + len_source] in train
        source = self.source_provider()  # [batch_size, len_source]
        # source = debug_shape(source,"src")
        source_sl = self.source_provider_sl()

        condition_encoding = self.condition_encoding()
        # condition_encoding = debug_shape(condition_encoding, "cond enc")

        batch_size = tf.shape(source)[0]
        len_source = tf.shape(source)[1]
        shape = tf.shape(inputs)
        is_eval = len(inputs.get_shape()) == 2

        beam_width = tf.constant(1) if is_eval else shape[1]
        # len_target = tf.Print(len_target, [len_target, batch_size, shape[-1]], "input reshape")
        # inputs = tf.reshape(inputs, [-1, shape[-1]])  # [len_target * batch_size, len_source + emb_dim]
        inputs_new = tf.reshape(inputs,
                                [batch_size*beam_width, shape[-1]])  # [len_target, batch_size, len_source + emb_dim]

        # inputs_new = debug_shape(inputs_new, "inputs_new")
        # -- [len_target, batch_size, embedding_dim] attention, []
        # -- [len_target, batch_size, len_source] alignments
        # attention, alignments = tf.split(inputs, [self.embedding_dim, -1], axis=1)
        attention, alignments = tf.split(inputs_new, num_or_size_splits=[self.embedding_dim, -1], axis=-1)
        # [len_target, batch_size, vocab_size]
        if FLAGS.out_vocab_cpu:
            with tf.device('/cpu:*'):
                shortlist = tf.layers.dense(attention, self.vocab_size, activation=tf.nn.softmax, use_bias=False)
        else:
            shortlist = tf.layers.dense(attention, self.vocab_size, activation=tf.nn.softmax, use_bias=False)

        # attention = debug_shape(attention, "attn")
        # alignments = debug_shape(alignments, "align ("+str(self.units)+" desired)")
        # alignments = debug_tensor(alignments, "alignments")
        # print(alignments)
        # shortlist = debug_shape(shortlist, "shortlist")

        # TEMP: kill OOVs
        s = tf.shape(shortlist)
        mask = tf.concat([tf.ones((s[0],1)),tf.zeros((s[0],1)),tf.ones((s[0],s[1]-2))], axis=1)
        shortlist = tf.cond(self.mask_oovs, lambda: shortlist * mask, lambda: shortlist)


        # pad the alignments to the longest possible source st output vocab is fixed size
        # TODO: Check for non zero alignments outside the seq length
        # alignments_padded = debug_shape(alignments_padded, "align padded")
        # switch takes st, vt and yt−1 as inputs
        # vt = concat(weighted context encoding at t; condition encoding)
        # st = hidden state at t
        # y_t-1 is previous generated token

        condition_encoding_tiled = tf.contrib.seq2seq.tile_batch(condition_encoding, multiplier=beam_width)

        vt = tf.concat([attention, condition_encoding_tiled], axis=1)
        # NOTE: this is missing the previous input y_t-1 and s_t
        switch_input = tf.concat([vt],axis=1)
        switch_h1 = tf.layers.dropout(tf.layers.dense(switch_input, self.switch_units, activation=tf.nn.tanh, kernel_initializer=tf.glorot_uniform_initializer()), rate=0.3, training=self.training_mode)
        switch_h2 = tf.layers.dropout(tf.layers.dense(switch_h1, self.switch_units, activation=tf.nn.tanh, kernel_initializer=tf.glorot_uniform_initializer()), rate=0.3, training=self.training_mode)
        self.switch = tf.layers.dense(switch_h2, 1, activation=tf.sigmoid, kernel_initializer=tf.glorot_uniform_initializer())
        # switch = debug_shape(switch, "switch")
        if FLAGS.disable_copy:
            self.switch = 0
        elif FLAGS.disable_shortlist:
            self.switch = 1


        # if self.output_mask is not None:
        #     alignments = self.output_mask() * alignments


        source_tiled = tf.contrib.seq2seq.tile_batch(source, multiplier=beam_width)
        source_tiled_sl = tf.contrib.seq2seq.tile_batch(source_sl, multiplier=beam_width)

        shortlist = (1-self.switch)*shortlist
        alignments = self.switch*alignments


        # Take any tokens that are the same in either vocab and combine their probabilities
        # old: mult by a big sparse matrix - not v mem efficient..
        # opt1: mult the copy dist by a vocab x copy matrix and add to vocab part
        # opt2: do an nd_gather to copy the relevant prob mass, then mask carefully to remove it
        if FLAGS.combine_vocab:
            # copy everything in real shortlist except special toks
            # print(len_source, self.max_copy_size)
            source_tiled_sl_padded = tf.pad(source_tiled_sl, [[0, 0], [0, self.max_copy_size-tf.shape(source_tiled_sl)[-1]]], 'CONSTANT', constant_values=0)

            # attempt 2!
            batch_ix = tf.tile(tf.expand_dims(tf.range(batch_size*beam_width),axis=-1),[1,len_source])
            # seq_ix = tf.tile(tf.expand_dims(tf.range(len_source),axis=0),[batch_size*beam_width,1])
            tgt_indices = tf.reshape(tf.concat([tf.expand_dims(batch_ix,-1),tf.expand_dims(source_tiled_sl,-1)], axis=2),[-1,2])
            ident_indices = tf.where(tf.greater(source_tiled_sl, -1)) # get ixs of all elements
            # ident_indices = tf.where()
            # tgt_indices = debug_tensor(tgt_indices)

            # get the copy probs at each point in the source..
            updates = tf.reshape(tf.gather_nd(alignments, ident_indices),[-1])
            # and send them to the their shortlist index
            sum_part = tf.scatter_nd(tgt_indices, updates, [batch_size*beam_width, self.vocab_size+self.max_copy_size])
            # then zero out the ix's that got copied
            align_zeroed = alignments * tf.cast(tf.greater_equal(source_tiled_sl,self.vocab_size),tf.float32)
            align_moved = alignments * tf.cast(tf.less(source_tiled_sl,self.vocab_size),tf.float32) # ie only let through stuff that *isnt* in SL
            # and add the correct pieces together
            alignments = align_zeroed
            shortlist = shortlist + sum_part[:,:self.vocab_size]
            # result = tf.Print(result, [tf.reduce_sum(result[:,:self.vocab_size],-1)], "result sl sum")
            # shortlist = tf.Print(shortlist, [tf.reduce_sum(align_moved,-1)], "sum align_moved")
            # shortlist = tf.Print(shortlist, [tf.reduce_sum(sum_part[:,:self.vocab_size],-1)], "sum sum_part")


        # convert position probs to ids
        if self.context_as_set:
            # print(source) # batch x seq
            # print(alignments) # batch x seq
            pos_to_id = tf.one_hot(source_tiled-self.vocab_size, depth=self.max_copy_size) # batch x seq x vocab
            if FLAGS.maxout_pointer:
                copy_dist = tf.reduce_max(pos_to_id * tf.expand_dims(alignments, 2), axis=1)
            else:
                copy_dist = tf.squeeze(tf.matmul(tf.expand_dims(alignments,1), pos_to_id), axis=1)
        else:
            copy_dist=alignments




        copy_dist_padded = tf.pad(copy_dist, [[0, 0], [0, self.max_copy_size-tf.shape(copy_dist)[-1]]], 'CONSTANT', constant_values=0)

        result = tf.concat([shortlist,copy_dist_padded], axis=1) # this used to be safe_log'd

        # if FLAGS.combine_vocab:
            # result = tf.Print(result, [tf.reduce_sum(result,-1)], "result sum")

        target_shape = tf.concat([shape[:-1], [-1]], 0)

        result =tf.reshape(result, target_shape)

        return result
        # return tf.Print(result, [tf.reduce_max(switch), tf.reduce_max(shortlist),
        #                          tf.reduce_max(alignments)], summarize=10)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)

        # print(input_shape)
        if input_shape[-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.units+self.vocab_size if not self.context_as_set else self.vocab_size+self.max_copy_size)

    # this for older tf versions
    def _compute_output_shape(self, input_shape):
        return self.compute_output_shape(input_shape)

def dense(
        inputs, units,
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=init_ops.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        reuse=None):
    """Functional interface for the densely-connected layer.

  This layer implements the operation:
  `outputs = activation(inputs.kernel + bias)`
  Where `activation` is the activation function passed as the `activation`
  argument (if not `None`), `kernel` is a weights matrix created by the layer,
  and `bias` is a bias vector created by the layer
  (only if `use_bias` is `True`).

  Note: if the `inputs` tensor has a rank greater than 2, then it is
  flattened prior to the initial matrix multiply by `kernel`.

  Arguments:
    inputs: Tensor input.
    units: Integer or Long, dimensionality of the output space.
    activation: Activation function (callable). Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: Initializer function for the weight matrix.
      If `None` (default), weights are initialized using the default
      initializer used by `tf.get_variable`.
    bias_initializer: Initializer function for the bias.
    kernel_regularizer: Regularizer function for the weight matrix.
    bias_regularizer: Regularizer function for the bias.
    activity_regularizer: Regularizer function for the output.
    kernel_constraint: An optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: An optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: String, the name of the layer.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

  Returns:
    Output tensor.

  Raises:
    ValueError: if eager execution is enabled.
  """
    layer = CopyLayer(units,
                      activation=activation,
                      use_bias=use_bias,
                      kernel_initializer=kernel_initializer,
                      bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer,
                      bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer,
                      kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint,
                      trainable=trainable,
                      name=name,
                      dtype=inputs.dtype.base_dtype,
                      _scope=name,
                      _reuse=reuse)

    print("inside copy layer, yaaay!")
    sys.exit(0)

    return layer.apply(inputs)
