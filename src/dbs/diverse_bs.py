# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A decoder that performs beam search."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np

from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import nest

from tensorflow.contrib.seq2seq import BeamSearchDecoder, BeamSearchDecoderState, BeamSearchDecoderOutput

class DiverseBeamSearchDecoder(BeamSearchDecoder):


  def __init__(self,
               cell,
               embedding,
               start_tokens,
               end_token,
               initial_state,
               beam_width,
               output_layer=None,
               length_penalty_weight=0.0,
               coverage_penalty_weight=0.0,
               reorder_tensor_arrays=True,
               num_groups=1,
               diversity_param=0.5):
    super().__init__(cell,
                    embedding,
                    start_tokens,
                    end_token,
                    initial_state,
                    beam_width,
                    output_layer,
                    length_penalty_weight,
                    coverage_penalty_weight,
                    reorder_tensor_arrays)
    self.num_groups = num_groups
    self.diversity_param = diversity_param


def _beam_search_step(time, logits, next_cell_state, beam_state, batch_size,
                      beam_width, end_token, length_penalty_weight,
                      coverage_penalty_weight):
  """Performs a single step of Beam Search Decoding.

  Args:
    time: Beam search time step, should start at 0. At time 0 we assume
      that all beams are equal and consider only the first beam for
      continuations.
    logits: Logits at the current time step. A tensor of shape
      `[batch_size, beam_width, vocab_size]`
    next_cell_state: The next state from the cell, e.g. an instance of
      AttentionWrapperState if the cell is attentional.
    beam_state: Current state of the beam search.
      An instance of `BeamSearchDecoderState`.
    batch_size: The batch size for this input.
    beam_width: Python int.  The size of the beams.
    end_token: The int32 end token.
    length_penalty_weight: Float weight to penalize length. Disabled with 0.0.
    coverage_penalty_weight: Float weight to penalize the coverage of source
      sentence. Disabled with 0.0.

  Returns:
    A new beam state.
  """
  static_batch_size = tensor_util.constant_value(batch_size)

  # Calculate the current lengths of the predictions
  prediction_lengths = beam_state.lengths
  previously_finished = beam_state.finished
  not_finished = math_ops.logical_not(previously_finished)

  # Calculate the total log probs for the new hypotheses
  # Final Shape: [batch_size, beam_width, vocab_size]
  step_log_probs = nn_ops.log_softmax(logits)
  step_log_probs = _mask_probs(step_log_probs, end_token, previously_finished)
  total_probs = array_ops.expand_dims(beam_state.log_probs, 2) + step_log_probs

  # Calculate the continuation lengths by adding to all continuing beams.
  vocab_size = logits.shape[-1].value or array_ops.shape(logits)[-1]
  lengths_to_add = array_ops.one_hot(
      indices=array_ops.fill([batch_size, beam_width], end_token),
      depth=vocab_size,
      on_value=np.int64(0),
      off_value=np.int64(1),
      dtype=dtypes.int64)
  add_mask = math_ops.to_int64(not_finished)
  lengths_to_add *= array_ops.expand_dims(add_mask, 2)
  new_prediction_lengths = (
      lengths_to_add + array_ops.expand_dims(prediction_lengths, 2))

  # Calculate the accumulated attention probabilities if coverage penalty is
  # enabled.
  accumulated_attention_probs = None
  attention_probs = get_attention_probs(
      next_cell_state, coverage_penalty_weight)
  if attention_probs is not None:
    attention_probs *= array_ops.expand_dims(math_ops.to_float(not_finished), 2)
    accumulated_attention_probs = (
        beam_state.accumulated_attention_probs + attention_probs)

  # Calculate the scores for each beam
  scores = _get_scores(
      log_probs=total_probs,
      sequence_lengths=new_prediction_lengths,
      length_penalty_weight=length_penalty_weight,
      coverage_penalty_weight=coverage_penalty_weight,
      finished=previously_finished,
      accumulated_attention_probs=accumulated_attention_probs)

  time = ops.convert_to_tensor(time, name="time")
  # During the first time step we only consider the initial beam
  scores_flat = array_ops.reshape(scores, [batch_size, -1])

  # Pick the next beams according to the specified successors function
  next_beam_size = ops.convert_to_tensor(
      beam_width, dtype=dtypes.int32, name="beam_width")
  next_beam_scores, word_indices = nn_ops.top_k(scores_flat, k=next_beam_size)

  next_beam_scores.set_shape([static_batch_size, beam_width])
  word_indices.set_shape([static_batch_size, beam_width])

  # Pick out the probs, beam_ids, and states according to the chosen predictions
  next_beam_probs = _tensor_gather_helper(
      gather_indices=word_indices,
      gather_from=total_probs,
      batch_size=batch_size,
      range_size=beam_width * vocab_size,
      gather_shape=[-1],
      name="next_beam_probs")
  # Note: just doing the following
  #   math_ops.to_int32(word_indices % vocab_size,
  #       name="next_beam_word_ids")
  # would be a lot cleaner but for reasons unclear, that hides the results of
  # the op which prevents capturing it with tfdbg debug ops.
  raw_next_word_ids = math_ops.mod(
      word_indices, vocab_size, name="next_beam_word_ids")
  next_word_ids = math_ops.to_int32(raw_next_word_ids)
  next_beam_ids = math_ops.to_int32(
      word_indices / vocab_size, name="next_beam_parent_ids")

  # Append new ids to current predictions
  previously_finished = _tensor_gather_helper(
      gather_indices=next_beam_ids,
      gather_from=previously_finished,
      batch_size=batch_size,
      range_size=beam_width,
      gather_shape=[-1])
  next_finished = math_ops.logical_or(
      previously_finished,
      math_ops.equal(next_word_ids, end_token),
      name="next_beam_finished")

  # Calculate the length of the next predictions.
  # 1. Finished beams remain unchanged.
  # 2. Beams that are now finished (EOS predicted) have their length
  #    increased by 1.
  # 3. Beams that are not yet finished have their length increased by 1.
  lengths_to_add = math_ops.to_int64(math_ops.logical_not(previously_finished))
  next_prediction_len = _tensor_gather_helper(
      gather_indices=next_beam_ids,
      gather_from=beam_state.lengths,
      batch_size=batch_size,
      range_size=beam_width,
      gather_shape=[-1])
  next_prediction_len += lengths_to_add
  next_accumulated_attention_probs = ()
  if accumulated_attention_probs is not None:
    next_accumulated_attention_probs = _tensor_gather_helper(
        gather_indices=next_beam_ids,
        gather_from=accumulated_attention_probs,
        batch_size=batch_size,
        range_size=beam_width,
        gather_shape=[batch_size * beam_width, -1],
        name="next_accumulated_attention_probs")

  # Pick out the cell_states according to the next_beam_ids. We use a
  # different gather_shape here because the cell_state tensors, i.e.
  # the tensors that would be gathered from, all have dimension
  # greater than two and we need to preserve those dimensions.
  # pylint: disable=g-long-lambda
  next_cell_state = nest.map_structure(
      lambda gather_from: _maybe_tensor_gather_helper(
          gather_indices=next_beam_ids,
          gather_from=gather_from,
          batch_size=batch_size,
          range_size=beam_width,
          gather_shape=[batch_size * beam_width, -1]),
      next_cell_state)
  # pylint: enable=g-long-lambda

  next_state = BeamSearchDecoderState(
      cell_state=next_cell_state,
      log_probs=next_beam_probs,
      lengths=next_prediction_len,
      finished=next_finished,
      accumulated_attention_probs=next_accumulated_attention_probs)

  output = BeamSearchDecoderOutput(
      scores=next_beam_scores,
      predicted_ids=next_word_ids,
      parent_ids=next_beam_ids)

  return output, next_state
