# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.ops import rnn_cell
from utils import seq2seq

from tensorflow.models.rnn.translate import data_utils


class Seq2SeqModel(object):
  """Sequence-to-sequence model with attention and for multiple buckets.

  This class implements a multi-layer recurrent neural network as encoder,
  and an attention-based decoder. This is the same as the model described in
  this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
  or into the seq2seq library for complete model implementation.
  This class also allows to use GRU cells in addition to LSTM cells, and
  sampled softmax to handle large output vocabulary size. A single-layer
  version of this model, but with bi-directional encoder, was presented in
    http://arxiv.org/abs/1409.0473
  and sampled softmax is described in Section 3 of the following paper.
    http://arxiv.org/pdf/1412.2007v2.pdf
  """

  def __init__(self, source_vocab_size, target_vocab_size, buckets, size,
               num_layers, max_gradient_norm, batch_size, learning_rate,
               learning_rate_decay_factor, use_lstm=False,
               num_samples=512, forward_only=False):
    """Create the model.

    Args:
      source_vocab_size: size of the source vocabulary.
      target_vocab_size: size of the target vocabulary.
      buckets: a list of pairs (I, O), where I specifies maximum input length
        that will be processed in that bucket, and O specifies maximum output
        length. Training instances that have inputs longer than I or outputs
        longer than O will be pushed to the next bucket and padded accordingly.
        We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
      size: number of units in each layer of the model.
      num_layers: number of layers in the model.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      use_lstm: if true, we use LSTM cells instead of GRU cells.
      num_samples: number of samples for sampled softmax.
      forward_only: if set, we do not construct the backward pass in the model.
    """
    self.source_vocab_size = source_vocab_size
    self.target_vocab_size = target_vocab_size
    self.buckets = buckets
    self.batch_size = batch_size
    self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)

    # If we use sampled softmax, we need an output projection.
    output_projection = None
    softmax_loss_function = None

    ######### Sampled softmax for large vocab (more than 512) #########
    #Stackoverflow: If your target vocabulary(or in other words amount of classes you want to predict) is really big, it is
    # very hard to use regular softmax, because you have to calculate probability for every word in dictionary.
    # By Using  sampled_softmax_loss you only take in account subset V of your vocabulary to calculate your loss. (see UsingLargeVoocab.pdf, Cho.pdf)
    # Sampled softmax only makes sense if we sample less than vocabulary size.
    if num_samples > 0 and num_samples < self.target_vocab_size:
      with tf.device("/cpu:0"):
        w = tf.get_variable("proj_w", [size, self.target_vocab_size])
        w_t = tf.transpose(w)
        b = tf.get_variable("proj_b", [self.target_vocab_size])
      output_projection = (w, b)

      def sampled_loss(inputs, labels):
        with tf.device("/cpu:0"):
          labels = tf.reshape(labels, [-1, 1])
          return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples,
                                            self.target_vocab_size)
      softmax_loss_function = sampled_loss

    # Create the internal multi-layer cell for our RNN.
    single_cell = rnn_cell.GRUCell(size)
    if use_lstm:
      single_cell = rnn_cell.BasicLSTMCell(size)
    cell = single_cell
    if num_layers > 1:
      cell = rnn_cell.MultiRNNCell([single_cell] * num_layers)

    # The seq2seq function: we use embedding for the input and attention.
    def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
      return seq2seq.embedding_attention_seq2seq(
          encoder_inputs, decoder_inputs, cell, num_encoder_symbols=source_vocab_size,
          num_decoder_symbols=target_vocab_size, output_projection=output_projection,
          feed_previous=do_decode)

    # Feeds for inputs.
    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []
    for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
      self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder{0}".format(i)))
    for i in xrange(buckets[-1][1] + 1):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))
      self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                name="weight{0}".format(i)))

    # Our targets (target words) are decoder inputs shifted by one.
    targets = [self.decoder_inputs[i + 1]
               for i in xrange(len(self.decoder_inputs) - 1)]

    ########## output feed for session.run() ######
    # Training outputs and losses.
    if forward_only:
      self.outputs, self.losses = seq2seq.model_with_buckets(encoder_inputs=
          self.encoder_inputs, decoder_inputs=self.decoder_inputs, targets=targets,
          weights=self.target_weights, buckets=buckets, num_decoder_symbols=self.target_vocab_size,
          seq2seq=lambda x, y: seq2seq_f(x, y, do_decode=True),
          softmax_loss_function=softmax_loss_function)
      ########## target word scoring (if comment out -> output = decoder's hidden state) #######
      # If we use output projection, we need to project outputs for decoding.
      if output_projection is not None:
        for b in xrange(len(buckets)):
          self.outputs[b] = [tf.nn.xw_plus_b(output, output_projection[0], #xw_plus_b = nn_ops.bias_add(math_ops.matmul(x, weights), biases) in nn.py   tf.matmul([[1,2]], [[2],[3]]) = 8
                                             output_projection[1])
                             for output in self.outputs[b]]
    else:
      self.outputs, self.losses = seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, buckets, self.target_vocab_size,
          lambda x, y: seq2seq_f(x, y, False),
          softmax_loss_function=softmax_loss_function)

    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()
    if not forward_only:
      self.gradient_norms = []
      self.updates = []
      opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      for b in xrange(len(buckets)):
        gradients = tf.gradients(self.losses[b], params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                         max_gradient_norm)
        self.gradient_norms.append(norm)
        self.updates.append(opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step))

    self.saver = tf.train.Saver(tf.all_variables())

  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only):
    """Run a step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      decoder_inputs: list of numpy int vectors to feed as decoder inputs.
      target_weights: list of numpy float vectors to feed as target weights.
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.

    Raises:
      ValueError: if length of enconder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    encoder_size, decoder_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
    else:
      output_feed = [self.losses[bucket_id]]  # Loss for this batch.
      for l in xrange(decoder_size):  # Output logits.   decoder_size = len(decoder_inputs) = len()
        output_feed.append(self.outputs[bucket_id][l])
    # print ("output feed:", output_feed, len(output_feed))
    # print ("input feed:", sorted(input_feed.items()), len(input_feed.items()))
    # print ("output feed:", output_feed)
    outputs = session.run(output_feed, input_feed) #If the *i*th element of fetches(in this case output_feed) is a Tensor, the *i*th return value will be a numpy ndarray containing the value of that tensor
                                                   #===> returns [loss, output1, output2, output3...]
    # print ("--outputs--: ",outputs, "length of outputs:", len(outputs)-1, "size of output vector:", len(outputs[1][0]))

    '''
    for "Hello!" [3707, 299]
    encoder input: [array([0], dtype=int32), array([0], dtype=int32), array([0], dtype=int32), array([299], dtype=int32), array([3707], dtype=int32)] 5
    decoder input: [array([1], dtype=int32), array([0], dtype=int32), array([0], dtype=int32), array([0], dtype=int32), array([0], dtype=int32), array([0], dtype=int32), array([0], dtype=int32), array([0], dtype=int32), array([0], dtype=int32), array([0], dtype=int32)] 10
    target weights: [array([ 0.], dtype=float32), array([ 0.], dtype=float32), array([ 0.], dtype=float32), array([ 0.], dtype=float32), array([ 0.], dtype=float32), array([ 0.], dtype=float32), array([ 0.], dtype=float32), array([ 0.], dtype=float32), array([ 0.], dtype=float32), array([ 0.], dtype=float32)] 10
    input feed (as sorted list): [(u'decoder0:0', array([1], dtype=int32)), (u'decoder10:0', array([0], dtype=int32)), (u'decoder1:0', array([0], dtype=int32)), (u'decoder2:0', array([0], dtype=int32)),
    (u'decoder3:0', array([0], dtype=int32)), (u'decoder4:0', array([0], dtype=int32)), (u'decoder5:0', array([0], dtype=int32)), (u'decoder6:0', array([0], dtype=int32)), (u'decoder7:0', array([0], dtype=int32)),
    (u'decoder8:0', array([0], dtype=int32)), (u'decoder9:0', array([0], dtype=int32)), (u'encoder0:0', array([0], dtype=int32)), (u'encoder1:0', array([0], dtype=int32)), (u'encoder2:0', array([0], dtype=int32)),
    (u'encoder3:0', array([299], dtype=int32)), (u'encoder4:0', array([3707], dtype=int32)), (u'weight0:0', array([ 0.], dtype=float32)), (u'weight1:0', array([ 0.], dtype=float32)), (u'weight2:0', array([ 0.], dtype=float32)),
    (u'weight3:0', array([ 0.], dtype=float32)), (u'weight4:0', array([ 0.], dtype=float32)), (u'weight5:0', array([ 0.], dtype=float32)), (u'weight6:0', array([ 0.], dtype=float32)), (u'weight7:0', array([ 0.], dtype=float32)),
     (u'weight8:0', array([ 0.], dtype=float32)), (u'weight9:0', array([ 0.], dtype=float32))] 26

    output feed: list of operations length decode bucket+1 = 11
    self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                               name="encoder{0}".format(i)))
    self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))
    self.outputs, self.losses = seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, buckets, self.target_vocab_size,
          lambda x, y: seq2seq_f(x, y, True),
          softmax_loss_function=softmax_loss_function)
    -> with session.run() feed encoder_inputs, decoder_inputs etc  -> outputs=output vectors of each decoder's hidden state

    --session.run()--
    y = x^2 + b
    p_x = tf.placeholder(tf.types.float32)
    p_b = tf.placeholder(tf.types.float32)
    p_x2_plus_b = tf.add(tf.square(p_x), p_b)

    with tf.Session() as sess:
        result = sess.run([p_x2_plus_b], feed_dict={p_x: [2.], p_b: [3.]})
        print result  -> [7.0]
    '''
    if not forward_only:
      return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
      return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

  def get_batch(self, data, bucket_id):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in xrange(self.batch_size):
      encoder_input, decoder_input = random.choice(data[bucket_id])

      # Encoder inputs are padded and then reversed.
      encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                            [data_utils.PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in xrange(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights
