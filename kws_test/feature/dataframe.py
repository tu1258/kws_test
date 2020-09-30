# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""split input speech signal into frames."""
import tensorflow.compat.v2 as tf


class DataFrame(tf.keras.layers.Layer):
  """Frame splitter with support of streaming inference.

  In training mode we use tf.signal.frame.
  It receives input data [batch, time] and
  converts it into [batch, frames, frame_size].
  More details at:
  https://www.tensorflow.org/api_docs/python/tf/signal/frame
  In inference mode we do a streaming version of tf.signal.frame:
  we receive input packet with dims [batch, frame_step].
  Then we use it to update internal state buffer in a sliding window manner.
  Return output data with size [batch, frame_size].
  """

  def __init__(self,
              #  mode='TRANING',
               inference_batch_size=1,
               frame_size=400,
               frame_step=160,
               dtype=tf.float32,
               **kwargs):
    super(DataFrame, self).__init__(**kwargs)

    if frame_step > frame_size:
      raise ValueError('frame_step:%d must be <= frame_size:%d' %
                       (frame_step, frame_size))
    # self.mode = mode
    self.inference_batch_size = inference_batch_size
    self.frame_size = frame_size
    self.frame_step = frame_step

  def build(self, input_shape):
    super(DataFrame, self).build(input_shape)

  def call(self, inputs):

    if inputs.shape.rank != 2:  # [Batch, Time]
      raise ValueError('inputs.shape.rank:%d must be 2' % inputs.shape.rank)

    # Extract frames from [Batch, Time] -> [Batch, Frames, frame_size]
    framed_signal = tf.signal.frame(
        inputs, frame_length=self.frame_size, frame_step=self.frame_step)
    return framed_signal

  def get_config(self):
    config = {
        # 'mode': self.mode,
        'inference_batch_size': self.inference_batch_size,
        'frame_size': self.frame_size,
        'frame_step': self.frame_step
    }
    base_config = super(DataFrame, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

