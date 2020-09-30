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

"""Model based on combination of n by 1 convolutions with residual blocks."""

import tensorflow.compat.v2 as tf
from tensorflow.keras.layers import *
from ..train import utils
from ..feature import speech_features
from ..feature import dataframe 
from ..feature import dct 
from ..feature import magnitude_rdft_mel 
from ..feature import windowing 
# from ..qkeras import qkeras

def model(flags):
  """Temporal Convolution ResNet model.

  It is based on paper:
  Temporal Convolution for Real-time Keyword Spotting on Mobile Devices
  https://arxiv.org/pdf/1904.03814.pdf
  Args:
    flags: data/model parameters

  Returns:
    Keras model for training
  """
  # input_audio = tf.keras.layers.Input(
  #               shape=flags.desired_samples,
  #               batch_size=flags.batch_size,
  #               dtype=tf.float32)
  input_audio = Input(
                shape=flags.desired_samples,
                batch_size=flags.batch_size)
  net = input_audio

  frame_size = int(flags.sample_rate * flags.window_size_ms / 1000)
  frame_stride = int(flags.sample_rate * flags.window_stride_ms / 1000)

  net = speech_features.SpeechFeatures(
        speech_features.SpeechFeatures.get_params(flags),
        name='feature_layer')(
          net)
  # net = dataframe.DataFrame(
  #       inference_batch_size=1,
  #       frame_size=frame_size,
  #       frame_step=frame_stride)(
  #       net)
  # net = windowing.Windowing(
  #       window_size=frame_size, 
  #       window_type=flags.window_type)(
  #       net)
  # net = magnitude_rdft_mel.MagnitudeRDFTmel(
  #       num_mel_bins=flags.mel_num_bins,
  #       lower_edge_hertz=flags.mel_lower_edge_hertz,
  #       upper_edge_hertz=flags.mel_upper_edge_hertz,
  #       sample_rate=flags.sample_rate,
  #       mel_non_zero_only=flags.mel_non_zero_only)(
  #       net)
  # net = Lambda(lambda x: tf.math.log(tf.math.maximum(x, flags.log_epsilon)))(net)
  # net = dct.DCT(num_features=flags.dct_num_features)(net)
          

  time_size, feature_size = net.shape[1:3]

  channels = utils.parse(flags.channels)

  net = tf.keras.backend.expand_dims(net)
    
  net = tf.reshape(
      net, [-1, time_size, 1, feature_size])  # [batch, time, 1, feature]
  first_kernel = utils.parse(flags.first_kernel)
  conv_kernel = utils.parse(flags.kernel_size)
  groups = flags.groups
  layer = 0

  net = Conv2D(
        filters=channels[0],
        kernel_size=first_kernel,
        strides=1,
        padding='same',
        use_bias=True,
        activation='linear',
        name='first_conv')(
          net)

  net = BatchNormalization(
        momentum=0.997,
        name='first_bn')(
          net)
  # net = tf.keras.layers.Activation('relu',name='first_act')(net)
  net = Activation('relu',name='first_act')(net)

  channels = channels[1:]

  for n in channels:
    # if(flags.residual):
    #   residual = net

    if(flags.mobile):
      # net = tf.keras.layers.SeparableConv2D(
      # filters=n,
      # kernel_size=conv_kernel,
      # strides=1,
      # padding='same',
      # use_bias=True,
      # activation='linear',
      # name='sp'+str(layer))(
      #   net)
      net = DepthwiseConv2D(
            kernel_size=conv_kernel,
            strides=1,
            padding='same',
            use_bias=True,
            activation='linear',
            name='dw'+str(layer))(
              net)
      net = Conv2D(
            filters=n,
            groups=groups,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=True,
            activation='linear',
            name='pw'+str(layer))(
              net)
        
    else:
      # net = tf.keras.layers.Conv2D(
      #       filters=n,
      #       groups=groups,
      #       kernel_size=conv_kernel,
      #       strides=1,
      #       padding='same',
      #       use_bias=True,
      #       activation='linear',
      #       name='conv'+str(layer))(
      #         net)
      net = Conv2D(
            filters=n,
            groups=groups,
            kernel_size=conv_kernel,
            strides=1,
            padding='same',
            use_bias=True,
            activation='linear',
            name='conv'+str(layer))(
              net)

    net = BatchNormalization(
          momentum=0.997,
          name='bn'+str(layer))(
            net)
      # if(flags.residual and layer%2==1):
      #   net = tf.keras.layers.Add(name='add'+str(layer))([net, residual])

      # net = tf.keras.layers.Activation('relu',name='act'+str(layer))(net)
    net = Activation('relu',name='act'+str(layer))(net)

    layer = layer + 1

  net = AveragePooling2D(pool_size=net.shape[1:3], strides=1,name='pool')(net)

  # net = tf.keras.layers.Dropout(rate=flags.dropout)(net)

  # fully connected layer
  net = Conv2D(
        filters=flags.label_count,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=True,
        activation='linear',
        name='last_conv')(
          net)

  net = tf.reshape(net, shape=(-1, net.shape[3]), name='reshape')
  return tf.keras.Model(input_audio, net)
