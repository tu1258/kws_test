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

"""Base parser with training/testing data speech features flags ."""

import argparse
import absl.logging as logging

def base_parser():
  """Base parser.

  Flags parsing is splitted into two parts:
  1) base parser for training/testing data and speech flags, defined here.
  2) model parameters parser - it is defined in model file.

  Returns:
    parser
  """
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/speech_dataset/',
      help="""\
      Where to download the speech training data to.
      """)
  parser.add_argument(
      '--lr_schedule',
      type=str,
      default='linear',
      help="""\
      Learning rate schedule: linear, exp.
      """)
  parser.add_argument(
      '--optimizer',
      type=str,
      default='adam',
      help='Optimizer: adam, sgdm')
  parser.add_argument(
      '--background_volume',
      type=float,
      default=0.1,
      help="""\
      How loud the background noise should be, between 0 and 1.
      """)
#   parser.add_argument(
#       '--regularizer',
#       type=str,
#       default='',
#       help="""\
#       regularization: l1, l2, None
#       """)
  parser.add_argument(
      '--background_frequency',
      type=float,
      default=0.8,
      help="""\
      How many of the training samples have background noise mixed in.
      """)
  parser.add_argument(
      '--silence_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be silence.
      """)
  parser.add_argument(
      '--unknown_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be unknown words.
      """)
  parser.add_argument(
      '--time_shift_ms',
      type=float,
      default=100.0,
      help="""\
      Range to randomly shift the training audio by in time.
      """)
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a test set.')
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a validation set.')
  parser.add_argument(
      '--how_many_training_epochs',
      type=str,
      default='100,100,100',
      help='How many training loops to run',
  )
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=200,
      help='How often to evaluate the training results.')
  parser.add_argument(
      '--learning_rate',
      type=str,
      default='0.0005,0.0001,0.00002',
      help='How large a learning rate to use when training.')
  parser.add_argument(
      '--batch_size',
      type=int,
      default=128,
      help='How many items to train with at once',
  )
  parser.add_argument(
      '--wanted_words',
      type=str,
      default='yes,no,up,down,left,right,on,off,stop,go',
      help='Words to use (others will be added to an unknown label)',
  )
  parser.add_argument(
      '--train_dir',
      type=str,
      default='/tmp/speech_commands_train',
      help='Directory to write event logs and checkpoint.')
  parser.add_argument(
      '--save_step_interval',
      type=int,
      default=1000,
      help='Save model checkpoint every save_steps.')
  parser.add_argument(
      '--start_checkpoint',
      type=str,
      default='',
      help='If specified, restore this pretrained model before any training.')
  parser.add_argument(
      '--resample',
      type=float,
      default=0.15,
      help='Resample input signal to generate more data [0.0...0.15].',
  )
  parser.add_argument(
      '--volume_resample',
      type=float,
      default=0.0,
      help='Controlls audio volume, '
      'sampled in range: [1-volume_resample...1+volume_resample].',
  )

  parser.add_argument(
      '--train',
      type=int,
      default=1,
      help='If 1 run train and test, else run only test',
  )
  parser.add_argument(
      '--qkeras',
      type=int,
      default=0,
      help='If 1 import qkeras, else import keras',
  )
  parser.add_argument(
      '--debug',
      type=int,
      default=0,
      help='If 1 run only 10 steps, else run full epoch',
  )
  parser.add_argument(
      '--generate',
      type=int,
      default=0,
      help='If 1 generate test data, else run only test',
  )
  parser.add_argument(
      '--print',
      type=int,
      default=0,
      help='If 1 print the value',
  )
  # speech feature extractor properties
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Expected sample rate of the wavs',
  )
  parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=1000,
      help='Expected duration in milliseconds of the input wavs',
  )
  parser.add_argument(
      '--window_size_ms',
      type=float,
      default=40.0,
      help='How long each spectrogram timeslice is.',
  )
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=20.0,
      help='How far to move in time between spectrogram timeslices.',
  )
  parser.add_argument(
      '--preemph',
      type=float,
      default=0.0,
      help='Preemphasis filter standard value 0.97, '
      'if 0.0 - preemphasis will disabled',
  )
  parser.add_argument(
      '--window_type',
      type=str,
      default='hann',
      help='Window type used on input frame before computing Fourier Transform',
  )
  parser.add_argument(
      '--mel_lower_edge_hertz',
      type=float,
      default=20.0,
      help='Lower bound on the frequencies to be included in the mel spectrum.'
      'This corresponds to the lower edge of the lowest triangular band.',
  )
  parser.add_argument(
      '--mel_upper_edge_hertz',
      type=float,
      default=7000.0,
      help='The desired top edge of the highest frequency band',
  )
  parser.add_argument(
      '--log_epsilon',
      type=float,
      default=1e-12,
      help='epsilon for log function in speech feature extractor',
  )
  parser.add_argument(
      '--dct_num_features',
      type=int,
      default=20,
      help='Number of features left after DCT',
  )
  parser.add_argument(
      '--mel_non_zero_only',
      type=int,
      default=1,
      help='if 1 we will check non zero range of mel spectrum and use it'
      'to reduce DFT computation, otherwise full DFT is computed, '
      'it can not be used together with use_tf_fft')
  parser.add_argument(
      '--mel_num_bins',
      type=int,
      default=40,
      help='How many bands in the resulting mel spectrum.',
  )
  parser.add_argument(
      '--mobile',
      type=int,
      default=0,
      help="""\
      mobilenet
      """)
  parser.add_argument(
      '--residual',
      type=int,
      default=0,
      help="""\
      residual net
      """)
  parser.add_argument(
      '--bn',
      type=int,
      default=0,
      help="""\
      batch normalization
      """)
  parser.add_argument(
      '--channels',
      type=str,
      default='36, 36, 36, 36, 36, 36, 36',
      help='Number of channels per convolutional block (including first conv)',
  )
  parser.add_argument(
      '--groups',
      type=int,
      default='1',
      help='Number of groups per layer ',
  )
#   parser.add_argument(
#       '--pool_size',
#       type=str,
#       default='',
#       help="Pool size for example '4,4'",
#   )
  parser.add_argument(
      '--kernel_size',
      type=str,
      default='(3,1)',
      help='Kernel size of conv layer',
  )
  parser.add_argument(
      '--first_kernel',
      type=str,
      default='(3,1)',
      help='Kernel size of conv layer',
  )
  parser.add_argument(
      '--dropout',
      type=float,
      default=0.2,
      help='Percentage of data dropped',
  )
  return parser
