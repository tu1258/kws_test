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

"""
Simple speech recognition to spot a limited number of keywords.

It is based on tensorflow/examples/speech_commands
This is a self-contained example script that will train a very basic audio recognition model in TensorFlow.

This network uses a keyword detection style to spot discrete words from a small vocabulary,
consisting of "yes", "no", "up", "down", "left", "right", "on", "off", "stop", and "go".

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys
import absl.logging as logging
import tensorflow.compat.v1 as tf
from ..train import base_parser 
from ..train import model_flags 
from ..train import train 
from ..train import test 
# from ..train import model_print_data
FLAGS = None


def main(_):
  # Update flags
  flags = model_flags.update_flags(FLAGS)

  if flags.train:
    # Create model folders where logs and model will be stored
    os.makedirs(flags.train_dir)
    os.mkdir(flags.summaries_dir)

    # Model training
    train.train(flags)

    # convert to SavedModel
    test.convert_model_saved(flags, 'model')

  else:
    if not os.path.isdir(flags.train_dir):
      raise ValueError('model is not trained set "--train 1" and retrain it')

  # write all flags settings into json
  # with open(os.path.join(flags.train_dir, 'flags.json'), 'wt') as f:
  #   json.dump(flags.__dict__, f)
  
  # try:
  #   test.convert_model_saved(flags, 'stream_state_internal',
  #                            Modes.STREAM_INTERNAL_STATE_INFERENCE)
  # except (ValueError, IndexError) as e:
  #   logging.info('FAILED to run TF streaming: %s', e)

  logging.info('run best weights model accuracy evaluation')
  # with TF
  folder_name = 'test'
  test.tf_model_accuracy(flags, folder_name)

  # if(flags.print):
  #   print_data.print_test_data(flags, folder_name)

if __name__ == '__main__':
  # parser for training/testing data and speach feature flags
  parser = base_parser.base_parser()

 
  FLAGS, unparsed = parser.parse_known_args()
  if unparsed and tuple(unparsed) != ('--alsologtostderr',):
    raise ValueError('Unknown argument: {}'.format(unparsed))
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
