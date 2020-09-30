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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys
import absl.logging as logging
import numpy as np
import tensorflow.compat.v1 as tf
from ..train import base_parser 
from ..train import model_flags 
# from ..train import tcn
FLAGS = None
tf.enable_eager_execution()


def main(_):
  # Update flags
  flags = model_flags.update_flags(FLAGS)
  print_test_data(flags)


def print_test_data(
    flags):
  """Compute accuracy of non streamable model using TF.

  Args:
      flags: model and data settings

  Returns:
    print
  """
  np.set_printoptions(threshold=sys.maxsize)
  tf.reset_default_graph()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  tf.keras.backend.set_session(sess)

  # audio_processor = input_data.AudioProcessor(flags)
  
  tf.keras.backend.set_learning_phase(0)
  model = tf.keras.models.load_model(flags.train_dir + "/model")
  # model = tcn.model(flags)
  weights_path = os.path.join(flags.train_dir + '/best_weights')
  model.load_weights(weights_path).expect_partial()

  total_accuracy = 0.0
  count = 0.0
  read_fingerprints = np.loadtxt('yes_data.csv', delimiter=',')
  test_fingerprints = read_fingerprints.reshape(1,flags.desired_samples)
  test_ground_truth = 2
  model.run_eagerly = True
  predictions = model.predict(test_fingerprints)
  print('tf.executing_eagerly:',tf.executing_eagerly())
  print('model.run_eagerly:',model.run_eagerly)
  for i in range(len(model.layers)):
    intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                              outputs=model.get_layer(index=i).output)
    intermediate_output = intermediate_layer_model(test_fingerprints)
    print('intermediate_output \n layer=', i, '\n', intermediate_output.numpy)
    # np.savetxt('layer'+str(i)+'.csv'), intermediate_output.numpy, delimiter=',')
    
  predicted_labels = np.argmax(predictions, axis=1)
  total_accuracy = total_accuracy + np.sum(
        predicted_labels == test_ground_truth)

  logging.info('best weights Final test accuracy = %.2f%% ',
               (total_accuracy * 100))

  
if __name__ == '__main__':
  # parser for training/testing data and speach feature flags
  parser = base_parser.base_parser()
  FLAGS, unparsed = parser.parse_known_args()
  if unparsed and tuple(unparsed) != ('--alsologtostderr',):
    raise ValueError('Unknown argument: {}'.format(unparsed))
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

