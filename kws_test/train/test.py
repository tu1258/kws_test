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

"""Test utility functions."""
import os
import sys
import absl.logging as logging
import numpy as np
import tensorflow.compat.v1 as tf
from ..train import input_data 
from ..train import tcn 
from ..train import utils

def tf_model_accuracy(
    flags,
    folder,
    # time_shift_samples=0,
    # weights_name='best_weights',
    accuracy_name='tf_test_model_accuracy.txt'):
  """Compute accuracy of non streamable model using TF.

  Args:
      flags: model and data settings
      folder: folder name where accuracy report will be stored
      time_shift_samples: time shift of audio data it will be applied in range:
        -time_shift_samples...time_shift_samples
        We can use non stream model for processing stream of audio.
        By default it will be slow, so to speed it up
        we can use non stream model on sampled audio data:
        for example instead of computing non stream model
        on every 20ms, we can run it on every 200ms of audio stream.
        It will reduce total latency by 10 times.
        To emulate sampling effect we use time_shift_samples.
      weights_name: file name with model weights
      accuracy_name: file name for storing accuracy in path + accuracy_name
  Returns:
    accuracy
  """
  np.set_printoptions(threshold=sys.maxsize)
  tf.reset_default_graph()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  tf.keras.backend.set_session(sess)

  audio_processor = input_data.AudioProcessor(flags)

  if(flags.generate and not flags.train):
    generate = 1
    set_size = 1
  else:
    generate = 0
    set_size = audio_processor.set_size('testing')

  tf.keras.backend.set_learning_phase(0)
  flags.batch_size = 1  # set batch size for inference
  # set_size = int(set_size / flags.batch_size) * flags.batch_size
  model = tcn.model(flags)
  weights_path = os.path.join(flags.train_dir + '/best_weights')
  model.load_weights(weights_path).expect_partial()

  total_accuracy = 0.0
  count = 0.0
  if(generate):
    test_fingerprints, test_ground_truth = audio_processor.get_data(
        flags.batch_size, 1, flags, 0.0, 0.0, 0, 'training',
        0.0, 0.0, sess)
    # print('test_fingerprints:',test_fingerprints)
    # print('shape',test_fingerprints.shape)    
    np.savetxt('data.csv', test_fingerprints, delimiter=',')
    read_fingerprints = np.loadtxt('data.csv', delimiter=',')
    # print('read_fingerprints',read_fingerprints)
    # print('shape',read_fingerprints.shape)
    # print('same:',(test_fingerprints==read_fingerprints))
    read_fingerprints = read_fingerprints.reshape(1,flags.desired_samples)
    # print('shape',read_fingerprints.shape)
    print('test_ground_truth:',test_ground_truth)
    predictions = model.predict(read_fingerprints)
    predicted_labels = np.argmax(predictions, axis=1)
    total_accuracy = total_accuracy + np.sum(
        predicted_labels == test_ground_truth)

  else:
    for i in range(0, set_size, flags.batch_size):
      test_fingerprints, test_ground_truth = audio_processor.get_data(
          flags.batch_size, i, flags, 0.0, 0.0, 0, 'testing',
          0.0, 0.0, sess)
      predictions = model.predict(test_fingerprints)
      predicted_labels = np.argmax(predictions, axis=1)
      total_accuracy = total_accuracy + np.sum(
          predicted_labels == test_ground_truth)
      count = count + len(test_ground_truth)
    if(count):
      total_accuracy = total_accuracy / count
    

  logging.info('best weights Final test accuracy = %.2f%% (N=%d)',
               *(total_accuracy * 100, set_size))

  path = os.path.join(flags.train_dir, folder)
  if not os.path.exists(path):
    os.makedirs(path)

  fname_summary = 'model_summary_test'
  utils.save_model_summary(model, path, file_name=fname_summary + '.txt')

  tf.keras.utils.plot_model(
      model,
      to_file=os.path.join(path, fname_summary + '.png'),
      show_shapes=True,
      expand_nested=True)

  with open(os.path.join(path, accuracy_name), 'wt') as fd:
    fd.write('%f on set_size %d' % (total_accuracy * 100, set_size))
  return total_accuracy * 100

# def convert_model_tflite(flags,
#                          folder,
#                          mode,
#                          fname,
#                          weights_name='best_weights',
#                          optimizations=None):
#   """Convert model to streaming and non streaming TFLite.

#   Args:
#       flags: model and data settings
#       folder: folder where converted model will be saved
#       mode: inference mode
#       fname: file name of converted model
#       weights_name: file name with model weights
#       optimizations: list of optimization options
#   """
#   tf.reset_default_graph()
#   config = tf.ConfigProto()
#   config.gpu_options.allow_growth = True
#   sess = tf.Session(config=config)
#   tf.keras.backend.set_session(sess)
#   tf.keras.backend.set_learning_phase(0)
#   flags.batch_size = 1  # set batch size for inference
#   model = models.MODELS[flags.model_name](flags)
#   weights_path = os.path.join(flags.train_dir, weights_name)
#   # model.load_weights(weights_path).expect_partial()
#   # convert trained model to non streaming TFLite stateless
#   # to finish other tests we do not stop program if exception happen here
#   path_model = os.path.join(flags.train_dir, folder)
#   if not os.path.exists(path_model):
#     os.makedirs(path_model)
#   try:
#     with open(os.path.join(path_model, fname), 'wb') as fd:
#       fd.write(
#           utils.model_to_tflite(sess, model, flags, mode, path_model,
#                                 optimizations))
#   except IOError as e:
#     logging.warning('FAILED to write file: %s', e)
#   except (ValueError, AttributeError, RuntimeError, TypeError) as e:
#     logging.warning('FAILED to convert to mode %s, tflite: %s', mode, e)


def convert_model_saved(flags, folder):#, weights_name='best_weights'):
  """Convert model to streaming and non streaming SavedModel.

  Args:
      flags: model and data settings
      folder: folder where converted model will be saved
      mode: inference mode
      weights_name: file name with model weights
  """
  tf.reset_default_graph()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  tf.keras.backend.set_session(sess)
  tf.keras.backend.set_learning_phase(0)
  flags.batch_size = 1  # set batch size for inference
  model = tcn.model(flags)
  weights_path = os.path.join(flags.train_dir + '/best_weights')
  model.load_weights(weights_path).expect_partial()

  path_model = os.path.join(flags.train_dir, folder)
  if not os.path.exists(path_model):
    os.makedirs(path_model)
  
    # convert trained model to SavedModel
  # save_model_summary(model, path_model)
  model.save(path_model, include_optimizer=False, save_format='tf')
    # model.save(path_model+'_h5', include_optimizer=False, save_format='h5')
    # utils.model_to_saved(model, flags, path_model)#, mode)
  # except IOError as e:
  #   logging.warning('FAILED to write file: %s', e)
  # except (ValueError, AttributeError, RuntimeError, TypeError,
  #         AssertionError) as e:
  #   logging.warning('WARNING: failed to convert to SavedModel: %s', e)
