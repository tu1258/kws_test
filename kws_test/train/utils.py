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

"""Utility functions for operations on Model."""

import ast
import os.path


def save_model_summary(model, path, file_name='model_summary.txt'):
  """Saves model summary in text format.

  Args:
    model: Keras model
    path: path where to store model summary
    file_name: model summary file name
  """
  with open(os.path.join(path, file_name), 'wt') as fd:
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))  # pylint: disable=unnecessary-lambda
    model_summary = '\n'.join(stringlist)
    fd.write(model_summary)

def parse(text):
  """Parse model parameters.

  Args:
    text: string with layer parameters: '128,128' or "'relu','relu'".

  Returns:
    list of parsed parameters
  """
  if not text:
    return []
  res = ast.literal_eval(text)
  if isinstance(res, tuple):
    return res
  else:
    return [res]

