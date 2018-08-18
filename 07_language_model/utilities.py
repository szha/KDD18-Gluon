# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Utilities"""

import time
import math

import mxnet as mx
from mxnet import gluon, autograd

def detach(hidden):
    # Detach gradients on states for truncated BPTT
    if isinstance(hidden, (tuple, list)):
        hidden = [detach(i) for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden

def train_one_epoch(epoch, model, train_data, batch_size, grad_clip, log_interval, loss, parameters, trainer, context):
    total_L = 0.0
    start_log_interval_time = time.time()
    hiddens = [model.begin_state(batch_size//len(context), func=mx.nd.zeros, ctx=ctx) 
               for ctx in context]
    for i, (data, target) in enumerate(train_data):
        data_list = gluon.utils.split_and_load(data, context, 
                                               batch_axis=1, even_split=True)
        target_list = gluon.utils.split_and_load(target, context, 
                                                 batch_axis=1, even_split=True)
        hiddens = detach(hiddens)
        Ls = []
        with autograd.record():
            for j, (X, y, h) in enumerate(zip(data_list, target_list, hiddens)):
                output, h = model(X, h)
                batch_L_per_token = loss(output.reshape(-3, -1), y.reshape(-1,)) / X.size
                Ls.append(batch_L_per_token)
                hiddens[j] = h
        for L in Ls:
            L.backward()
        grads = [p.grad(x.context) for p in parameters for x in data_list]
        gluon.utils.clip_global_norm(grads, grad_clip)

        trainer.step(1)

        total_L += sum([L.sum().asscalar() for L in Ls])

        if i % log_interval == 0 and i > 0:
            cur_L = total_L / log_interval
            print('[Epoch %d Batch %d/%d] loss %.2f, ppl %.2f, '
                  'throughput %.2f samples/s'%(
                epoch, i, len(train_data), cur_L, math.exp(cur_L), 
                batch_size * log_interval / (time.time() - start_log_interval_time)))
            total_L = 0.0
            start_log_interval_time = time.time()