import numpy as np
import mxnet as mx
from mxnet import gluon

def normalize_and_copy(dataset, ctx):
    data = mx.nd.array(dataset._data, ctx=ctx) \
                .transpose((0, 3, 1, 2)) \
                .astype(np.float32)/255
    label = dataset._label
    output = gluon.data.ArrayDataset(data, label)
    output._data[1] = mx.nd.array(label, ctx=ctx)
    return output
