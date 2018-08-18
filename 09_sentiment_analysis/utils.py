import time
import mxnet as mx
import gluonnlp as nlp
import numpy as np
from mxnet import autograd, gluon

def evaluate(net, dataloader, context):
    log_interval = 400
    loss = mx.gluon.loss.SigmoidBCELoss()
    total_L = 0.0
    total_sample_num = 0
    total_correct_num = 0
    start_log_interval_time = time.time()
    print('Begin Testing...')
    for i, ((data, valid_length), label) in enumerate(dataloader):
        data = mx.nd.transpose(data.as_in_context(context))
        valid_length = valid_length.as_in_context(context).astype(np.float32)
        label = label.as_in_context(context)
        output = net(data, valid_length)
        L = loss(output, label)
        pred = (output > 0.5).reshape(-1)
        total_L += L.sum().asscalar()
        total_sample_num += label.shape[0]
        total_correct_num += (pred == label).sum().asscalar()
        if (i + 1) % log_interval == 0:
            print('[Batch {}/{}] elapsed {:.2f} s'.format(
                i + 1, len(dataloader),
                time.time() - start_log_interval_time))
            start_log_interval_time = time.time()
    avg_L = total_L / float(total_sample_num)
    acc = total_correct_num / float(total_sample_num)
    return avg_L, acc

def train_one_epoch(epoch, trainer, train_dataloader, net, loss, context):
    log_interval = 300
    # Epoch training stats
    start_epoch_time = time.time()
    epoch_L = 0.0
    epoch_sent_num = 0
    epoch_wc = 0
    # Log interval training stats
    start_log_interval_time = time.time()
    log_interval_wc = 0
    log_interval_sent_num = 0
    log_interval_L = 0.0

    for i, ((data, length), label) in enumerate(train_dataloader):
        L = 0
        wc = length.sum().asscalar()
        log_interval_wc += wc
        epoch_wc += wc
        log_interval_sent_num += data.shape[1]
        epoch_sent_num += data.shape[1]
        with autograd.record():
            output = net(data.as_in_context(context).T,
                         length.as_in_context(context)
                               .astype(np.float32))
            L = L + loss(output, label.as_in_context(context)).mean()
        L.backward()
        # Update parameter
        trainer.step(1)
        log_interval_L += L.asscalar()
        epoch_L += L.asscalar()
        if (i + 1) % log_interval == 0:
            print(
                '[Epoch {} Batch {}/{}] elapsed {:.2f} s, '
                'avg loss {:.6f}, throughput {:.2f}K wps'.format(
                    epoch, i + 1, len(train_dataloader),
                    time.time() - start_log_interval_time,
                    log_interval_L / log_interval_sent_num, log_interval_wc
                    / 1000 / (time.time() - start_log_interval_time)))
            # Clear log interval training stats
            start_log_interval_time = time.time()
            log_interval_wc = 0
            log_interval_sent_num = 0
            log_interval_L = 0
    end_epoch_time = time.time()
    
    return epoch_L / epoch_sent_num, epoch_wc / 1000 / (end_epoch_time - start_epoch_time)
