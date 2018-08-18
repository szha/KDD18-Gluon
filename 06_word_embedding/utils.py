import itertools
import logging
import math
import random
import time

import numpy as np

import gluonnlp as nlp
import mxnet as mx
from gluonnlp.base import numba_jitclass, numba_prange, numba_types
from gluonnlp.data import DataStream

window_size = 5
num_negatives = 5


@numba_jitclass([('idx_to_subwordidxs',
                  numba_types.List(numba_types.int_[::1]))])
class SubwordLookup(object):
    """Just-in-time compiled helper class for fast, padded subword lookup.

    SubwordLookup holds a mapping from token indices to variable length subword
    arrays and allows fast access to padded and masked batches of subwords
    given a list of token indices.

    Parameters
    ----------
    length : int
         Number of tokens for which to hold subword arrays.

    """

    def __init__(self, length):
        self.idx_to_subwordidxs = [
            np.arange(1).astype(np.int_) for _ in range(length)
        ]

    def set(self, i, subwords):
        """Set the subword array of the i-th token."""
        self.idx_to_subwordidxs[i] = subwords

    def get(self, indices):
        """Get a padded array and mask of subwords for specified indices."""
        subwords = [self.idx_to_subwordidxs[i] for i in indices]
        lengths = np.array([len(s) for s in subwords])
        length = np.max(lengths)
        subwords_arr = np.zeros((len(subwords), length))
        mask = np.zeros((len(subwords), length))
        for i in numba_prange(len(subwords)):
            s = subwords[i]
            subwords_arr[i, :len(s)] = s
            mask[i, :len(s)] = 1
        return subwords_arr, mask


def skipgram_fasttext_batch(data,
                            negatives_sampler,
                            subword_lookup,
                            context):
    """Create a batch for Skipgram training objective."""
    centers, word_context, word_context_mask = data
    assert len(centers.shape) == 2
    negatives_shape = (len(word_context), 2 * window_size * num_negatives)
    negatives, negatives_mask = remove_accidental_hits(
        negatives_sampler(negatives_shape), word_context)
    context_negatives = mx.nd.concat(word_context, negatives, dim=1)
    masks = mx.nd.concat(word_context_mask, negatives_mask, dim=1)
    labels = mx.nd.concat(
        word_context_mask, mx.nd.zeros_like(negatives), dim=1)

    unique, inverse_unique_indices = np.unique(
        centers.asnumpy(), return_inverse=True)
    inverse_unique_indices = mx.nd.array(inverse_unique_indices, ctx=context)
    subwords, subwords_mask = subword_lookup.get(unique.astype(int))

    return (centers.as_in_context(context),
            context_negatives.as_in_context(context),
            masks.as_in_context(context), labels.as_in_context(context),
            mx.nd.array(subwords, ctx=context),
            mx.nd.array(subwords_mask, ctx=context), inverse_unique_indices)


def train_fasttext_embedding(num_epochs, embedding, embedding_out, data,
                             weights, idx_to_subwordidxs, context,
                             trainer, batch_size=2048):
    loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss()
    subword_lookup = SubwordLookup(len(idx_to_subwordidxs))
    for i, subwords in enumerate(idx_to_subwordidxs):
        subword_lookup.set(i, np.array(subwords, dtype=np.int_))

    negatives_sampler = nlp.data.UnigramCandidateSampler(weights)

    # Helpers for bucketing
    def length_fn(data):
        """Return lengths for bucketing."""
        centers, _, _ = data
        lengths = [
            len(idx_to_subwordidxs[i])
            for i in centers.asnumpy().astype(int).flat
        ]
        return lengths

    def bucketing_batchify_fn(indices, data):
        """Select elements from data batch based on bucket indices."""
        centers, word_context, word_context_mask = data
        return (centers[indices], word_context[indices],
                word_context_mask[indices])

    bucketing_split = 16
    batchify = nlp.data.batchify.EmbeddingCenterContextBatchify(
        batch_size=batch_size * bucketing_split,
        window_size=window_size)
    batches = batchify(data)
    batches = BucketingStream(
        batches, bucketing_split, length_fn, bucketing_batchify_fn)

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train_l_sum = 0
        num_samples = 0
        for i, data in enumerate(batches):
            (center, context_negatives, mask, label, subwords, subwords_mask,
             inverse_unique_indices) = skipgram_fasttext_batch(
                 data, negatives_sampler, subword_lookup, context)
            with mx.autograd.record():
                emb_in = embedding(
                    center,
                    subwords,
                    subwordsmask=subwords_mask,
                    words_to_unique_subwords_indices=inverse_unique_indices)
                emb_out = embedding_out(context_negatives, mask)
                pred = mx.nd.batch_dot(emb_in, emb_out.swapaxes(1, 2))
                l = (
                    loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1))
            l.backward()
            trainer.step(1)
            train_l_sum += l.sum()
            num_samples += center.shape[0]
            if i % 500 == 0:
                mx.nd.waitall()
                wps = num_samples / (time.time() - start_time)
                print(
                    'epoch %d, time %.2fs, iteration %d, throughput=%.2fK wps'
                    % (epoch, time.time() - start_time, i, wps / 1000))

        print('epoch %d, time %.2fs, train loss %.2f' %
              (epoch, time.time() - start_time,
               train_l_sum.asscalar() / num_samples))
        print("")


def remove_accidental_hits(candidates, true_samples):
    """Compute a candidates_mask surpressing accidental hits.

    Accidental hits are candidates that occur in the same batch dimension of
    true_samples.

    """
    candidates_np = candidates.asnumpy()
    true_samples_np = true_samples.asnumpy()

    candidates_mask = np.ones(candidates.shape, dtype=np.bool_)
    for j in range(true_samples.shape[1]):
        candidates_mask &= ~(candidates_np == true_samples_np[:, j:j + 1])

    return candidates, mx.nd.array(candidates_mask, ctx=candidates.context)



class BucketingStream(DataStream):
    """Transform a stream of batches into bucketed batches.

    Parameters
    ----------
    stream : DataStream
        Stream of list of list/tuple of integers (a stream over shards of
        the dataset).
    split : int
        Number of batches to return for each incoming batch.
    length_fn : callable
        Callable to determine the length of each batch dimension in the
        input batch. The resulting array of lengths is used as sort key for
        the buckets.
    batchify_fn : callable
        Extract a bucket batch given selected indices and the input batch.

    """

    def __init__(self, stream, split, length_fn, batchify_fn):
        self._stream = stream
        self._split = split
        self._length_fn = length_fn
        self._batchify_fn = batchify_fn

    def __iter__(self):
        for input_batch in self._stream:
            lengths = self._length_fn(input_batch)
            if isinstance(lengths, mx.nd.NDArray):
                lengths = lengths.asnumpy()
            sorted_lengths = np.argsort(lengths)
            splits = np.array_split(sorted_lengths, self._split)
            for split in splits:
                if len(split):
                    yield self._batchify_fn(split, input_batch)
