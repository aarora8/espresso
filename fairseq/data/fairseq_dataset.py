# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import logging
import torch.utils.data
from fairseq.data import data_utils
from collections import Counter

logger = logging.getLogger(__name__)

class EpochListening:
    """Mixin for receiving updates whenever the epoch increments."""

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        """
        Whether we can reuse the :class:`fairseq.data.EpochBatchIterator` for
        this dataset across epochs.

        This needs to return ``False`` if the sample sizes can change across
        epochs, in which case we may need to regenerate batches at each epoch.
        If your dataset relies in ``set_epoch`` then you should consider setting
        this to ``False``.
        """
        return True

    def set_epoch(self, epoch):
        """Will receive the updated epoch number at the beginning of the epoch."""
        pass


class FairseqDataset(torch.utils.data.Dataset, EpochListening):
    """A dataset that provides helpers for batching."""

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        raise NotImplementedError

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        raise NotImplementedError

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        raise NotImplementedError

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return np.arange(len(self), dtype=np.int64)

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False

    def attr(self, attr: str, index: int):
        return getattr(self, attr, None)

    def prefetch(self, indices):
        """Prefetch the data required for this epoch."""
        raise NotImplementedError

    def get_batch_shapes(self):
        """
        Return a list of valid batch shapes, for example::

            [(8, 512), (16, 256), (32, 128)]

        The first dimension of each tuple is the batch size and can be ``None``
        to automatically infer the max batch size based on ``--max-tokens``.
        The second dimension of each tuple is the max supported length as given
        by :func:`fairseq.data.FairseqDataset.num_tokens`.

        This will be used by :func:`fairseq.data.FairseqDataset.batch_by_size`
        to restrict batch shapes. This is useful on TPUs to avoid too many
        dynamic shapes (and recompilations).
        """
        return None

    def batch_by_size(
        self,
        indices,
        max_tokens=None,
        max_sentences=None,
        required_batch_size_multiple=1,
    ):
        """
        Given an ordered set of indices, return batches according to
        *max_tokens*, *max_sentences* and *required_batch_size_multiple*.
        """
        fixed_shapes = self.get_batch_shapes()
        logger.info("the max number of tokens are {}".format(max_tokens))
        print(self)
        print("the batch shape is {}".format(fixed_shapes))
        print("the max number of sentence is {}".format(max_sentences))
        print("printing number of tokens in sentence 1 {}".format(self.num_tokens(1)))
        if fixed_shapes is not None:

            def adjust_bsz(bsz, num_tokens):
                if bsz is None:
                    assert max_tokens is not None, "Must specify --max-tokens"
                    bsz = max_tokens // num_tokens
                if max_sentences is not None:
                    bsz = min(bsz, max_sentences)
                elif (
                    bsz >= required_batch_size_multiple
                    and bsz % required_batch_size_multiple != 0
                ):
                    bsz -= bsz % required_batch_size_multiple
                return bsz

            fixed_shapes = np.array(
                [
                    [adjust_bsz(bsz, num_tokens), num_tokens]
                    for (bsz, num_tokens) in fixed_shapes
                ]
            )
        logger.info("end batch by size fairseq dataset")
        return self.batch_by_size_utils(
            indices,
            num_tokens_fn=self.num_tokens,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
            fixed_shapes=fixed_shapes,
        )

    def batch_by_size_utils(
            self,
            indices,
            num_tokens_fn,
            max_tokens=None,
            max_sentences=None,
            required_batch_size_multiple=1,
            fixed_shapes=None,
    ):
        try:
            from fairseq.data.data_utils_fast import (
                batch_by_size_fast,
                batch_fixed_shapes_fast,
            )
        except ImportError:
            raise ImportError(
                "Please build Cython components with: `pip install --editable .` "
                "or `python setup.py build_ext --inplace`"
            )

        max_tokens = max_tokens if max_tokens is not None else -1
        max_sentences = max_sentences if max_sentences is not None else -1
        bsz_mult = required_batch_size_multiple

        if not isinstance(indices, np.ndarray):
            indices = np.fromiter(indices, dtype=np.int64, count=-1)

        logger.info("enter batch by size utils")
        num_tokens_vec = []
        for i in range(indices.size):
            num_tokens_vec.append(num_tokens_fn(i))
        if fixed_shapes is None:
            return self.batch_by_size_utils_baseline_triplet(
                indices,
                num_tokens_vec,
                max_tokens,
                max_sentences,
                bsz_mult,
            )
        else:
            fixed_shapes = np.array(fixed_shapes, dtype=np.int64)
            sort_order = np.lexsort(
                [
                    fixed_shapes[:, 1].argsort(),  # length
                    fixed_shapes[:, 0].argsort(),  # bsz
                ]
            )
            fixed_shapes_sorted = fixed_shapes[sort_order]
            return batch_fixed_shapes_fast(indices, num_tokens_fn, fixed_shapes_sorted)

    def batch_by_size_utils_baseline(
            self,
            indices,
            num_tokens_vec,
            max_tokens,
            max_sentences,
            bsz_mult,
    ):
        """Simple, reliable and slow implementation of batch by size """
        batches = []
        start = 0
        logger.info("enter batch by size utils baseline")
        print(self.tgt_dict.values())
        print(
            "***************************************************************************************||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        print('| {} examples/sentences'.format(len(indices)))
        while start < len(indices):
            for end in range(start + 1, len(indices) + 1):
                max_val = max(num_tokens_vec[pos] for pos in range(start, end))
                sent_count = end - start
                num_tokens = max_val * sent_count
                overflow = num_tokens > max_tokens > 0 or sent_count > max_sentences > 0
                terminate = overflow or end == len(indices)
                if overflow:
                    sent_count -= 1
                if terminate:
                    if sent_count > bsz_mult:
                        sent_count = sent_count - sent_count % bsz_mult
                    batches.append(indices[start: start + sent_count])
                    start = start + sent_count
                    break

        print("the number of batches are {}".format(len(batches)))
        print("the batch shape is {}".format(np.shape(batches)))
        print("the batch 1 is {}".format(batches[1]))
        # sentence = input('\nInput: ')
        return batches

    def batch_by_size_utils_baseline_triplet(
            self,
            indices,
            num_tokens_vec,
            max_tokens,
            max_sentences,
            bsz_mult,
    ):
        """Simple, reliable and slow implementation of batch by size """
        batches = []
        start = 0
        logger.info("enter batch by size utils baseline")
        print(self.tgt_dict.values())
        print(
            "***************************************************************************************||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        print('| {} examples/sentences'.format(len(indices)))
        while start < len(indices):
            for end in range(start + 1, len(indices) + 1):
                max_val = max(num_tokens_vec[pos] for pos in range(start, end))
                sent_count = end - start
                num_tokens = max_val * sent_count
                overflow = num_tokens > max_tokens > 0 or sent_count > max_sentences > 0
                terminate = overflow or end == len(indices)
                if overflow:
                    sent_count -= 1
                if terminate:
                    if sent_count > bsz_mult:
                        sent_count = sent_count - sent_count % bsz_mult
                    batches.append(indices[start: start + sent_count])
                    start = start + sent_count
                    break

        print("the number of batches are {}".format(len(batches)))
        print("the batch shape is {}".format(np.shape(batches)))
        print("the batch 1 is {}".format(batches[1]))
        # sentence = input('\nInput: ')

        print("unique lables are {}".format(np.unique(self.tgt)))
        print("lables at fifth location is {}".format(self.tgt[4].numpy()))
        print("lables at fifth location is {}".format(self.tgt[253].numpy()))
        tensor_to_np_labels = np.array(self.tgt)
        id_counts = Counter(tensor_to_np_labels)
        print(tensor_to_np_labels)
        same = []
        for index, word_id in enumerate(self.tgt):  # collect same samples
            indices = np.where(tensor_to_np_labels == word_id.numpy())[0]
            same.append(np.random.permutation(indices[indices != index])[:1])
        # same = np.array(same)
        # print("value same {}".format(same))
        # print("shape same ", same.shape)


        diff_ids = np.random.randint(0, len(self.tgt_dict) - 1, (len(self.tgt), 5))
        diff_ids[diff_ids >= np.tile(tensor_to_np_labels.reshape(-1, 1), [1, 5])] += 1
        diff = np.full_like(diff_ids, 0, dtype=np.int32)
        for word_id, count in id_counts.items():  # collect diff samples
            indices = np.where(diff_ids == word_id)
            diff[indices] = np.where(tensor_to_np_labels == word_id)[0][np.random.randint(0, count, len(indices[0]))]
            # for i in range(len(labels)):
            #     print("value same {}".format(diff[i]))
            # sentence = input('\nInput: ')
        return batches

    def filter_indices_by_size(self, indices, max_sizes):
        """
        Filter a list of sample indices. Remove those that are longer than
        specified in *max_sizes*.

        WARNING: don't update, override method in child classes

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        if isinstance(max_sizes, float) or isinstance(max_sizes, int):
            if hasattr(self, "sizes") and isinstance(self.sizes, np.ndarray):
                ignored = indices[self.sizes[indices] > max_sizes].tolist()
                indices = indices[self.sizes[indices] <= max_sizes]
            elif (
                hasattr(self, "sizes")
                and isinstance(self.sizes, list)
                and len(self.sizes) == 1
            ):
                ignored = indices[self.sizes[0][indices] > max_sizes].tolist()
                indices = indices[self.sizes[0][indices] <= max_sizes]
            else:
                indices, ignored = data_utils._filter_by_size_dynamic(
                    indices, self.size, max_sizes
                )
        else:
            indices, ignored = data_utils._filter_by_size_dynamic(
                indices, self.size, max_sizes
            )
        return indices, ignored

    @property
    def supports_fetch_outside_dataloader(self):
        """Whether this dataset supports fetching outside the workers of the dataloader."""
        return True


class FairseqIterableDataset(torch.utils.data.IterableDataset, EpochListening):
    """
    For datasets that need to be read sequentially, usually because the data is
    being streamed or otherwise can't be manipulated on a single machine.
    """

    def __iter__(self):
        raise NotImplementedError
