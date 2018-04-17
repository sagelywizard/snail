"""A module for samplers for SNAIL training
"""
import itertools
import random

import torch.utils.data
import numpy as np


class KShotSampler(torch.utils.data.sampler.Sampler):
    """A sampler for K-shot training. Note that the supplied dataset object for
    the sampler must follow a few rules (see the dataset argument description).

    Arguments:
        dataset (Dataset): A dataset from which to sample. The dataset must
            have integer labels, which represents the class to which each item
            belongs. Each class must have the same number of examples. Also,
            examples from each class must be contiguous.
            `query_size + support_size` must be the number of examples per
            class.
        n (integer): the number of classes to sample per minibatch
        k (integer): the number of examples per class to return for each
            minibatch
        class_size (integer): the number of examples per class
    """
    def __init__(self, dataset, n, k, class_size=20):
        self.n = n
        self.k = k
        self.class_size = class_size
        self.dataset = dataset

    def __iter__(self):
        indexes = list(range(len(self)))
        np.random.shuffle(indexes)

        for i in range(0, len(self), self.n):
            # if there aren't enough remaining classes for a batch, stop
            if len(self) - i < self.n:
                raise StopIteration
            support = []
            classes = list(enumerate(indexes[i:i+self.n]))
            # shuffle the classes, so the query class isn't always the same
            # index
            np.random.shuffle(classes)
            # initialize first class, which will be the query class
            k_offsets = np.random.choice(
                self.class_size,
                size=self.k+1,
                replace=False
            )
            # add k samples of first class to support set
            support = [(
                self.dataset[(classes[0][1]*self.class_size)+offset][0],
                classes[0][0]
            ) for offset in k_offsets[0:-1]]
            query = (self.dataset[(classes[0][1]*self.class_size)+k_offsets[-1]][0], classes[0][0])
            # populate the support set with the remaining classes
            for cls_i, cls in classes[1:]:
                k_offsets = np.random.choice(
                    self.class_size,
                    size=self.k,
                    replace=False
                )
                for offset in k_offsets:
                    support.append((
                        self.dataset[(cls*self.class_size)+offset][0],
                        cls_i
                    ))
            # shuffle the support set, so it's not in contiguous classes
            np.random.shuffle(support)
            yield support, query
            ret = []

    def __len__(self):
        return len(self.dataset)//self.class_size


class KShotBatchToTensor(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, minibatch):
        imgs, labels = zip(*minibatch)
        return torch.stack(imgs), torch.FloatTensor(labels)

    def __repr__(self):
        return self.__class__.__name__ + '()'
