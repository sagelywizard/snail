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
        query_size (integer): the size of the query set for each class
        support_size (integer): the size of the support set for each class
    """
    def __init__(self, dataset, n, k, query_size=10, support_size=10):
        self.n = n
        self.k = k
        self.query_size = query_size
        self.support_size = support_size 
        self.examples_per_class = query_size + support_size
        self.dataset = dataset

    def __iter__(self):
        indexes = list(range(len(self)))
        np.random.shuffle(indexes)
        ret = []

        for i in range(0, len(self), self.n):
            classes = list(enumerate(indexes[i:i+self.n]))
            label_class = random.choice(classes)
            ret = []
            for cls_i, cls in enumerate(classes):
                k_offsets = np.random.choice(
                    self.query_size,
                    size=self.k,
                    replace=False
                )
                for k_offset in k_offsets:
                    ret.append((
                        self.dataset[(cls[1]*self.examples_per_class)+k_offset][0],
                        cls_i
                    ))
            offset = self.query_size + np.random.randint(self.support_size)
            label = self.dataset[label_class[1]*self.examples_per_class+offset][0]
            np.random.shuffle(ret)
            yield ret, (label, label_class[0])
            ret = []

    def __len__(self):
        return len(self.dataset)//self.examples_per_class


class KShotBatchToTensor(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, minibatch):
        imgs, labels = zip(*minibatch)
        return torch.stack(imgs), torch.FloatTensor(labels)

    def __repr__(self):
        return self.__class__.__name__ + '()'
