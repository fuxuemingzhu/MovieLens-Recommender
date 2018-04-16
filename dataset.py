# -*- coding = utf-8 -*-
"""
The :mod:`dataset` module defines the :class:`Dataset` class
and other subclasses which are used for managing datasets.

Created on 2018-04-15

@author: fuxuemingzhu
"""
import collections
import os
import itertools
import random

ml_1m = {
    'url'          : 'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
    'path'         : 'ml-1m/ratings.dat',
    'reader_params': dict(line_format='user item rating timestamp',
                          rating_scale=(1, 5),
                          sep='::')
}

# modify the random seed will change dataset spilt.
# if you want to use the model saved before, please don't modify this seed.
random.seed(0)


class DataSet:
    """Base class for loading datasets.

    Note that you should never instantiate the :class:`Dataset` class directly
    (same goes for its derived classes), but instead use one of the below
    available methods for loading datasets."""

    def __init__(self):
        pass

    @classmethod
    def load_ml_1m(cls):
        """Load ml-1m dataset.

        :return: ratings for each line.
        """
        if not os.path.isfile(ml_1m['path']):
            raise OSError("Dataset ml-1m could not be found in this project. Please download it from " + ml_1m[
                'url'] + ' manually and unzip it to this directory.')
        with open(ml_1m['path']) as f:
            ratings = [cls.parse_line(line) for line in itertools.islice(f, 0, None)]
        print("Load ml-1m dataset success.")
        return ratings

    @classmethod
    def parse_line(cls, line: str):
        """
        Parse a line.

        Ratings as ensured to positive integers.

        the separator in rating.data is `::`.

        :param line: The line to parse

        :return: tuple: User id, item id, rating score.
                The timestamp will be ignored cause it wasn't used in Collaborative filtering.
        """
        user, movie, rate, _ = line.strip('\r\n').split("::")
        return user, movie, rate

    @classmethod
    def train_test_split(cls, ratings, test_size=0.2):
        """
        Split rating data to training set and test set.

        The default `test_size` is the test percentage of test size.

        The rating file should be a instance of DataSet.

        :param test_size: the percentage of test size.
        :return: train_set and test_set
        """
        train, test = collections.defaultdict(dict), collections.defaultdict(dict)
        trainset_len = 0
        testset_len = 0
        for user, movie, rate in ratings:
            if random.random() < test_size:
                test[user][movie] = int(rate)
                testset_len += 1
            else:
                train[user][movie] = int(rate)
                trainset_len += 1
        print('split rating data to training set and test set success.')
        print('train set size = %s' % trainset_len)
        print('test set size = %s' % testset_len)
        return train, test
