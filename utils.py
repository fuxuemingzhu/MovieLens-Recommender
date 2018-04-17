# -*- coding = utf-8 -*-
"""
Utils in order to simplify coding.

Created on 2018-04-16

@author: fuxuemingzhu
"""
import time
import pickle

import os
import shutil


class LogTime:
    """
    Time used help.
    You can use count_time() in for-loop to count how many times have looped.
    Call finish() when your for-loop work finish.
    WARNING: Consider in multi-for-loop, call count_time() too many times will slow the speed down.
            So, use count_time() in the most outer for-loop are preferred.
    """

    def __init__(self, print_step=20000, words=''):
        """
        How many steps to print a progress log.
        :param print_step: steps to print a progress log.
        :param words: help massage
        """
        self.proccess_count = 0
        self.PRINT_STEP = print_step
        # record the calculate time has spent.
        self.start_time = time.time()
        self.words = words
        self.total_time = 0.0

    def count_time(self):
        """
        Called in for-loop.
        :return:
        """
        # log steps and times.
        if self.proccess_count % self.PRINT_STEP == 0:
            curr_time = time.time()
            print(self.words + ' steps(%d), %.2f seconds have spent..' % (
                self.proccess_count, curr_time - self.start_time))
        self.proccess_count += 1

    def finish(self):
        """
        Work finished! Congratulations!
        :return:
        """
        print('total %s step number is %d' % (self.words, self.get_curr_step()))
        print('total %.2f seconds have spent\n' % self.get_total_time())

    def get_curr_step(self):
        return self.proccess_count

    def get_total_time(self):
        return time.time() - self.start_time


class ModelManager:
    """
    Model manager is designed to load and save all models.
    No matter what dataset name.
    """
    # This dataset_name belongs to the whole class.
    # So it should be init for only once.
    path_name = ''

    @classmethod
    def __init__(cls, dataset_name=None, test_size=0.3):
        """
        cls.dataset_name should only init for only once.
        :param dataset_name:
        """
        if not cls.path_name:
            cls.path_name = "model/" + dataset_name + '-testsize' + str(test_size)

    def save_model(self, model, save_name: str):
        """
        Save model to model/ dir.
        :param model: source model
        :param save_name: model saved name.
        :return: None
        """
        if 'pkl' not in save_name:
            save_name += '.pkl'
        if not os.path.exists('model'):
            os.mkdir('model')
        pickle.dump(model, open(self.path_name + "-%s" % save_name, "wb"))

    def load_model(self, model_name: str):
        """
        Load model from model/ dir via model name.
        :param model_name:
        :return: loaded model
        """
        if 'pkl' not in model_name:
            model_name += '.pkl'
        if not os.path.exists(self.path_name + "-%s" % model_name):
            raise OSError('There is no model named %s in model/ dir' % model_name)
        return pickle.load(open(self.path_name + "-%s" % model_name, "rb"))

    @staticmethod
    def clean_workspace(clean=False):
        """
        Clean the whole workspace.
        All File in model/ dir will be removed.
        :param clean: Boolean. Clean workspace or not.
        :return: None
        """
        if clean and os.path.exists('model'):
            shutil.rmtree('model')
