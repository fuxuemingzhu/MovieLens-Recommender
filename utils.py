# -*- coding = utf-8 -*-
"""
Utils in order to simplify coding.

Created on 2018-04-16

@author: fuxuemingzhu
"""
import time
import pickle

import os


class LogTime:
    def __init__(self, print_step=20000, words=''):
        self.proccess_count = 0
        self.PRINT_STEP = print_step
        # record the calculate time has spent.
        self.start_time = time.time()
        self.words = words
        self.total_time = 0.0

    def count_time(self):
        # log steps and times.
        if self.proccess_count % self.PRINT_STEP == 0:
            curr_time = time.time()
            print(self.words + ' steps(%d), %.2f seconds have spent..' % (
                self.proccess_count, curr_time - self.start_time))
        self.proccess_count += 1

    def finish(self):
        print('total %s step number is %d' % (self.words, self.get_curr_step()))
        print('total %.2f seconds have spent' % self.get_total_time())

    def get_curr_step(self):
        return self.proccess_count

    def get_total_time(self):
        return time.time() - self.start_time


def save_model(model, save_name: str):
    if 'pkl' not in save_name:
        save_name += '.pkl'
    if not os.path.exists('model'):
        os.mkdir('model')
    pickle.dump(model, open("model/%s" % save_name, "wb"))


def load_model(model_name: str):
    if 'pkl' not in model_name:
        model_name += '.pkl'
    if not os.path.exists('model/' + model_name):
        raise OSError('There is no model named %s in model/ dir' % model_name)
    return pickle.load(open("model/%s" % model_name, "rb"))
