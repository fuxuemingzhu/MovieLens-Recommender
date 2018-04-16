# -*- coding = utf-8 -*-
"""
Main function to build recommendation systems.

Created on 2018-04-16

@author: fuxuemingzhu
"""
from UserCF import UserBasedCF
from dataset import DataSet

if __name__ == '__main__':
    ratings = DataSet.load_ml_1m()
    train, test = DataSet.train_test_split(ratings, test_size=0.3)
    usercf = UserBasedCF()
    usercf.fit(train)
    # recommend100 = usercf.recommend('100')
    # recommend88 = usercf.recommend('88')
    # recommend89 = usercf.recommend('89')
    # print("recommend for userid = 100:\n", recommend100)
    # print("recommend for userid = 88:\n", recommend88)
    # print("recommend for userid = 89:\n", recommend89)
    usercf.test(test)
