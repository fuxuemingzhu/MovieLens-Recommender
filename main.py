# -*- coding = utf-8 -*-
"""
Main function to build recommendation systems.

Created on 2018-04-16

@author: fuxuemingzhu
"""
import utils
from ItemCF import ItemBasedCF
from UserCF import UserBasedCF
from dataset import DataSet
from utils import LogTime

if __name__ == '__main__':
    main_time = LogTime("Main Function")
    dataset_name = 'ml-100k'
    model_manager = utils.ModelManager(dataset_name)
    try:
        train = model_manager.load_model('trainset')
        test = model_manager.load_model('testset')
    except OSError:
        ratings = DataSet.load_dataset(name=dataset_name)
        train, test = DataSet.train_test_split(ratings, test_size=0.3)
        model_manager.save_model(train, 'trainset')
        model_manager.save_model(test, 'testset')
    '''Do you want to clean workspace and retrain model again?'''
    '''if you want to change test_size or retrain model, please set clean_workspace True'''
    # utils.clean_workspace(False)
    # usercf = UserBasedCF()
    # usercf.fit(train)
    # recommend100 = usercf.recommend('100')
    # recommend88 = usercf.recommend('88')
    # recommend89 = usercf.recommend('89')
    # print("recommend for userid = 100:\n", recommend100)
    # print("recommend for userid = 88:\n", recommend88)
    # print("recommend for userid = 89:\n", recommend89)
    # usercf.test(test)
    itemcf = ItemBasedCF()
    itemcf.fit(train)
    recommend100 = itemcf.recommend('100')
    recommend88 = itemcf.recommend('88')
    recommend89 = itemcf.recommend('89')
    print("recommend for userid = 100:\n", recommend100)
    print("recommend for userid = 88:\n", recommend88)
    print("recommend for userid = 89:\n", recommend89)
    itemcf.test(test)

    main_time.finish()
