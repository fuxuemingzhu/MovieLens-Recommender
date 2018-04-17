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


def run_model(model_name='UserCF', test_size=0.3, clean=False):
    model_manager = utils.ModelManager(dataset_name, test_size)
    try:
        pre_test_size = model_manager.load_model('test_size')
        assert pre_test_size == test_size
        trainset = model_manager.load_model('trainset')
        testset = model_manager.load_model('testset')
    except OSError:
        ratings = DataSet.load_dataset(name=dataset_name)
        trainset, testset = DataSet.train_test_split(ratings, test_size=test_size)
        model_manager.save_model(trainset, 'trainset')
        model_manager.save_model(testset, 'testset')
    '''Do you want to clean workspace and retrain model again?'''
    '''if you want to change test_size or retrain model, please set clean_workspace True'''
    model_manager.clean_workspace(clean)
    if model_name == 'UserCF':
        model = UserBasedCF()
    elif model_name == 'ItemCF':
        model = ItemBasedCF()
    else:
        raise ValueError('No model named' + model_name)
    model.fit(trainset)
    recommend_test(model, [1, 100, 233, 666, 888])
    model.test(testset)


def recommend_test(model, user_list):
    for user in user_list:
        recommend = model.recommend(str(user))
        print("recommend for userid = %s:" % user)
        print(recommend)
        print()


if __name__ == '__main__':
    main_time = LogTime(words="Main Function")
    # dataset_name = 'ml-100k'
    dataset_name = 'ml-1m'
    model_type = 'UserCF'
    # model_type = 'ItemCF'
    run_model(model_type, 0.1, False)
    main_time.finish()
