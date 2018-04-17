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


def run_model(model_name='UserCF', clean=False):
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
    model_manager.clean_workspace(clean)
    if model_name == 'UserCF':
        usercf = UserBasedCF()
        usercf.fit(train)
        recommend_test(usercf, [1, 100, 233, 666, 888])
        usercf.test(test)
    elif model_name == 'ItemCF':
        itemcf = ItemBasedCF()
        itemcf.fit(train)
        recommend_test(itemcf, [1, 100, 233, 666, 888])
        itemcf.test(test)
    else:
        raise ValueError('No model named' + model_name)


def recommend_test(model, user_list):
    for user in user_list:
        recommend = model.recommend(str(user))
        print("recommend for userid = %s:" % user)
        print(recommend)
        print()


if __name__ == '__main__':
    main_time = LogTime(words="Main Function")
    dataset_name = 'ml-100k'
    model_type = 'UserCF'
    # model_type = 'ItemCF'
    run_model(model_type, False)
    main_time.finish()
