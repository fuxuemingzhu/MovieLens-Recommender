#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: fuxuemingzhu 
@site: www.fuxuemingzhu.cn

@file: random_pred.py
@time: 18-4-17 下午5:48

Description : Recommend via Random Choice.
"""
import random

import math

from collections import defaultdict

import similarity
import utils


class RandomPredict:
    """
    Recommend via Random Choice.
    Top-N recommendation.
    """

    def __init__(self, n_rec_movie=10, save_model=True):
        """
        Init RandomPredict with n_rec_movie.
        :return: None
        """
        print("RandomPredict start...\n")
        self.n_rec_movie = n_rec_movie
        self.trainset = None
        self.save_model = save_model

    def fit(self, trainset):
        """
        Fit the trainset via count movies.
        :param trainset: train dataset
        :return: None
        """
        model_manager = utils.ModelManager()
        try:
            self.movie_popular = model_manager.load_model('movie_popular')
            self.movie_count = model_manager.load_model('movie_count')
            self.trainset = model_manager.load_model('trainset')
            self.total_movies = model_manager.load_model('total_movies')
            print('RandomPredict model has saved before.\nLoad model success...\n')
        except OSError:
            print('No model saved before.\nTrain a new model...')
            self.trainset = trainset
            self.movie_popular, self.movie_count = similarity.calculate_movie_popular(trainset)
            self.total_movies = list(self.movie_popular.keys())
            print('Train a new model success.')
            if self.save_model:
                model_manager.save_model(self.movie_popular, 'movie_popular')
                model_manager.save_model(self.movie_count, 'movie_count')
                model_manager.save_model(self.total_movies, 'total_movies')
                print('The new model has saved success.\n')

    def recommend(self, user):
        """
        Random recommend N movies for the user.
        :param user: The user we recommend movies to.
        :return: the N best score movies
        """
        if not self.n_rec_movie or not self.trainset or not self.movie_popular or not self.movie_count:
            raise NotImplementedError('RandomPredict has not init or fit method has not called yet.')
        N = self.n_rec_movie
        predict_movies = list()
        watched_movies = self.trainset[user]
        # Random recommend N movies for the user.
        while len(predict_movies) < N:
            movie = random.choice(self.total_movies)
            if movie not in watched_movies:
                predict_movies.append(movie)
        return predict_movies[:N]

    def test(self, testset):
        """
        Test the recommendation system by recommending scores to all users in testset.
        :param testset: test dataset
        :return:
        """
        if not self.n_rec_movie or not self.trainset or not self.movie_popular or not self.movie_count:
            raise ValueError('UserCF has not init or fit method has not called yet.')
        self.testset = testset
        print('Test recommendation system start...')
        N = self.n_rec_movie
        #  varables for precision and recall
        hit = 0
        rec_count = 0
        test_count = 0
        # varables for coverage
        all_rec_movies = set()
        # varables for popularity
        popular_sum = 0

        # record the calculate time has spent.
        test_time = utils.LogTime(print_step=1000)
        for i, user in enumerate(self.trainset):
            test_movies = self.testset.get(user, {})
            rec_movies = self.recommend(user)  # type:list
            for movie in rec_movies:
                if movie in test_movies:
                    hit += 1
                all_rec_movies.add(movie)
                popular_sum += math.log(1 + self.movie_popular[movie])
                # log steps and times.
            rec_count += N
            test_count += len(test_movies)
            # print time per 500 times.
            test_time.count_time()
        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_movies) / (1.0 * self.movie_count)
        popularity = popular_sum / (1.0 * rec_count)

        print('Test recommendation system success.')
        test_time.finish()

        print('precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f\n' %
              (precision, recall, coverage, popularity))

    def predict(self, testset):
        """
        Recommend movies to all users in testset.
        :param testset: test dataset
        :return: `dict` : recommend list for each user.
        """
        movies_recommend = defaultdict(list)
        print('Predict scores start...')
        # record the calculate time has spent.
        predict_time = utils.LogTime(print_step=500)
        for i, user in enumerate(testset):
            rec_movies = self.recommend(user)  # type:list
            movies_recommend[user].append(rec_movies)
            # log steps and times.
            predict_time.count_time()
        print('Predict scores success.')
        predict_time.finish()
        return movies_recommend
