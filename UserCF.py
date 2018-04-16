# -*- coding = utf-8 -*-
"""
User-based Collaborative filtering.

Created on 2018-04-15

@author: fuxuemingzhu
"""
import collections
from operator import itemgetter

import math

from collections import defaultdict

import similarity
import utils
from utils import LogTime


class UserBasedCF():
    """
    User-based Collaborative filtering.
    Top-N recommendation.
    """

    def __init__(self, n_sim_user=20, n_rec_movie=10, save_model=True):
        """
        Init UserBasedCF with n_sim_user and n_rec_movie.
        :return: None
        """
        self.n_sim_user = n_sim_user
        self.n_rec_movie = n_rec_movie
        self.trainset = None
        self.save_model = save_model

    def fit(self, trainset):
        """
        Fit the trainset by calculate user similarity matrix.
        :param trainset: train dataset
        :return: None
        """
        try:
            print('The model has saved before.\nBegin loading model...')
            self.user_sim_mat = utils.load_model('user_sim_mat')
            self.movie_popular = utils.load_model('movie_popular')
            self.movie_count = utils.load_model('movie_count')
            self.trainset = utils.load_model('trainset')
            print('Load model success.')
        except OSError:
            print('No model saved before.\nTrain a new model...')
            self.user_sim_mat, self.movie_popular, self.movie_count = \
                similarity.calculate_user_similarity(trainset=trainset)
            self.trainset = trainset
            print('Train a new model success.')
            if self.save_model:
                utils.save_model(self.user_sim_mat, 'user_sim_mat')
                utils.save_model(self.movie_popular, 'movie_popular')
                utils.save_model(self.movie_count, 'movie_count')
                utils.save_model(self.trainset, 'trainset')
                print('The new model has saved success.')

    def recommend(self, user):
        """
        Find K similar users and recommend N movies for the user.
        :param user: The user we recommend movies to.
        :return: the N best score movies
        """
        if not self.n_rec_movie or not self.trainset or not self.movie_popular or not self.movie_count:
            raise NotImplementedError('UserCF has not init or fit method has not called yet.')
        K = self.n_sim_user
        N = self.n_rec_movie
        predict_score = collections.defaultdict(int)
        if user not in self.trainset:
            print('The user (%s) not in trainset.' % user)
            return
        # print('Recommend movies to user start...')
        watched_movies = self.trainset[user]
        # record the calculate time has spent.
        for similar_user, similarity_factor in sorted(self.user_sim_mat[user].items(),
                                                      key=itemgetter(1), reverse=True)[0:K]:
            for movie in self.trainset[similar_user]:
                if movie in watched_movies:
                    continue
                # predict the user's "interest" for each movie
                predict_score[movie] += similarity_factor
                # log steps and times.
        # print('Recommend movies to user success.')
        # return the N best score movies
        return sorted(predict_score.items(), key=itemgetter(1), reverse=True)[0:N]

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
        test_time = LogTime(print_step=1000)
        for i, user in enumerate(self.trainset):
            test_movies = self.testset.get(user, {})
            rec_movies = self.recommend(user)  # type:list
            for movie, _ in rec_movies:
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

        print('precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f' %
              (precision, recall, coverage, popularity))

    def predict(self, testset):
        """
        Predict scores of movies to all users in testset.
        :param testset: test dataset
        :return: `dict` : recommend list for each user.
        """
        movies_recommend = defaultdict(list)
        print('Predict scores start...')
        # record the calculate time has spent.
        predict_time = LogTime(print_step=500)
        for i, user in enumerate(self.trainset):
            test_movies = testset.get(user, {})
            rec_movies = self.recommend(user)  # type:list
            for movie, _ in rec_movies:
                if movie in test_movies:
                    movies_recommend[user].append(movie)
                    # log steps and times.
                    predict_time.count_time()
        print('Predict scores success.')
        predict_time.finish()
        return movies_recommend
