# -*- coding = utf-8 -*-
"""
Evaluate the recommendation system.

Created on 2018-04-15

@author: fuxuemingzhu
"""
import math


def evaluate(recommend_method, n_rec_movie, trainset, testset, movie_popular, movie_count):
    """
    Evaluation recommend system.
    print result: precision, recall, coverage and popularity.

    :param recommend_method: The recommend method should only contains one param `user`.
    :param n_rec_movie: number of recommend movie
    :param trainset: train dataset
    :param testset: test dataset
    :param movie_popular: movie popularity
    :param movie_count: movie count
    :return: None
    """
    print('Evaluation start...')
    N = n_rec_movie
    # varables for precision and recall
    hit = 0
    rec_count = 0
    test_count = 0
    # varables for coverage
    all_rec_movies = set()
    # varables for popularity
    popular_sum = 0

    for i, user in enumerate(trainset):
        if i % 500 == 0:
            print('recommended for %d users' % i)
        test_movies = testset.get(user, {})
        # recommend movie to this user.
        # the recommend_method uses its trainset.
        rec_movies = recommend_method(user)
        for movie, _ in rec_movies:
            if movie in test_movies:
                hit += 1
            all_rec_movies.add(movie)
            popular_sum += math.log(1 + movie_popular[movie])
        rec_count += N
        test_count += len(test_movies)

    precision = hit / (1.0 * rec_count)
    recall = hit / (1.0 * test_count)
    coverage = len(all_rec_movies) / (1.0 * movie_count)
    popularity = popular_sum / (1.0 * rec_count)

    print('precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f' %
          (precision, recall, coverage, popularity))
