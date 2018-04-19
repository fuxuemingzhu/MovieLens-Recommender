# -*- coding = utf-8 -*-
"""
Calculate user similarity matrix.

Created on 2018-04-15

@author: fuxuemingzhu
"""
import collections

import math

from collections import defaultdict

from utils import LogTime


def calculate_user_similarity(trainset, use_iif_similarity=False):
    """
    Calculate user similarity matrix by building movie-users inverse table.
    The calculating will only between users which have common items votes.

    :param use_iif_similarity:  This is based on User IIF similarity.
                                if the item is very popular, users' similarity will be lower.
    :param trainset: trainset
    :return: similarity matrix
    """
    # build inverse table for item-users
    # key=movieID, value=list of userIDs who have seen this movie
    print('building movie-users inverse table...')
    movie2users = collections.defaultdict(set)
    movie_popular = defaultdict(int)

    for user, movies in trainset.items():
        for movie in movies:
            movie2users[movie].add(user)
            movie_popular[movie] += 1
    print('building movie-users inverse table success.')

    # save the total movie number, which will be used in evaluation
    movie_count = len(movie2users)
    print('total movie number = %d' % movie_count)

    # count co-rated items between users
    print('generate user co-rated movies similarity matrix...')
    # the keys of usersim_mat are user1's id,
    # the values of usersim_mat are dicts which save {user2's id: co-occurrence times}.
    # so you can seem usersim_mat as a two-dim table.
    # TODO DO NOT USE DICT TO SAVE MATRIX, USE LIST INDEED.
    # TODO IF USE LIST, THE MATRIX WILL BE VERY SPARSE.
    usersim_mat = {}
    # record the calculate time has spent.
    movie2users_time = LogTime(print_step=1000)
    for movie, users in movie2users.items():
        for user1 in users:
            # set default similarity between user1 and other users equals zero
            usersim_mat.setdefault(user1, defaultdict(int))
            for user2 in users:
                if user1 == user2:
                    continue
                # ignore the score they voted.
                # user similarity matrix only focus on co-occurrence.
                if use_iif_similarity:
                    # if the item is very popular, users' similarity will be lower.
                    usersim_mat[user1][user2] += 1 / math.log(1 + len(users))
                else:
                    # origin method, users'similarity based on common items count.
                    usersim_mat[user1][user2] += 1
        # log steps and times.
        movie2users_time.count_time()
    print('generate user co-rated movies similarity matrix success.')
    movie2users_time.finish()

    # calculate user-user similarity matrix
    print('calculate user-user similarity matrix...')
    # record the calculate time has spent.
    usersim_mat_time = LogTime(print_step=1000)
    for user1, related_users in usersim_mat.items():
        len_user1 = len(trainset[user1])
        for user2, count in related_users.items():
            len_user2 = len(trainset[user2])
            # The similarity of user1 and user2 is len(common movies)/sqrt(len(user1 movies)* len(user2 movies)
            usersim_mat[user1][user2] = count / math.sqrt(len_user1 * len_user2)
            # log steps and times.
        usersim_mat_time.count_time()

    print('calculate user-user similarity matrix success.')
    usersim_mat_time.finish()
    return usersim_mat, movie_popular, movie_count


def calculate_item_similarity(trainset, use_iuf_similarity=False):
    """
    Calculate item similarity matrix by building movie-users inverse table.
    The calculating will only between items which are voted by common users.

    :param use_iuf_similarity:  This is based on Item IUF similarity.
                                if a person views a lot of movies, items' similarity will be lower.
    :param trainset: trainset
    :return: similarity matrix
    """
    movie_popular, movie_count = calculate_movie_popular(trainset)

    # count co-rated items between users
    print('generate items co-rated similarity matrix...')
    # the keys of item_sim_mat are movie1's id,
    # the values of item_sim_mat are dicts which save {movie2's id: co-occurrence times}.
    # so you can seem item_sim_mat as a two-dim table.
    # TODO DO NOT USE DICT TO SAVE MATRIX, USE LIST INDEED.
    # TODO IF USE LIST, THE MATRIX WILL BE VERY SPARSE.
    movie_sim_mat = {}
    # record the calculate time has spent.
    movie2users_time = LogTime(print_step=1000)
    for user, movies in trainset.items():
        for movie1 in movies:
            # set default similarity between movie1 and other users equals zero
            movie_sim_mat.setdefault(movie1, defaultdict(int))
            for movie2 in movies:
                if movie1 == movie2:
                    continue
                # ignore the score they voted.
                # item similarity matrix only focus on co-occurrence.
                if use_iuf_similarity:
                    # if a person views a lot of movies, items' similarity will be lower.
                    movie_sim_mat[movie1][movie2] += 1 / math.log(1 + len(movies))
                else:
                    # origin method, users'similarity based on common items count.
                    movie_sim_mat[movie1][movie2] += 1
        # log steps and times.
        movie2users_time.count_time()
    print('generate items co-rated similarity matrix success.')
    movie2users_time.finish()

    # calculate item-item similarity matrix
    print('calculate item-item similarity matrix...')
    # record the calculate time has spent.
    movie_sim_mat_time = LogTime(print_step=1000)
    for movie1, related_items in movie_sim_mat.items():
        len_movie1 = movie_popular[movie1]
        for movie2, count in related_items.items():
            len_user2 = movie_popular[movie2]
            # The similarity of user1 and user2 is len(common movies)/sqrt(len(user1 movies)* len(user2 movies)
            movie_sim_mat[movie1][movie2] = count / math.sqrt(len_movie1 * len_user2)
            # log steps and times.
        movie_sim_mat_time.count_time()

    print('calculate item-item similarity matrix success.')
    movie_sim_mat_time.finish()
    return movie_sim_mat, movie_popular, movie_count


def calculate_movie_popular(trainset):
    movie_popular = defaultdict(int)
    print('counting movies number and popularity...')

    for user, movies in trainset.items():
        for movie in movies:
            # count item popularity
            movie_popular[movie] += 1
    print('counting movies number and popularity success.')

    # save the total movie number, which will be used in evaluation
    movie_count = len(movie_popular)
    print('total movie number = %d' % movie_count)
    return movie_popular, movie_count
