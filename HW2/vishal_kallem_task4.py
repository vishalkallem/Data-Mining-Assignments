import sys
import numpy as np
import pandas as pd

from collections import defaultdict


def get_db(movies_dataset):
    users_db, movies_db = defaultdict(int), defaultdict(int)
    movies = pd.read_csv(movies_dataset, encoding='utf-8')
    for _, row in movies.iterrows():
        if int(row.userId) not in users_db:
            users_db[int(row.userId)] = len(users_db)
        if int(row.movieId) not in movies_db:
            movies_db[int(row.movieId)] = len(movies_db)
    return users_db, movies_db


def generate_matrix(train_dataset, user_db, movie_db):
    data = pd.read_csv(train_dataset, encoding='utf-8')
    matrix = np.zeros((len(user_db), len(movie_db)))
    for _, row in data.iterrows():
        matrix[user_db[int(row.userId)]][movie_db[int(row.movieId)]] = float(row.rating)
    return matrix


def predict_rating(bias, user_bias, movie_bias, left, right, row, col):
    return left[row, :].dot(right[col, :].T) + bias + user_bias[row] + movie_bias[col]


def update_bias(term, alpha, error, beta):
    return alpha * (error - beta * term)


def update_factor(term1, term2, alpha, error, beta):
    return alpha * (error * term2 - (beta * term1))


def train_model(matrix, users_db, movies_db, alpha, factors, beta, k):
    bias = matrix[np.where(matrix != 0)].mean()
    user_bias = np.zeros(len(users_db))
    movies_bias = np.zeros(len(movies_db))
    left = np.random.normal(size=(len(users_db), factors), scale=float(1/factors))
    right = np.random.normal(size=(len(movies_db), factors), scale=float(1/factors))

    for loop in range(k):
        print(f'Executing loop {loop+1}/{k}')
        for row in users_db.values():
            for col in movies_db.values():
                actual_rating = matrix[row][col]
                if actual_rating > 0:
                    error = actual_rating - predict_rating(bias, user_bias, movies_bias, left, right, row, col)
                    user_bias[row] += update_bias(user_bias[row], alpha, error, beta)
                    movies_bias[col] += update_bias(movies_bias[col], alpha, error, beta)

                    left[row, :] += update_factor(left[row, :], right[col, :], alpha, error, beta)
                    right[col, :] += update_factor(right[col, :], left[row, :], alpha, error, beta)
    return [bias, user_bias, movies_bias, left, right]


def test_model_data(test_dataset, matrix, users_db, movies_db):
    test_data = pd.read_csv(test_dataset, encoding='utf-8')
    predicted_ratings = []
    for _, row in test_data.iterrows():
        if int(row.movieId) not in movies_db:
            predicted_ratings.append(matrix[users_db[int(row.userId)]].mean())
        else:
            predicted_ratings.append(matrix[users_db[int(row.userId)]][movies_db[int(row.movieId)]])
    test_data['rating'] = predicted_ratings
    return test_data


def predict_model(parameters, users_db, movies_db):
    bias, user_bias, movies_bias, left, right = parameters
    predicted_matrix = np.zeros((len(users_db), len(movies_db)))
    for row in users_db.values():
        for col in movies_db.values():
            predicted_matrix[row][col] = predict_rating(bias, user_bias, movies_bias, left, right, row, col)
    return predicted_matrix


def main():
    alpha, factors, beta, k = 0.01, 80, 0.02, 30

    movies_filename, rating_train_filename, ratings_test_filename, output_filename = sys.argv[1:]
    users_db, movies_db = get_db(rating_train_filename)
    matrix = generate_matrix(rating_train_filename, users_db, movies_db)
    parameters = train_model(matrix, users_db, movies_db, alpha, factors, beta, k)
    predicted_matrix = predict_model(parameters, users_db, movies_db)
    test_data = test_model_data(ratings_test_filename, predicted_matrix, users_db, movies_db)
    test_data.to_csv(output_filename, encoding='utf-8', index=False)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Incorrect number of command line arguments passed")
        print("Usage: python firstname_lastname_task2.py <movies_filename> <rating_train_filename> "
              "<rating_test_filename> <output_filename>")
        exit(1)
    main()