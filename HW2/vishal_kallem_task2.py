import sys
import numpy as np
import pandas as pd
import scipy.sparse

from collections import defaultdict


def tokenize(movies):
    token_list = []
    for _, row in movies.iterrows():
        token_list.append(tokenize_string(row.genres))
    movies['tokens'] = token_list
    return movies


def tokenize_string(genre):
    return genre.split('|')


def get_shingles(movies_dataset):
    movies = pd.read_csv(movies_dataset, encoding='utf-8')
    movies_tokenized = tokenize(movies)

    vocab_set = set(i for s in movies_tokenized['tokens'].tolist() for i in s)
    vocab = {i: x for x, i in enumerate(sorted(list(vocab_set)))}

    db = defaultdict(int)
    shingles = []
    for _, row in movies_tokenized.iterrows():
        shingle = set()
        for genre in row.tokens:
            shingle.add(vocab[genre])
        if row.movieId not in db:
            db[int(row.movieId)] = len(db)
        shingles.append(shingle)
    return shingles, db


def get_hash_coefficients(size):
    random_number = np.random.choice(2 ** 10, (2, size), replace=False)
    c = 1048583
    return random_number[0], random_number[1], c


def min_hashing(shingles, hash_coefficients, size):
    count = len(shingles)

    (a, b, c) = hash_coefficients
    a = a.reshape(1, -1)
    M = np.zeros((size, count), dtype=int)
    for i, s in enumerate(shingles):
        row_idx = np.asarray(list(s)).reshape(-1, 1)
        try:
            m = (np.matmul(row_idx, a) + b) % c
            m_min = np.min(m, axis=0)
            M[:, i] = m_min
        except ValueError:
            pass

    return M


def LSH(M, b, r, band_hash_size):
    count = M.shape[1]

    bucket_list = []
    for band_index in range(b):
        row_idx = []
        col_idx = []

        row_start = band_index * r
        for c in range(count):
            v = M[row_start:(row_start + r), c]
            v_hash = hash(tuple(v.tolist())) % band_hash_size
            row_idx.append(v_hash)
            col_idx.append(c)

        data_ary = [True] * len(row_idx)

        m = scipy.sparse.csr_matrix((data_ary, (row_idx, col_idx)), shape=(band_hash_size, count), dtype=bool)
        bucket_list.append(m)

    return bucket_list


def get_user_rated_movies(train_dataset): # You have to store the information of user: movies
    ratings = pd.read_csv(train_dataset, encoding='utf-8')

    user_ratings = defaultdict(list)
    user_rated_movies = defaultdict(list)

    for _, row in ratings.iterrows():
        user_rated_movies[int(row.userId)].append(int(row.movieId))
        user_ratings[row.userId].append((int(row.movieId), row.rating))

    return user_rated_movies, user_ratings


def get_similar_list(shingles, movie_id, threshold, bucket_list, M, b, r, band_hash_size, user_rated_movies, db):
    candidates = set()
    for band_index in range(b):
        row_start = band_index * r
        v = M[row_start:(row_start + r), movie_id]
        v_hash = hash(tuple(v.tolist())) % band_hash_size

        m = bucket_list[band_index]
        bucket = m[v_hash].indices
        candidates = candidates.union(bucket)

    sims = []
    query_set = shingles[movie_id]
    for col_idx in user_rated_movies:
        col_set = shingles[db[col_idx]]

        sim = len(query_set & col_set) / len(query_set | col_set)
        if sim >= threshold:
            sims.append((col_idx, sim))

    sims = sorted(sims, key=lambda x: x[1], reverse=True)
    return sims


def predict_rating(similar_movies, user_ratings):
    movie_ratings = {m: r for m, r in user_ratings}
    avg_user_rating = sum(movie_ratings.values()) / len(user_ratings)

    for m, r in movie_ratings.items():
        movie_ratings[m] = r - avg_user_rating

    rating, denominator = 0.0, 0.0

    for m_id, jac_sim in similar_movies:
        if m_id in movie_ratings.keys():
            rating += (movie_ratings[m_id] * jac_sim)
            denominator += jac_sim

    rating = rating / denominator if denominator else 0

    return rating + avg_user_rating


def get_predicted_rating(test_dataset, shingles, threshold, bucket_list, M, b, r, band_hash_size, db,
                         user_rated_movies, user_ratings):
    test_data = pd.read_csv(test_dataset, encoding='utf-8')
    predicted_ratings = []
    for _, row in test_data.iterrows():
        sim = get_similar_list(shingles, db[row.movieId], threshold, bucket_list, M, b, r, band_hash_size,
                               user_rated_movies[row.userId], db)
        predicted_ratings.append(predict_rating(sim, user_ratings[row.userId]))
    test_data['rating'] = predicted_ratings
    return test_data


def main():
    movies_filename, rating_train_filename, ratings_test_filename, output_filename = sys.argv[1:]
    threshold, band_hash_size, bands, rows = 0, 2 ** 16, 20, 3
    size = bands * rows

    shingles, db = get_shingles(movies_filename)

    hash_coefficients = get_hash_coefficients(size)

    min_hash = min_hashing(shingles, hash_coefficients, size)

    bucket_list = LSH(min_hash, bands, rows, band_hash_size)

    user_rated_movies, user_ratings = get_user_rated_movies(rating_train_filename)

    test_data = get_predicted_rating(ratings_test_filename, shingles, threshold, bucket_list, min_hash, bands,
                                             rows, band_hash_size, db, user_rated_movies, user_ratings)

    test_data.to_csv(output_filename, encoding='utf-8', index=False)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Incorrect number of command line arguments passed")
        print("Usage: python firstname_lastname_task2.py <movies_filename> <rating_train_filename> "
              "<rating_test_filename> <output_filename>")
        exit(1)
    main()
