import sys
import json
import numpy as np
from sklearn.neighbors import kneighbors_graph
from matplotlib import pyplot as plt


def get_json_data():
    with open(sys.argv[1], 'rt', encoding='utf-8') as file:
        return json.load(file)


def main():
    data = get_json_data()
    adj_matrix = kneighbors_graph(data, n_neighbors=8).toarray()
    adj_matrix = adj_matrix + adj_matrix.T - np.diag(adj_matrix.diagonal())
    diag_matrix = np.diag(adj_matrix.sum(axis=1))
    laplacian_matrix = diag_matrix - adj_matrix

    eigen_values, eigen_vectors = np.linalg.eig(laplacian_matrix)
    eigen_vectors = eigen_vectors[:, np.argsort(eigen_values)].real
    clusters = eigen_vectors[:, 1] > eigen_vectors.mean()
    x = [d for d, _ in data]
    y = [d for _, d in data]

    colors = ['purple' if c else 'yellow' for c in clusters]
    plt.scatter(x, y, c=colors)
    plt.savefig(sys.argv[2])


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Incorrect number of command line arguments passed")
        print("Usage: python firstname_lastname_task5.py <data_filename> <output_filename>")
        exit(1)
    main()
