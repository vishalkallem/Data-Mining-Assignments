import sys
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def generate_adjacent_matrix():
    with open(sys.argv[1], "rt", encoding='utf-8') as file:
        data = [edge.strip() for edge in file.readlines()]
        edges = [list(map(int, d.split())) for d in data]

        size = max([max(e) for e in edges]) + 1
        A = np.zeros((size, size))

        for src, dest in edges:
            if src != dest:
                A[src][dest] = 1
                A[dest][src] = 1

    return A


def dump_output(nodes, labels):
    with open(sys.argv[4], "wt", encoding='utf-8') as file:
        for node, label in zip(nodes, labels):
            file.write(f'{node} {label}\n')


def main():
    adj_matrix = generate_adjacent_matrix()
    diag_matrix = np.diag(adj_matrix.sum(axis=1))
    laplacian_matrix = diag_matrix - adj_matrix

    eigen_values, eigen_vectors = np.linalg.eigh(laplacian_matrix)
    eigen_vectors = eigen_vectors[:, np.argwhere(eigen_values >= 0.00001).flatten()]

    train_df = pd.read_csv(sys.argv[2], sep=' ', header=None)
    test_df = pd.read_csv(sys.argv[3], sep=' ', header=None)

    neigh = KNeighborsClassifier()
    neigh.fit(eigen_vectors[train_df.iloc[:, 0].to_numpy(), :68], train_df.iloc[:, 1].to_numpy())
    predicted_labels = neigh.predict(eigen_vectors[test_df.iloc[:, 0].to_numpy(), :68])

    dump_output(test_df.iloc[:, 0].to_numpy(), predicted_labels)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Incorrect number of command line arguments passed")
        print("Usage: python firstname_lastname_task4.py <edge_filename> <label_train_filename> <label_test_filename> "
              "<output_filename>")
        exit(1)
    main()
