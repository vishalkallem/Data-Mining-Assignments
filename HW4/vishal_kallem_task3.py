import sys
import numpy as np
from sklearn.cluster import KMeans


def create_adjacency_matrix():
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


def dump_output(predicted_labels):
    with open(sys.argv[2], "wt", encoding='utf-8') as file:
        for idx, label in enumerate(predicted_labels):
            file.write(f'{idx} {label}\n')


def main():
    adj_matrix = create_adjacency_matrix()
    diag_matrix = np.diag(adj_matrix.sum(axis=1))
    laplacian_matrix = diag_matrix - adj_matrix

    eigen_values, eigen_vectors = np.linalg.eigh(laplacian_matrix)
    eigen_vectors = eigen_vectors[:, np.argwhere(eigen_values > 0.00001).flatten()]

    k_means = KMeans(n_clusters=int(sys.argv[3]))
    k_means.fit_predict(eigen_vectors[:, :3])
    predicted_labels = k_means.labels_
    dump_output(predicted_labels)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Incorrect number of command line arguments passed")
        print("Usage: python firstname_lastname_task3.py <edge_file_name> <output_file_name> <k>")
        exit(1)
    main()
