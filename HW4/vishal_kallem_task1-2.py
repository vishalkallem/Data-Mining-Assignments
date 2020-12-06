import numpy as np
from sklearn.cluster import KMeans


def create_adjacency_matrix():
    return np.array([
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    ])


def dump_output_task1(laplacian_matrix, param, p1, p2):
    with open('data/vishal_kallem_task1.txt', 'wt') as file:
        for row in laplacian_matrix:
            file.write(",".join([str(r) for r in row]) + "\n")
        file.write(",".join([str(p) for p in param]) + "\n")
        file.write('{' + ",".join([str(p) for p in p1]) + "}" + ",{" + ",".join([str(p) for p in p2]) + "}")


def dump_output_task2(param, cluster_centers_, param1):
    with open('data/vishal_kallem_task2.txt', 'wt') as file:
        for row in param:
            file.write(",".join([str(r) for r in row]) + "\n")
        for cc in cluster_centers_:
            file.write(",".join([str(c) for c in cc]) + '\n')
        file.write(",".join(['{' + ",".join([str(pp) for pp in p]) + '}' for p in param1]))


def main():
    adj_matrix = create_adjacency_matrix()
    diag_matrix = np.diag(adj_matrix.sum(axis=1))
    laplacian_matrix = diag_matrix - adj_matrix
    print(laplacian_matrix.shape)
    eigen_values, eigen_vectors = np.linalg.eig(laplacian_matrix)
    print(eigen_vectors.shape, eigen_values.shape)
    smallest_eigen_value_idx = np.argsort(eigen_values)[1]
    clusters = eigen_vectors[:, np.argsort(eigen_values)][:, 1] > 0
    p1 = [i for i, c in enumerate(clusters, 1) if c]
    p2 = [i for i, c in enumerate(clusters, 1) if not c]

    dump_output_task1(laplacian_matrix, eigen_vectors[:, smallest_eigen_value_idx], p1, p2)

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(eigen_vectors[:, np.argsort(eigen_values)][:, :3])
    colors = kmeans.labels_
    from collections import defaultdict
    p = defaultdict(list)
    for i, c in enumerate(colors, 1):
        p[c].append(i)

    dump_output_task2(eigen_vectors[:, np.argsort(eigen_values)][:, :3], kmeans.cluster_centers_, p.values())


if __name__ == '__main__':
    main()
