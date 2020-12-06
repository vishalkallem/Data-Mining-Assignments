import sys
import numpy as np


def create_adjacency_matrix():
    with open(sys.argv[1], "rt", encoding='utf-8') as file:
        data = [edge.strip() for edge in file.readlines()]
        edges = [list(map(int, d.split())) for d in data]

        size = max([max(e) for e in edges]) + 1
        A = np.zeros((size, size))

        for src, dest in edges:
            if src != dest:
                A[dest][src] += 1
    return A


def page_rank(G, d=0.8, tol=1e-2, max_iter=100, log=False):
    matrix = G
    out_degree = matrix.sum(axis=0)
    N = len(matrix[0])
    weight = np.divide(matrix, out_degree, out=np.ones_like(matrix) / N, where=out_degree != 0)
    pr = np.ones(N).reshape(N, 1) * 1. / N

    for it in range(max_iter):
        old_pr = pr[:]
        pr = d * weight.dot(pr) + (1 - d) / N
        if log:
            print(f'old_pr: {np.asarray(old_pr).squeeze()}, pr: {np.array(pr).squeeze()}')
        err = np.absolute(pr - old_pr).sum()
        if err < tol:
            return pr

    return pr


def main():
    adj_matrix = create_adjacency_matrix()
    pr = page_rank(adj_matrix)
    most_important_nodes = sorted(enumerate(np.asarray(pr).squeeze()), reverse=True, key=lambda x: x[1])[:20]
    with open(sys.argv[2], mode='wt', encoding='utf-8') as file:
        for node_index, score in most_important_nodes:
            file.write(f'{node_index}\n')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Incorrect number of command line arguments passed")
        print("Usage: python firstname_lastname_task6.py <edge_filename> <output_filename>")
        exit(1)
    main()
