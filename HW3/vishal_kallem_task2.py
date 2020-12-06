import sys
import json
import networkx as nx
from collections import defaultdict
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def process_data(filename):
    retweet_data, tweet_data = [], defaultdict(str)
    with open(filename, 'rt', encoding='utf-8') as file:
        for obj in file.readlines():
            obj = json.loads(obj)
            if 'retweeted_status' in obj:
                retweet_data.append((obj['user']['screen_name'], obj['retweeted_status']['user']['screen_name']))
                tweet_data[obj['retweeted_status']['user']['screen_name']] += obj['retweeted_status']['text'] + ' '
            else:
                retweet_data.append((obj['user']['screen_name'], None))
            tweet_data[obj['user']['screen_name']] += obj['text'] + ' '

    return retweet_data, tweet_data


def create_graph(data):
    G = nx.Graph()

    for source, target in data:
        if not target:
            G.add_node(source)
        else:
            if G.has_edge(source, target):
                G[source][target]['weight'] += 1
            else:
                G.add_weighted_edges_from([(source, target, 1)])
    return G


def calculate_params(G):
    m, degree_dict = 0.0, {}
    for _, _, d in G.edges(data=True):
        m += d['weight']

    for node in G.nodes():
        degree_dict[node] = int(G.degree(node, weight='weight'))
    return m, degree_dict


def girvan_newman(G):
    modularity, communities = 0.0, []
    number_of_components = nx.number_connected_components(G)

    m, degree_dict = calculate_params(G)

    while G.number_of_edges():
        edge_betweenness = nx.edge_betweenness_centrality(G, weight='weight', normalized=False)
        max_edge_betweenness = max(edge_betweenness.values())

        for nodes, betweenness in edge_betweenness.items():
            if betweenness == max_edge_betweenness:
                G.remove_edge(nodes[0], nodes[1])

        if nx.number_connected_components(G) != number_of_components:
            Q = calculate_modularity(G, m, degree_dict)
            if Q > modularity:
                modularity = Q
                communities = list(nx.connected_components(G))
        number_of_components = nx.number_connected_components(G)

    return modularity, sorted(map(sorted, communities), key=lambda x: (len(x), x[0]))


def calculate_modularity(G, m, degree_dict):
    modularity = 0
    for comp in nx.connected_components(G):
        comp = list(comp)
        for idx1, source in enumerate(comp):
            for target in comp[idx1+1:]:
                adj_value = G[source][target]['weight'] if G.has_edge(source, target) else 0
                modularity += adj_value - (degree_dict[source] * degree_dict[target]/(2 * m))
    return modularity/(2 * m)


def classify_communities(components, tweet_data, vectorizer):
    tweet_train = {'data': [], 'target': []}
    tweet_test = {'data': [], 'target': []}
    predicted_data = defaultdict(list)

    for label, comp in enumerate(components[-2:]):
        for node in comp:
            tweet_train['data'].append(tweet_data[node] if node in tweet_data else '')
            tweet_train['target'].append(label)
            predicted_data[label].append(node)

    X_train_counts = vectorizer.fit_transform(tweet_train['data'])

    clf = MultinomialNB().fit(X_train_counts, tweet_train['target'])

    for comp in components[:-2]:
        for node in comp:
            tweet_test['data'].append(tweet_data[node] if node in tweet_data else '')
            tweet_test['target'].append(node)

    X_new_counts = vectorizer.transform(tweet_test['data'])
    predicted = clf.predict(X_new_counts)

    for label, node in zip(predicted, tweet_test['target']):
        predicted_data[label].append(node)

    return predicted_data.values()


def dump_output(communities, filename, modularity=None):
    with open(filename, 'wt', encoding='utf-8') as file:
        if modularity:
            file.write(f'Best Modularity is: {modularity}\n')
        else:
            communities = sorted(map(sorted, communities), key=lambda x: (len(x), x[0]))
        for com in communities:
            file.write(','.join(f"'{c}'" for c in com))
            file.write('\n')


def main():
    graph_data, model_data = process_data(sys.argv[1])
    G = create_graph(graph_data)
    modularity, communities = girvan_newman(G)

    tfidf_output = classify_communities(communities, model_data, TfidfVectorizer())
    count_output = classify_communities(communities, model_data, CountVectorizer())

    dump_output(communities, sys.argv[2], modularity)
    dump_output(tfidf_output, sys.argv[3])
    dump_output(count_output, sys.argv[4])


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Incorrect number of command line arguments passed")
        print("Usage: python firstname_lastname_task2.py <input_file_name> <taskA_output_file_name> "
              "<taskB_output_file_name> <taskC_output_file_name>")
        exit(1)
    main()
