import sys
import json
import networkx as nx


def process_data(filename):
    retweet_data = []
    with open(filename, 'rt', encoding='utf-8') as file:
        for obj in file.readlines():
            obj = json.loads(obj)
            retweet_data.append(
                (obj['user']['screen_name'], obj['retweeted_status']['user']['screen_name'])
                if 'retweeted_status' in obj else (obj['user']['screen_name'], None)
            )
    return retweet_data


def create_graph(data, graph_filename):
    G = nx.DiGraph()
    for source, target in data:
        if not target:
            G.add_node(source)
        else:
            if G.has_edge(source, target):
                G[source][target]['weight'] += 1
            else:
                G.add_weighted_edges_from([(source, target, 1)])
    nx.write_gexf(G, graph_filename)
    return G


def collect_output_data(G):
    output_json['n_nodes'] = G.number_of_nodes()

    output_json['n_edges'] = G.number_of_edges()

    min_retweet = sorted(G.in_degree(weight='weight'), key=lambda x: x[1], reverse=True)[0]
    max_retweet = sorted(G.out_degree(weight='weight'), key=lambda x: x[1], reverse=True)[0]

    output_json['max_retweeted_user'] = min_retweet[0]

    output_json['max_retweeted_number'] = min_retweet[1]

    output_json['max_retweeter_user'] = max_retweet[0]

    output_json['max_retweeter_number'] = max_retweet[1]


def dump_output_data():
    with open(sys.argv[3], 'wt', encoding='utf-8') as file:
        json.dump(output_json, file, indent=1, ensure_ascii=False)


def main():
    data = process_data(sys.argv[1])

    G = create_graph(data, sys.argv[2])

    collect_output_data(G)

    dump_output_data()


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Incorrect number of command line arguments passed")
        print("Usage: python firstname_lastname_task1.py <input_file_name> <gexf_output_file_name> "
              "<json_output_file_name>")
        exit(1)
    output_json = {}
    main()
