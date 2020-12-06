import sys
import json
import bisect
import itertools
from operator import itemgetter
from collections import defaultdict, Counter, UserList


def process_line(line):
    tokens = line.replace(',,', ',').split(',')
    return tokens[:2] if len(tokens) == 4 and float(tokens[2]) == 5.0 else []


def get_baskets(file):
    baskets = defaultdict(list)
    with open(file, mode='rt') as in_file:
        header = True
        for line in in_file:
            if header:
                header = False
                continue
            tokens = process_line(line.strip())
            if tokens:
                user_id, movie_id = tokens
                baskets[user_id].append(movie_id)
        return list(baskets.items())


class FirstList(UserList):
    def __lt__(self, other):
        return self[0].__lt__(other)


def filter_basket(baskets, item_dict, k):
    possible_items = item_dict if k == 2 else set()

    if not possible_items:
        possible_items = possible_items.union(*item_dict.keys())

    for idx, basket in enumerate(baskets):
        baskets[idx] = (basket[0], list(filter(lambda x: True if x in possible_items else False, basket[1])))


def inverse_dict(d):
    return {v: k for k, v in d.items()}


def get_possible_k(item_dict, k):
    possible_k = {}
    for pair in itertools.combinations(item_dict.keys(), 2):
        pair_set = set()
        for i in range(2):
            pair_set = pair_set.union(tuple_wrapper(pair[i]))
        if len(pair_set) == k:
            possible_k[frozenset(pair_set)] = [pair[0], pair[1]]
    return possible_k


def tuple_wrapper(s):
    return s if type(s) is tuple else (s,)


def get_freq_item_sets(baskets, support, item_dict, k=2):
    filter_basket(baskets, item_dict, k)

    item_dict_inv = inverse_dict(item_dict)
    n, possible_k = len(item_dict), {}

    if k >= 3:
        possible_k = get_possible_k(item_dict, k)

    triples = []

    for user_id, items in baskets:
        for k_pair in itertools.combinations(items, k):
            if k >= 3:
                pair_set = frozenset(k_pair)
                k_pair = possible_k.get(pair_set, None)
                if not k_pair:
                    continue

            i = item_dict[k_pair[0]]
            j = item_dict[k_pair[1]]

            if i > j:
                j, i = i, j

            idx = i * n + j

            insert_idx = bisect.bisect_left(triples, idx)

            if insert_idx >= len(triples):
                triples.append(FirstList([idx, 1]))
            else:
                tp = triples[insert_idx]
                if tp[0] == idx:
                    tp[1] += 1
                else:
                    triples.insert(insert_idx, FirstList([idx, 1]))

    freq_item_set = []
    for idx, count in triples:
        if count >= support:
            i = idx // n
            j = idx % n

            item_i = item_dict_inv[i]
            item_j = item_dict_inv[j]

            item_all = set()
            for item in (item_i, item_j):
                item_all = item_all.union(tuple_wrapper(item))

            freq_item_set.append((tuple(item_all), count))

    freq_item_set = sorted(freq_item_set, key=lambda x: (-x[1], x[0]))
    return freq_item_set


def get_item_sets(baskets, support=10):
    item_sets = Counter(m_id for _, movies in baskets for m_id in movies)
    return {m_id: count for m_id, count in item_sets.items() if count >= support}


def get_item_dict(items):
    item_sets = {}
    for item in items:
        item_sets[item] = len(item_sets)
    return item_sets


def get_probabilities(items, T):
    probability = defaultdict(float)
    for item, count in items.items():
        probability[item] = count / T
    return probability


def apriori(baskets, item_sets_dict, support=10, k=1):
    freq_item_sets = []
    item_sets = sorted(item_sets_dict.items(), key=itemgetter(1), reverse=True)

    while item_sets:
        k += 1
        freq_items = [item[0] for item in item_sets]
        freq_item_sets.append(item_sets)

        item_dict = get_item_dict(freq_items)
        item_sets = get_freq_item_sets(baskets, support, item_dict, k=k)

    return freq_item_sets


def generate_rules(item_sets, probabilities, I=0.2):
    support_dict = {frozenset(tuple_wrapper(candidate)): support for size in item_sets for candidate, support in size}

    output = []
    possible_candidates = filter(lambda x: len(x) > 1, support_dict.keys())

    for candidate in possible_candidates:
        for i in itertools.combinations(candidate, len(candidate) - 1):
            j = next(iter(set(candidate).difference(set(i))))
            i = frozenset(i)
            if i in support_dict.keys():
                confidence = support_dict[candidate] / support_dict[i]
                interest = confidence - probabilities[j]
                support = support_dict[candidate]
                if interest >= I:
                    output.append([sorted(map(int, i)), int(j), interest, support])

    return sorted(output, key=lambda x: (-x[2], -x[3], x[0], x[1]))


def dump_output_data(output_filename, output_json):
    with open(output_filename, 'wt', encoding='utf-8') as file:
        json.dump(output_json, file, indent=1, ensure_ascii=False)


def main():
    input_filename, output_filename, I, S = sys.argv[1:]
    I, S = float(I), int(S)

    baskets = get_baskets(input_filename)

    item_sets_dict = get_item_sets(baskets, support=S)

    probabilities = get_probabilities(item_sets_dict, len(baskets))

    freq_item_sets = apriori(baskets, item_sets_dict, support=S)

    output_json = generate_rules(freq_item_sets, probabilities, I=I)

    dump_output_data(output_filename, output_json)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Incorrect number of command line arguments passed")
        print("Usage: python firstname_lastname_task1.py <input_filename> <output_filename> <I> <S>")
        exit(1)
    main()
