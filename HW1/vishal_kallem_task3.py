import sys
import json
from pyspark import SparkContext


def process_data(s):
    s = s.strip().split('|********************')
    s.remove('')
    for i, d in enumerate(s):
        d = d.strip().split(' ********************|')
        s[i] = "".join(d)
    return "".join(s)


def collect_output_data(sc):
    data = sc.textFile(sys.argv[1])

    word_data = data.flatMap(lambda x: x.split()).map(lambda x: (x, 1)).reduceByKey(lambda x, y: x+y)
    output_json['max_word'] = word_data.sortBy(lambda x: x[1], False).first()

    output_json['mindless_count'] = word_data.sortByKey().lookup('mindless')

    data = data.map(lambda x: process_data(x))
    output_json['chunk_count'] = data.count()


def dump_output_data():
    with open(sys.argv[2], 'wt', encoding='utf-8') as file:
        json.dump(output_json, file, indent=1, ensure_ascii=False)


def main():
    sc = SparkContext("local[*]", "DS 553 HW1")
    sc.setLogLevel('OFF')  # Limits the output to be printed

    collect_output_data(sc)

    dump_output_data()

    sc.stop()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Incorrect number of command line arguments passed")
        print("Usage: python firstname_lastname_task3.py input_file_path output_file_path")
        exit(1)
    output_json = {}
    main()
