import sys
import json
from pyspark import SparkContext


def collect_output_data(sc):
    data = sc.textFile(sys.argv[1]).map(lambda row: json.loads(row)).map(lambda row: row['retweet_count'])
    output_json['mean_retweet'] = data.mean()
    output_json['max_retweet'] = data.max()
    output_json['stdev_retweet'] = data.stdev()


def dump_output_data():
    with open(sys.argv[2], 'wt', encoding='utf-8') as file:
        json.dump(output_json, file, indent=1, ensure_ascii=False)


def main():
    sc = SparkContext("local[*]", "DS 553 HW1")
    sc.setLogLevel('OFF')  # Limits the output to be printed

    # start = time()  # Starts the timer to see the response time
    collect_output_data(sc)
    # print(f'Elapsed Time: {time() - start}')  # Prints the time taken to finish task 1

    dump_output_data()

    sc.stop()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Incorrect number of command line arguments passed")
        print("Usage: python firstname_lastname_task2.py input_file_path output_file_path")
        exit(1)
    output_json = {}
    main()
