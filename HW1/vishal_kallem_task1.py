import sys
import json
from pyspark import SparkContext


def collect_output_data(sc):
    data = sc.textFile(sys.argv[1]).map(lambda row: json.loads(row))
    output_json['n_tweet'] = data.count()

    assignment_data = data.map(lambda row: (row['retweet_count'], row['user']['id'], row['user']['screen_name'],
                                            row['user']['followers_count'], row['created_at'][:3])).cache()

    output_json['n_user'] = assignment_data.map(lambda row: (row[1], 1)).reduceByKey(lambda a, b: a + b).count()

    output_json['popular_users'] = assignment_data.map(lambda row: (row[2], row[3])).\
        sortBy(lambda x: x[1], False).take(3)

    output_json['Tuesday_Tweet'] = assignment_data.filter(lambda row: row[4] == 'Tue').count()


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
        print("Usage: python firstname_lastname_task1.py input_file_path output_file_path")
        exit(1)
    output_json = {}
    main()
