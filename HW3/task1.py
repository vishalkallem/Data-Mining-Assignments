import json
import re
import pandas as pd
import networkx as nx
import sys
 

def create_network(tweets_data,gexf_output_file_name,json_output_file_name):
  rows_list = []
  for tweet in tweets_data:
      tweeting_author = ""
      retweeted_author = ""
      if('retweeted_status' in tweet):
          tweeting_author = tweet['user']['screen_name']
          retweeted_author = tweet['retweeted_status']['user']['screen_name']
      else:
          tweeting_author = tweet['user']['screen_name']
      tweet_dict = {}
      tweet_dict.update({'t_author': tweeting_author, 'rt_author': retweeted_author})
      rows_list.append(tweet_dict)
    

  tweets = pd.DataFrame(rows_list)

  G = nx.DiGraph()

  for index, row in tweets.iterrows():
      tweet_author = row['t_author']
      rted_author = row['rt_author']
      G.add_node(tweet_author)

      if (rted_author != ""):
        G.add_node(rted_author)
        if (G.has_edge(tweet_author,rted_author)):
          G[tweet_author][rted_author]['weight'] += 1
        else:
          G.add_weighted_edges_from([(tweet_author,rted_author , 1.0)])

  nx.write_gexf(G, gexf_output_file_name)
  n_nodes =G.number_of_nodes()
  n_edges = G.number_of_edges()
  out_degree_list = G.out_degree(weight='weight')
  in_degree_list =  G.in_degree(weight='weight')

  max_retweeted_user=max(in_degree_list, key=lambda x:x[1])[0]
  max_retweeted_number=max(in_degree_list, key=lambda x:x[1])[1]


  max_retweeter_user=max(out_degree_list, key=lambda x:x[1])[0]
  max_retweeter_number=max(out_degree_list, key=lambda x:x[1])[1]

  output_json = {"n_nodes":n_nodes,"n_edges":n_edges,"max_retweeted_user":max_retweeted_user,"max_retweeted_number":max_retweeted_number,"max_retweeter_user":max_retweeter_user,"max_retweeter_number":max_retweeter_number}

  with open(json_output_file_name,'w') as outfile:
    json.dump(output_json,outfile)


if __name__ == "__main__":
  inputfile = sys.argv[1]
  gexf_output_file_name = sys.argv[2]
  json_output_file_name = sys.argv[3]
  tweets_data = []
  tweets_file = open(inputfile, "r")
  for line in tweets_file:
    tweet = json.loads(line)
    tweets_data.append(tweet)
  create_network(tweets_data,gexf_output_file_name,json_output_file_name)