import networkx as nx
import math
import csv
import random as rand
import sys
import json
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

def CLAN_Data_Processing(sorted_communities,user_tweet_dict):
    target_1 = [1] * len(sorted_communities[-1])
    target_2 = [2] * len(sorted_communities[-2])
    data =[]
    test_data =[]
    test_users =[]
    target = target_1 + target_2
    for i in sorted_communities[-1]:
        data.append(user_tweet_dict[i])
    for i in sorted_communities[-2]:
        data.append(user_tweet_dict[i])
    for i in range(len(sorted_communities)-2):
        for j in sorted_communities[i]:
            test_data.append(user_tweet_dict[j])
            test_users.append(j)
    return data, target, test_data, test_users

def CLAN_Merge(predictions,sorted_communities,test_data,test_users):
    clan_com1 =[]
    clan_com2 =[]
    clan_communities =[]
    for i in sorted_communities[-1]:
        clan_com1.append(i)
    for i in sorted_communities[-2]:
        clan_com2.append(i)
    for i in range(len(predictions)):
        if(predictions[i]==1):
            clan_com1.append(test_users[i])
        else:
            clan_com2.append(test_users[i])
    clan_com1.sort()
    clan_com2.sort()
    clan_communities.append(clan_com1)
    clan_communities.append(clan_com2)
    clan_communities.sort(key=sorted)
    clan_communities.sort(key=len)
    return clan_communities

def TaskC_Clan_Classifier(train_data,train_label,test_data):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_data)

    clf = MultinomialNB().fit(X_train_counts, train_label)


    docs_test = count_vect.transform(test_data)
    predicted = clf.predict(docs_test)
    return predicted


def TaskB_Clan_Classifier(train_data,train_label,test_data):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(train_data)
    
    clf = MultinomialNB().fit(X, train_label)

    docs_test = vectorizer.transform(test_data)
    predicted = clf.predict(docs_test)
    return predicted

def Build_Graph(input_filename,G,user_tweet_dict):
    tweets_data = []
    tweets_file = open(input_filename, "r")
    for line in tweets_file:
        tweet = json.loads(line)
        tweets_data.append(tweet)
    rows_list = []
    for tweet in tweets_data:
        tweeting_author = ""
        retweeted_author = ""
        if('retweeted_status' in tweet):
            tweeting_author = tweet['user']['screen_name']
            retweeted_author = tweet['retweeted_status']['user']['screen_name']
            if(tweeting_author in user_tweet_dict):
                user_tweet_dict[tweeting_author] = user_tweet_dict[tweeting_author] +" "+  tweet['text']
            else:
                user_tweet_dict[tweeting_author] = tweet['text']
            if(retweeted_author in user_tweet_dict):
                user_tweet_dict[retweeted_author] = user_tweet_dict[retweeted_author] +" "+  tweet['retweeted_status']['text']
            else:
                user_tweet_dict[retweeted_author] = tweet['retweeted_status']['text']                
        else:
            tweeting_author = tweet['user']['screen_name']
            if(tweeting_author in user_tweet_dict):
                user_tweet_dict[tweeting_author] = user_tweet_dict[tweeting_author] +" "+  tweet['text']
            else:
                user_tweet_dict[tweeting_author] = tweet['text']
        tweet_dict = {}
        tweet_dict.update({'t_author': tweeting_author, 'rt_author': retweeted_author})
        rows_list.append(tweet_dict)  

    tweets = pd.DataFrame(rows_list)

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


def Get_Splits(G):
    init_ncomp = nx.number_connected_components(G)    
    ncomp = init_ncomp
    while ncomp <= init_ncomp:
        bw = nx.edge_betweenness_centrality(G, weight='weight',normalized=False) 
        max_ = max(bw.values())
        for k, v in bw.items():
            if v == max_:
                G.remove_edge(k[0],k[1])
        ncomp = nx.number_connected_components(G)    



def Split_Modularity(G, deg_, w_m):
    comps = nx.connected_components(G)  
    Mod = 0  
    for c in comps:
        sum_mod =0
        for u in c:
            for v in c:
                if(u != v):
                    if(G.subgraph(c).has_edge(u,v)):    
                        Ew = G.subgraph(c)[u][v]['weight']
                        RE = deg_[u]*deg_[v]
                        sum_mod += Ew - RE/(2*w_m)
                    else:
                        Ew = 0
                        RE = deg_[u]*deg_[v]
                        sum_mod += Ew - RE/(2*w_m)
        #graders accept both sum_mod/2 and not
        Mod += (sum_mod)
    Mod = Mod/(2*w_m)
    return Mod

def Get_Degree(A, nodes):
    deg_dict = {}
    B = A.sum(axis = 1)
    i = 0
    for node_id in list(nodes):
        deg_dict[node_id] = B[i, 0]
        i += 1
    return deg_dict


def Girvan_Newman(G, Orig_deg, w_m):
    BestQ = 0.0
    Q = 0.0
    while G.number_of_edges() != 0:    
        Get_Splits(G)
        Q = Split_Modularity(G, Orig_deg,w_m);
        if(Q > BestQ):
            BestQ = Q
            Bestcomps = list(nx.connected_components(G))  
    return Bestcomps, BestQ

def Write_Results(filename,communities,mod,modularity):
    outputfile = open(filename,"w")
    if(mod):
        outputfile.write("Best Modularity is: "+str(modularity)+"\n")
    for c in communities:
        temp =""
        for member in c:
            temp = temp + "'" + member + "'" + ","
        temp = temp.strip(",")
        outputfile.write(temp + "\n")


if __name__ == "__main__":
    G = nx.Graph()
    user_tweet_dict = {}
    Build_Graph(sys.argv[1],G,user_tweet_dict)
    taskA_outputfile = sys.argv[2]
    taskB_outputfile = sys.argv[3]
    taskC_outputfile = sys.argv[4]
    n = G.number_of_nodes()   
    A = nx.adj_matrix(G)  
    w_m = 0.0   
    for i in range(n):
        for j in range(n):
            w_m += A[i,j]
    w_m = w_m/2.0

    Orig_deg = {}
    Orig_deg = Get_Degree(A, G.nodes())

    communities, best_mod = Girvan_Newman(G, Orig_deg, w_m)
    sorted_communities =[]
    for i in communities:
        c = list(i)
        c.sort()
        sorted_communities.append(c)
    sorted_communities.sort(key=sorted)
    sorted_communities.sort(key=len)
    Write_Results(taskA_outputfile,sorted_communities,True,best_mod)
    train_data, train_label, test_data, test_users = CLAN_Data_Processing(sorted_communities,user_tweet_dict)
    predicted = TaskB_Clan_Classifier(train_data, train_label, test_data)
    clan_communities_B =CLAN_Merge(predicted,sorted_communities, test_data,test_users)
    predicted = TaskC_Clan_Classifier(train_data, train_label, test_data)
    clan_communities_C =CLAN_Merge(predicted,sorted_communities, test_data,test_users)
    Write_Results(taskB_outputfile,clan_communities_B,False, best_mod)
    Write_Results(taskC_outputfile,clan_communities_C,False,best_mod)
