import numpy as np
import pandas as pd
from tqdm import tqdm
from preprocess import normalizeTweet
from sklearn.model_selection import train_test_split


# goldfile = '/dgxhome/cra5302/data/twitter-disclosure/Gold_Standard.xlsx' 


# # ### Make Train and Test dataset
# train_df = pd.read_excel(goldfile)

# # for consistency with unlabeled data
# train_df["ID"] = train_df["TWEETID"].to_list()
# train_df["Text"] = list(np.vectorize(normalizeTweet)(train_df["Raw_Tweet"]))

# train_df, test_df = train_test_split(train_df, test_size=0.20, random_state=47)
# train_df, dev_df = train_test_split(train_df, test_size=0.20, random_state=47)

# train_df.to_csv("processed_data/train.csv")
# dev_df.to_csv("processed_data/dev.csv")
# test_df.to_csv("processed_data/test.csv")


# import pickle5 as pkl
# verified_file = 'verified_tweets/verified_from_1_21_3_10.pkl'
# with open(verified_file, 'rb') as handle:
#     df = pkl.load(handle)
#     df = df.dropna(subset=['Tweet'])
#     print (df.columns)
#     print (df.head())
#     df["ID"] = df["TweetID"].to_list()
#     df["Text"] = list(np.vectorize(normalizeTweet)(df["Tweet"]))

# df.to_csv('processed_data/unlabeled_phase1.csv')


# import pickle5 as pkl
# verified_file = 'verified_tweets/verified_from_3_11_6_30.pkl'
# with open(verified_file, 'rb') as handle:
#     df = pkl.load(handle)
#     df = df.dropna(subset=['Tweet'])
#     print (df.columns)
#     print (df.head())
#     df["ID"] = df["TweetID"].to_list()
#     df["Text"] = list(np.vectorize(normalizeTweet)(df["Tweet"]))

# df.to_csv('processed_data/unlabeled_phase2.csv')


import pickle5 as pkl
verified_file = 'verified_tweets/verified_from_7_1_8_28.pkl'
with open(verified_file, 'rb') as handle:
    df = pkl.load(handle)
    df = df.dropna(subset=['Tweet'])
    print (df.columns)
    print (df.head())
    df["ID"] = df["TweetID"].to_list()
    df["Text"] = list(np.vectorize(normalizeTweet)(df["Tweet"]))

df.to_csv('processed_data/unlabeled_phase3.csv')