import tweepy
#import configparser

import pandas as pd
# import csv
# import re 
# import emoji
# import nltk
# import string
import preprocessor as p
from tensorflow import keras
 
#Note API Keys are private so I have removed mine for purposes of public archival
#In order for this project to function please insert your own keys for the following fields from the twitter developer website
API_KEY = "[Fill In Here]"
API_KEY_SECRET = "[Fill In Here]"
BEARER_TOKEN = "[Fill In Here]"
ACCESS_TOKEN = "[Fill In Here]"
ACCESS_TOKEN_SECRET = "[Fill In Here]"

client = tweepy.Client(
consumer_key= API_KEY, 
consumer_secret= API_KEY_SECRET, 
bearer_token= BEARER_TOKEN, 
access_token=ACCESS_TOKEN, 
access_token_secret=ACCESS_TOKEN_SECRET)

auth = tweepy.OAuthHandler(client.consumer_key, client.consumer_secret)
auth.set_access_token(client.access_token, client.access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)


keyword = input("Enter keyword (add '#' to search for hashtags): ")
search_word = keyword + " -rt"
num_of_results = input("How Many Results would you like?: ")

tweets = client.search_recent_tweets(query=search_word, tweet_fields=['created_at'], max_results=num_of_results)

columns = ['Time', 'Tweet']
data = []

for tweet in tweets.data:
    #insert ML algorithm here
    data.append([tweet.created_at, tweet.text])

df = pd.DataFrame(data, columns=columns)

print(df)
df.to_csv('Sentiment_Analysis_of_{}_Tweets_About_{}.csv'.format(num_of_results, keyword))