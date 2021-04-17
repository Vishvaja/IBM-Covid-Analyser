#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 19:10:00 2020

@author: Vishvaja
"""

from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import tweepy
import json
import pandas as pd
import csv
import re
from textblob import TextBlob
import string
import preprocessor as p
import os
import time

consumer_key= ''
consumer_secret= ''
access_token= ''
access_token_secret= ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

try:
    api.verify_credentials()
    print("Authentication OK")
except:
    print("Error during authentication")


def scraptweets(search_words, date_since, numTweets, numRuns):
    
    # Define a for-loop to generate tweets at regular intervals
    # We cannot make large API call in one go. Hence, let's try T times
    
    # Define a pandas dataframe to store the date:
    db_tweets = pd.DataFrame(columns = ['username', 'acctdesc', 'location', 'following',
                                        'followers', 'totaltweets', 'usercreatedts', 'tweetcreatedts',
                                        'retweetcount', 'text', 'hashtags']
                                )
    program_start = time.time()
    for i in range(0, numRuns):
        # We will time how long it takes to scrape tweets for each run:
        start_run = time.time()
        
        # Collect tweets using the Cursor object
        # .Cursor() returns an object that you can iterate or loop over to access the data collected.
        # Each item in the iterator has various attributes that you can access to get information about each tweet
        tweets = tweepy.Cursor(api.search, q=search_words, lang="en", since=date_since, tweet_mode='extended').items(numTweets)
# Store these tweets into a python list
        tweet_list = [tweet for tweet in tweets]
# Obtain the following info (methods to call them out):
        # user.screen_name - twitter handle
        # user.description - description of account
        # user.location - where is he tweeting from
        # user.friends_count - no. of other users that user is following (following)
        # user.followers_count - no. of other users who are following this user (followers)
        # user.statuses_count - total tweets by user
        # user.created_at - when the user account was created
        # created_at - when the tweet was created
        # retweet_count - no. of retweets
        # (deprecated) user.favourites_count - probably total no. of tweets that is favourited by user
        # retweeted_status.full_text - full text of the tweet
        # tweet.entities['hashtags'] - hashtags in the tweet
# Begin scraping the tweets individually:
        noTweets = 0
    for tweet in tweet_list:
# Pull the values
        created_at = tweet.created_at
        username = tweet.user.name
        verified = tweet.user.verified
        lang = tweet.lang
        followers = tweet.user.followers_count
        favourites_count = tweet.user.favourites_count
        text = tweet.text
            
# Add the 11 variables to the empty list - ith_tweet:
        ith_tweet = [created_at,username, verified, lang,followers, favourites_count, text]
# Append to dataframe - db_tweets
        db_tweets.loc[len(db_tweets)] = ith_tweet
# increase counter - noTweets  
        noTweets += 1
        
        # Run ended:
        end_run = time.time()
        duration_run = round((end_run-start_run)/60, 2)
        
        print('no. of tweets scraped for run {} is {}'.format(i + 1, noTweets))
        print('time take for {} run to complete is {} mins'.format(i+1, duration_run))
        
        time.sleep(920) #15 minute sleep time
# Once all runs have completed, save them to a single csv file:
    from datetime import datetime
    
    # Obtain timestamp in a readable format
    to_csv_timestamp = datetime.today().strftime('%Y%m%d_%H%M%S')
# Define working path and filename
    path = os.getcwd()
    filename = path + '/data/' + to_csv_timestamp + 'covid_tweets.csv'
# Store dataframe in csv with creation date timestamp
    db_tweets.to_csv(filename, index = False)
    
    program_end = time.time()
    print('Scraping has completed!')
    print('Total time taken to scrap is {} minutes.'.format(round(program_end - program_start)/60, 2))



# Initialise these variables:
search_words = "coronavirus"
date_since = "2020-08-13"
numTweets = 2500
numRuns = 3
# Call the function scraptweets
scraptweets(search_words, date_since, numTweets, numRuns)
