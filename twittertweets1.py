import os
import csv
import tweepy as tw
import pandas as pd

consumer_key= 'hOYKG46TSV8TaTF5FovboxfpW'
consumer_secret= 'Ol1DQX2DOqJvXrRL0JlK7RoVR0fFVLf5Yfz8uu68zSC41z0kVh'
access_token= '1290894297311502337-7E5TY2a9QkRQGpey5nW6Ar27hXnL0H'
access_token_secret= '9b3FnzXc6A3QKKcjV7sd7C61xLU0OXWwFdwWQxLCI35gW'

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

try:
    api.verify_credentials()
    print("Authentication OK")
except:
    print("Error during authentication")
    
# Open/Create a file to append data
csvFile = open('ibm1.csv', 'a')
#Use csv Writer
csvWriter = csv.writer(csvFile)

for tweet in tw.Cursor(api.search,q="#coronavirus",count=50,lang=' ',
                           since="2020-08-13").items():
    print (tweet.created_at, tweet.text,tweet.user.verified,tweet.user.followers_count,tweet.user.favourites_count,tweet.user.lang,tweet.user.name,tweet.lang)
    csvWriter.writerow([tweet.created_at, tweet.text,tweet.user.verified,tweet.user.followers_count,tweet.user.favourites_count,tweet.user.name,tweet.lang])