import os
import csv
import tweepy as tw
import pandas as pd

consumer_key= ''
consumer_secret= ''
access_token= ''
access_token_secret= ''

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
