from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import datetime
import csv
import sqlite3

count=0

conn = sqlite3.connect("tweets2.db")
c = conn.cursor()

def create_table():
    conn.execute("CREATE TABLE TWEETS(created_at  VARCHAR(20) NOT NULL, name VARCHAR(20)  NOT NULL,text  VARCHAR(20)  NOT NULL,verified VARCHAR(5)    NOT NULL,followers_count  INT      NOT NULL,favourites_count  INT      NOT NULL,lang  VARCHAR(10)      NOT NULL);")
    conn.commit()

create_table()


class StdOutListener(StreamListener):
 def on_status(self, tweet):
# Altering tweet text so that it keeps to one line
     text_for_output = "'" + tweet.text.replace('\n', ' ') +"'"
     c.execute("INSERT INTO TWEETS ('created_at','name','text','verified','followers_count', 'favourites_count','lang') VALUES (?,?,?,?,?,?,?)", (tweet.created_at,tweet.user.name,tweet.text,tweet.user.verified,tweet.user.followers_count,tweet.user.favourites_count,tweet.lang))
     conn.commit()
     count=conn.execute("SELECT COUNT(*) FROM TWEETS ")
     print(count)
     return True
 def on_error(self, status_code):
    if status_code == 420:
    # Returning False in on_error disconnects the stream
        return False
    
#conn.execute("DROP TABLE TWEETS")
    

consumer_key= 'hOYKG46TSV8TaTF5FovboxfpW'
consumer_secret= 'Ol1DQX2DOqJvXrRL0JlK7RoVR0fFVLf5Yfz8uu68zSC41z0kVh'
access_token= '1290894297311502337-7E5TY2a9QkRQGpey5nW6Ar27hXnL0H'
access_token_secret= '9b3FnzXc6A3QKKcjV7sd7C61xLU0OXWwFdwWQxLCI35gW'



l = StdOutListener()
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
stream = Stream(auth, l)

stream.filter(track=['coronavirus'])

