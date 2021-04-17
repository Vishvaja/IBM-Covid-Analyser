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
    

consumer_key= ''
consumer_secret= ''
access_token= ''
access_token_secret= ''



l = StdOutListener()
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
stream = Stream(auth, l)

stream.filter(track=['coronavirus'])

