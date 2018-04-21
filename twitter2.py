from tweepy import Stream
from tweepy.streaming import StreamListener
import json
import tweepy
from tweepy import OAuthHandler
import csv
import io

consumer_key = '4qYbFPBgx25h4clyn6huZ8lbR'
consumer_secret = 'O0FrTA4s0UJMx7VDj2KyaxolLX6ywvdYAbPKTgZVKcYzsfSC5I'
access_token = '987714428098482179-aRHKyqZw6YtymDVE1JQXQQqlOrw6Gm2'
access_secret = 'licSyFy2CbTfwu3T536cXuNY9xkjcfN0xYSIxzNcmuoRr'
 
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
 
api = tweepy.API(auth) 

class MyListener(StreamListener):
    
    def on_data(self, data):
        try:
            # with open('python.json', 'a') as f:
            with open('data.csv','a') as o:
            
                # f.write(data)
                resp_dict = json.loads(data)
                time_stamp = resp_dict['created_at']
                text = resp_dict['text']
                csv_input = time_stamp + ',' + text + '\n'
                print(str(csv_input))

                # write into csv
                writer = csv.writer(o)
                writer.writerow([time_stamp, text])
                return True

        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True
 
    def on_error(self, status):
        print(status)
        
        return True
 
twitter_stream = Stream(auth, MyListener())
twitter_stream.filter(track=['bitcoin', 'btc'])