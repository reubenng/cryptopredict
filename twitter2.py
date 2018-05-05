from tweepy import Stream
from tweepy.streaming import StreamListener
import json
import tweepy
from tweepy import OAuthHandler
import unicodecsv as csv
import io

consumer_key = '4qYbFPBgx25h4clyn6huZ8lbR'
consumer_secret = 'O0FrTA4s0UJMx7VDj2KyaxolLX6ywvdYAbPKTgZVKcYzsfSC5I'
access_token = '987714428098482179-aRHKyqZw6YtymDVE1JQXQQqlOrw6Gm2'
access_secret = 'licSyFy2CbTfwu3T536cXuNY9xkjcfN0xYSIxzNcmuoRr'
 
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
 
api = tweepy.API(auth)

def this_is_lame(err):
    """ From https://stackoverflow.com/questions/36921759/getting-the-exception-class-name-in-python?rq=1 """
    if type(err).__module__ in ['__main__', 'builtins']:
        return "{}: {}".format(type(err).__name__, err)
    else:
        return "{}.{}: {}".format(type(err).__module__, type(err).__name__, err)

class MyListener(StreamListener):
    
    def on_data(self, data):
        resp_dict = None
        try:
            # with open('python.json', 'a') as f:
            with open('data.csv','ab') as o:
            
                # f.write(data)
                resp_dict = json.loads(data)
                # print(resp_dict)
                time_stamp = resp_dict['created_at']
                text = resp_dict['text']
                tweet_id = resp_dict['id']
                user_id = resp_dict['user']['id']
                verified = resp_dict['user']['verified']
                follower_count = resp_dict['user']['followers_count']
                friends_count = resp_dict['user']['friends_count']
                listed_count = resp_dict['user']['listed_count']
                favourites_count = resp_dict['user']['favourites_count']
                statuses_count = resp_dict['user']['statuses_count']
                lang = resp_dict['user']['lang']
                hashtag = resp_dict['entities']['hashtags']
                hashtag_text = ''
                if len(hashtag) >= 1:
                    for i in range(len(hashtag)):
                        hashtag_text = hashtag_text + ' ' + hashtag[i]['text']
                else:
                    hashtag_text = ''

                # csv_input = time_stamp + ',' + text + ',' + tweet_id + '\n'

                # write into csv
                writer = csv.writer(o)
                writer.writerow([time_stamp, user_id, tweet_id, text, verified, follower_count, friends_count, listed_count, favourites_count, statuses_count,lang,hashtag_text])
                return True

        except BaseException as e:
            if isinstance(e, KeyboardInterrupt):
                raise e
            from datetime import datetime
            with open('twitterapi.log','a') as log:
                error_msg = "{} {} on_data: {}".format(
                        datetime.utcnow(), this_is_lame(e), e)
                print(error_msg)
                log.write(error_msg)
                if isinstance(e, KeyError):
                    log.write("    json was {}".format(resp_dict))

        return True
 
    def on_error(self, status):
        print(status)
        
        return True
 
twitter_stream = Stream(auth, MyListener())
twitter_stream.filter(track=['bitcoin', 'btc'])
