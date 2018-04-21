from TwitterSearch import *

try:
    # create a TwitterUserOrder for user named 'NeinQuarterly'
    tuo = TwitterUserOrder('BTC') # is equal to TwitterUserOrder(458966079)

    # it's about time to create TwitterSearch object again
    ts = TwitterSearch(
        consumer_key = '4qYbFPBgx25h4clyn6huZ8lbR',
        consumer_secret = 'O0FrTA4s0UJMx7VDj2KyaxolLX6ywvdYAbPKTgZVKcYzsfSC5I',
        access_token = '987714428098482179-aRHKyqZw6YtymDVE1JQXQQqlOrw6Gm2',
        access_token_secret = 'licSyFy2CbTfwu3T536cXuNY9xkjcfN0xYSIxzNcmuoRr'
    )

    # start asking Twitter about the timeline
    for tweet in ts.search_tweets_iterable(tuo):
        print('@%s tweeted: %s' % (tweet['user']['screen_name'], tweet['text']))

except TwitterSearchException as e: # catch all those ugly errors
    print(e)