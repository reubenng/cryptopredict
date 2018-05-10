import pandas as pd
import codecs
import sys
from textblob import TextBlob
import time

# import csv to df
with codecs.open('./data.csv', "r",encoding='utf-8', errors='ignore') as fdata:
    df = pd.read_csv(fdata, sep=',', names = ["TimeStamp", "UserID", "TweetID", "Content","Verified","FollowersCount","FriendsCount","ListedCount","FavouritesCount","StatusesCount","Language","Hashtags"])

# add headers to df
df["Sentiment"] = ""


# df = df.iloc[:10000]

# print(df.head())
counter = 0
for index, row in df.iterrows():
    tweet_plus_hashtag = str(row['Content']) + ' ' + str(row['Hashtags'])
    testimonial = TextBlob(tweet_plus_hashtag)
    sentiment = (float(testimonial.sentiment.polarity) + 1) / 2
    # print(tweet_plus_hashtag)
    # print(sentiment)
    df.loc[index,'Sentiment'] = sentiment
    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(df['TimeStamp'][index],'%a %b %d %H:%M:%S +0000 %Y'))
    df.loc[index,'TimeStamp'] = ts
    counter = counter + 1

    # if counter == 100:
    #     # print(df.head(n=100))
    #     # sys.exit()
    #     break

# print(df.head())

# CHANGE FILTERING THRESHOLDS HERE
df = df.drop(df[df["FollowersCount"] < 1000].index)
df = df.drop(df[df["StatusesCount"] < 300].index)
df = df.drop(df[df["Language"] != 'en'].index)


# create a new df with the data
# filter the data with thresholds

# print(df.head(n=50))


# first, convert TimeStamp to pd.date_time
df['TimeStamp']= pd.to_datetime(df['TimeStamp'])
# print(df)
df_grouped = df.TimeStamp.groupby(df.TimeStamp.dt.hour)

df_for_reub = pd.DataFrame(columns=['Timeframe','Sentiment','NumberOfTweets','AverageFavourites'])
# fill in data for reub
for name, group in df_grouped:
    # print(group)
    timeframe = name # hour
    # print(group.index.values.tolist())
    sentiment = df['Sentiment'][group.index.values].mean()
    numberOfTweets = group.size
    AverageFavourites = df['FavouritesCount'][group.index.values].mean()

    df_for_reub.loc[len(df_for_reub)] = [timeframe, sentiment, numberOfTweets, AverageFavourites] 

# print(df_for_reub.head())

# print(df)

df_for_reub.to_csv('reuben_data.csv', sep='\t')



