
# coding: utf-8

# In[ ]:

if(False):
    import sys

    # install tflearn
    get_ipython().system('{sys.executable} -m pip install git+https://github.com/tflearn/tflearn.git')

    get_ipython().system('{sys.executable} -m pip install -U gensim')

    get_ipython().system('{sys.executable} -m pip install -U nltk')

    get_ipython().system('{sys.executable} -m pip install -U progressbar2')

print("Hello World")


# In[ ]:

# download and uncompress file
import urllib.request
import io
import gzip

import os
import os.path

word2vec_path = "GoogleNews-vectors-negative300.bin"

if not os.path.isfile(word2vec_path):

  url = r'https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz'
  response = urllib.request.urlopen(url)
  CHUNK = 16 * 1024
  downloaded_file = "download.tmp"

  with open(downloaded_file, 'wb') as f:
      while True:
          chunk = response.read(CHUNK)
          if not chunk:
              break
          f.write(chunk)

  with open(downloaded_file, 'rb') as f:
    with gzip.GzipFile(fileobj=f, mode='rb') as decompressedFile: 
      word2vec_path_tmp = word2vec_path + ".uncompress"
      with open(word2vec_path_tmp, 'wb') as outfile:
        while True:
          chunk = decompressedFile.read(CHUNK)
          if not chunk:
              break
          outfile.write(chunk)

  os.rename(word2vec_path_tmp, word2vec_path)


# In[1]:

headers = ["time_stamp", "user_id", "tweet_id", "text", "verified",
           "follower_count", "friends_count", "listed_count",
           "favourites_count", "statuses_count","lang", "hashtag_text"]
example_csv = r"""Thu May 10 11:48:51 +0000 2018,978065146760323072,992822518766604288,"RT @murthaburke: Join Elvis &amp; Kresse Now! Sustainable Luxury Since 2005! 
https://t.co/UxeFrHedc1 
#blockchain #cryptocurrency #crypto #eth…",False,0,21,0,14,10,en, blockchain cryptocurrency crypto
Sat May 05 17:45:10 +0000 2018,898575243308015618,992822526379249664,"Counter Currents daily wallet summary report. Rec: 8.0167 BTC ~$79,379.79 USD, Spent: 0.1477 BTC ~$1,462.54, Bal: 7.8690 BTC ~$77,917.25.",False,1567,7,19,29,2566,en,
Sat May 05 17:45:10 +0000 2018,1730216172,992822526953783301,Atención comunidad. Están Suplantando a un amigo que se llama Andrés riccie y se hacen pasar por el para ofrecer bt… https://t.co/j1GrYmuBD7,False,467,441,0,58,5098,es,
Sat May 05 17:45:10 +0000 2018,1611014586,992822527587160064,“Bitcoin Price Is Going To Surpass 20K in 2018” – Financial Experts Explain Bitcoin’s Future Growth https://t.co/xXaV8YiHYN,False,11,114,0,5,5888,en,"""

example_timestamp_csv = r"""timestamp,high,low,bid,ask,last,open,vwap,volume
1525952871,9393.00000000,9179.98000000,9322.00,9337.99,9329.40,9308.49,9299.07,8678.90015492
1525952880,9393.00000000,9179.98000000,9324.74,9329.86,9337.40,9308.49,9299.07,8678.29429216
1525952890,9393.00000000,9179.98000000,9323.89,9335.18,9329.86,9308.49,9299.13,8672.77747484
1525952899,9393.00000000,9179.98000000,9322.00,9330.18,9335.30,9308.49,9299.16,8669.62596904
1525952910,9393.00000000,9179.98000000,9322.97,9334.93,9322.43,9308.49,9299.16,8669.60547173
1525952923,9393.00000000,9179.98000000,9322.10,9331.02,9330.18,9308.49,9299.16,8669.63496654
1525952932,9393.00000000,9179.98000000,9322.67,9335.31,9322.67,9308.49,9299.16,8670.63496654
1525952942,9393.00000000,9179.98000000,9323.00,9324.73,9322.16,9308.49,9299.16,8670.65124837
1525952953,9393.00000000,9179.98000000,9323.00,9333.04,9323.00,9308.49,9299.17,8669.71338433
"""

import email.utils
import csv

times_to_save = [-60, -30, -15, -7, - 4, -2, -1, -0.5, 0, +0.5, +1, +2, +4, +7, +10, +30, +60]
times_headers = ["{}min".format(time) for time in times_to_save]

def combine_csvs(price_fileobject, tweets_fileobject, output_fileobject):
    num_tweets = 0
    for line in tweets_fileobject:
        num_tweets += 1
    tweets_fileobject.seek(0)
    
    import collections
    hist = list()
    maxlen_deque = 6*10*abs(times_to_save[-1]- times_to_save[0] + 1)
    
    # last value in each deque is the one directly afer the times_to_save
    hist.append(collections.deque(maxlen=1))
    for i, time in enumerate(times_to_save[:-1]):
        hist.append(collections.deque(maxlen=maxlen_deque))
    
    tweet_reader = csv.DictReader(tweets_fileobject, fieldnames=headers)
    price_reader = csv.DictReader(price_fileobject)
    output_writer = csv.DictWriter(output_fileobject, dialect="unix", fieldnames=headers+times_headers)
    #print(headers+times_headers)
    output_writer.writeheader()

    # convert minutes into seconds since epoch
    times_to_save_sec = [time * 60 for time in times_to_save]
    
    import progressbar
    widgets=[progressbar.Bar(),
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ', progressbar.AnimatedMarker()]

    for tweet in progressbar.progressbar(tweet_reader, max_value=num_tweets):
        #import time
        #time.sleep(0.01)

        tweet_time = email.utils.mktime_tz(email.utils.parsedate_tz(tweet["time_stamp"]))
        
        i, time = len(times_to_save_sec) - 1, times_to_save_sec[-1]
        expected_time = tweet_time + time

        try:
            current_time = hist[-1][-1]["timestamp"]
        except IndexError:
            current_time = 0

        while int(current_time) < expected_time:
            #print("Current time is {} and expected is {}".format(current_time, expected_time))
            try:
                price_point = next(price_reader)
            except StopIteration:
                # price_reader out of data
                break
            hist[-1].append(price_point)
            try:
                current_time = price_point["timestamp"]
            except KeyError:
                print(price_point)
        
        #print(list(enumerate(times_to_save_sec[:-1])))
        for i, time in reversed(list(enumerate(times_to_save_sec[:-1]))):
            expected_time = tweet_time + time
            #print("{} out of {}".format(i, len(times_to_save_sec)))
            old_hist = hist[i+1]
            this_hist = hist[i]
            
            try:
                current_time = this_hist[-1]["timestamp"]
            except IndexError:
                current_time = 0
            #print("Current time is {} and expected is {}".format(current_time, expected_time))
            while int(current_time) < expected_time:
                #print("Current time is {} and expected is {}".format(current_time, expected_time))
                try:
                    #input()
                    price_point = old_hist.popleft()
                except IndexError:
                    #print("Current time is {} and expected is {}".format(current_time, expected_time))
                    break
                this_hist.append(price_point)
                current_time = price_point["timestamp"]
        
        #print([len(x) for x in hist])
        for i, time in enumerate(times_to_save):
            try:
                val = hist[i][-1]["last"]
            except IndexError:
                val = None
            tweet[times_headers[i]] = val
        
        output_writer.writerow(tweet)


if __name__ == "__main__" and False:
    #with io.StringIO(example_csv) as tweet_file:
    #    with io.StringIO(example_timestamp_csv) as price_file:
    with open('data/tweets/combined.csv', "r", newline='',) as tweet_file:
        with open('data/prices/combined.csv', "r", newline='') as price_file:
            with open('data/made.csv', "w", newline='') as output_file:
                combine_csvs(price_file, tweet_file, output_file)


# In[1]:

import nltk
nltk.download('stopwords')
nltk.download('punkt')
import nltk.corpus

import numpy as np
import string

if(True):
    import gensim.models

    word_vectors = gensim.models.KeyedVectors.load_word2vec_format(
        word2vec_path, binary=True)  # C binary format

    stop_words = set(nltk.corpus.stopwords.words('english')) and set(string.punctuation)
    cryptocurrencies = {"ethereum", "Ethereum", "ETH",
                           "Cryptocurrency", "cryptocurrency", "cryptocurrencies",
                           "Monero", "monero",
                           "Litecoin", "litecoin", "LTC",
                           "Cardano", "cardano",
                           "crypto", "Crypto",
                           "blockchain", "Blockchain"}
    other_cryptos = word_vectors["Euro"] - word_vectors["Dollar"] + word_vectors["Bitcoin"] 
    bitcoins = {"bitcoin", "BTC"}

import re
url_pattern = re.compile(r"http\S+")
user_pattern = re.compile(r"@\S+")

import html

number_set = {",", "K", "k", "M"}

vector_headers = {"word2vec": slice(0, 300, None),
                  "number": 300}

input_shape = (140, 301)
def vectorize(line):
  # delete urls like http...etc.com
  line = url_pattern.sub("", line)
  # replace &amp with & and etc.
  line = html.unescape(line)
  
  # replace @etc with user
  line = user_pattern.sub("user ", line)

  tokens = nltk.word_tokenize(line)

  #print(tokens)
  tokens[:] = [token for token in tokens if token not in stop_words]

  #print(tokens)
  # limit of 280 characters per tweet, so if 50% = spaces, then max of 140 words in a tweet
  vectorized = np.zeros(shape=input_shape)
  i = -1
  for word in tokens:
    i += 1
    try:
      vectorized[i,0:300] = word_vectors[word]
    except KeyError:
      if word in cryptocurrencies:
        vectorized[i,0:300] = other_cryptos
      elif word in bitcoins:
        vectorized[i,0:300] = word_vectors["Bitcoin"]
      else:
        number = "".join([char for char in word if char not in number_set])
        places = 1000 if "K" in word or "k" in word else 1
        places = 1e6 if "M" in word else 1
        
        try:
            vectorized[i,vector_headers["number"]] = float(number) * places
            continue
        except ValueError:
            pass

        #print("No match for word: '{}'".format(word))
        # don't increase the counter for the output vector
        i -= 1
      # else do nothing
    
  return vectorized

def make_label(line, values):
    prices = np.array([float(num) for num in line[12:12+len(times_headers)]])
    returns = 1 - prices[:]/prices[8]
    prices_up = returns > 0
    return prices_up[values]

import csv
import progressbar
import h5py

def preprocess(combined_fileobject):
    num_tweets = 0
    for line in combined_fileobject:
        num_tweets += 1
    combined_fileobject.seek(0)
    
    with h5py.File("data/database.hdf5", "w") as f:
        text = f.create_dataset("text", (num_tweets//4, input_shape[0], input_shape[1]),
                                dtype="f4", chunks=(1, input_shape[0], input_shape[1]))
        prices = f.create_dataset("prices", (num_tweets//4, len(times_headers)), dtype="f4", chunks=True)
        prices_up = f.create_dataset("prices_up", (num_tweets//4, len(times_headers)), dtype="i1", chunks=True)
        
        reader = csv.DictReader(combined_fileobject)
        i = 0
        
        widgets=[progressbar.Bar(),
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ', progressbar.AnimatedMarker()]
                       
        pos = 0

        for row in progressbar.progressbar(reader, max_value=num_tweets):
            if pos == 0 or pos == 1 or pos == 2:
                pos += 1
                continue
            else:
                pos = 0
            text[i, :, :] = vectorize(row["text"])
                                 
            
            try:
                for j, time in enumerate(times_headers):
                    prices[i, j] = float(row[time])
            except ValueError:
                # instead of filling with NaNs, we ignore
                pass
                #print("row[{}] was {}".format(time,row[time]))
                
            returns = 1 - prices[i, :]/prices[i, 8]
            prices_up[i, :] = returns > 0
            i += 1
            
        text.resize((i + 1, input_shape[0], input_shape[1]))
        for dataset in (prices, prices_up):
            dataset.resize((i + 1, len(times_headers)))
        
if __name__ == "__main__":
    with open('data/made.csv', "r", newline='') as combined_fileobject:
         preprocess(combined_fileobject)


# In[ ]:



