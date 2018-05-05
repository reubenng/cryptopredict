import time
import json
from pprint import pprint
from urllib.request import urlopen
import urllib.error

cnt = 0
counter = 1
url = "https://www.bitstamp.net/api/ticker"

def create_row(json_data):
	table_row = [json_data["timestamp"], json_data["high"], json_data["low"], \
				 json_data["bid"], json_data["ask"], json_data["last"], json_data["open"], \
				 json_data["vwap"], json_data["volume"]]

	table_row = [str(item) for item in table_row]

	return ",".join(table_row) + "\n"

# Write the title row
df = open('cryptodata.csv', 'w')
title_row = ["timestamp", "high", "low", "bid", "ask", "last", "open", "vwap", "volume"]
df.write(",".join(title_row) + "\n")
df.close()

# Append the data
while cnt < 604800:
	try:
		data = json.load(urlopen(url))
		df = open('cryptodata.csv', 'a')
		df.write(create_row(data))
		df.close()
		print("Data points collected: " + str(counter), end="\r", flush=True)
	except (urllib.error.URLError, KeyError) as e:
		from datetime import datetime
		print("{} Error on_data: {}".format(datetime.utcnow(), str(e)))

	time.sleep(10)
	cnt += 10
	counter += 1

print("[+] Data collection completed. " + str(counter-1) + " data points acquired")
