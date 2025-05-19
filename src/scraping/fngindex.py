import requests, csv, json, urllib
import pandas as pd
import time
from fake_useragent import UserAgent
from datetime import datetime

BASE_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/"
START_DATE = '2022-03-01'
END_DATE = '2025-06-01'
ua = UserAgent()

headers = {
   'User-Agent': ua.random,
   }

r = requests.get(BASE_URL + START_DATE, headers = headers)
data = r.json()

# # store data as json
# with open('Fear_and_Greed_Index/fngindex.json', 'w') as f:
# 	json.dump(data, f)

fng_data = pd.DataFrame({'Date': pd.date_range(start=START_DATE, end=END_DATE, freq='D')})
fng_data['Fear Greed'] = 0
fng_data['Date'] = pd.to_datetime(fng_data['Date'], format='%Y-%m-%d')

fng_data.set_index('Date', inplace=True)


for data_point in (data['fear_and_greed_historical']['data']):
	x = int(data_point['x'])
	x = datetime.fromtimestamp(x / 1000).strftime('%Y-%m-%d')
	y = int(data_point['y'])
	rating = data_point.get('rating', 'unknown')  # Default to 'unknown' if 'rating' is missing
	fng_data.at[x, 'Fear Greed'] = y
	fng_data.at[x, 'Rating'] = rating
#currently any days that do not have data points from cnn are filled with zeros, uncomment the following line to backfill
#fng_data['Fear Greed'].replace(to_replace=0, method='bfill')

fng_data.to_csv('Fear_and_Greed_Index/full_data.csv')

