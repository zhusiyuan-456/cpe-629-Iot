import tweepy
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd

import tweepy
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression

def calc_function(enter_keyword, enter_amount):
    get_tweets = tweepy.Cursor(api.search, q=enter_keyword).items(enter_amount)
    positive = 0
    negative = 0
    neutral = 0
    for each_tweet in get_tweets:
        tweet_analysis = TextBlob(each_tweet.text)
        if tweet_analysis.sentiment.polarity == 0:
            neutral += 1
        elif tweet_analysis.sentiment.polarity < 0:
            negative += 1
        elif tweet_analysis.sentiment.polarity > 0:
            positive += 1
    output = np.array([positive, negative,neutral])
    print(output)
    return output


# # establish connection with API # #
consumer_key = 'BDfOsV7Wh547JzQKlyp3v19Ny'
consumer_secret = 'pw6a5ZvaiuHS2TD2NDUkFOCajsz6E0s24AepHFvq758KvQO2Op'
token = '1195182173281763330-aISv9NPkUkFiTl0jRbOFsG2JhiTXey'
token_secret = 'WZpHcgHK01KsyiXTxjyMmXHeluZ73a0YG8jAWVeBEhrS9'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(token, token_secret)
api = tweepy.API(auth)

stock_data1 = pd.read_csv('/Users/tgbus/Desktop/GOOG.csv')
stock_data = stock_data1.iloc[:20,:]
#print(stock_data)
#print(type(stock_data))
stock_data.loc[:,'positive'] = 0
stock_data.loc[:,'negative'] = 0
stock_data.loc[:,'neutral'] = 0
# print(stock_data)

enter_keyword = input('enter keyword: ')
enter_amount = int(input('enter amount: '))
# enter_keyword = "google"
# enter_amount = 100

for i in range(0,stock_data.shape[0]):
    keyword = enter_keyword + " "+str(stock_data.iloc[i, 0])
    amount = enter_amount
    stock_data.iloc[i,[-3,-2,-1]] = calc_function(keyword,amount)
print(stock_data.head())

names = stock_data.iloc[:,0]
x = range(len(names))
y_stockprice = stock_data.iloc[:,-4]
y_positive = stock_data.iloc[:,-3]
y_negative = stock_data.iloc[:,-2]
y_neutral = stock_data.iloc[:,-1]
plt.plot(x, y_stockprice, marker='o', mec='r', mfc='w', label='stockprice')
plt.plot(x, y_positive, marker='*', ms=10, label='positive')
plt.plot(x, y_negative, marker='o', mec='r', mfc='w', label='negative')
plt.plot(x, y_neutral, marker='*', ms=10, label='neutral')
plt.legend()
plt.xticks(x, names, rotation=45)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel("Date")
plt.ylabel("Number")
plt.title("price with positive,negative and neutral")

plt.show()




# ridge regression

x_data=stock_data[['positive','negative','neutral']]
y_data=stock_data['Close']
ridge=Ridge().fit(x_data,y_data)
print("Training set score:{}".format(ridge.score(x_data,y_data)))
print("ridge.coef_: {}".format(ridge.coef_))
print("ridge.intercept_: {}".format(ridge.intercept_))


#linear regression


x_data=stock_data[['positive','negative','neutral']]
y_data=stock_data['Close']
lr=LinearRegression().fit(x_data,y_data)
#print regression coefficient
print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))


