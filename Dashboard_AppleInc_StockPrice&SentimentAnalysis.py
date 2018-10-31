import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, Event
import plotly
import plotly.graph_objs as go
from collections import deque

import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
from time import mktime
import pandas as pd
import sqlite3

import os
import flask
import matplotlib.pyplot as plt
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# nlp par
stopwords = stopwords.words('english')#list(punctuation)) #+'’'+'‘'+

def calculate_TF(content):
    test = content.split()
    sentence = []

    for i in range(len(test)):
        global stopwords
        if test[i] not in stopwords:
            sentence.append(test[i])
    #print (sentence)
    #stemming and lemmatizing
    ps = PorterStemmer()
    stem_s = [ps.stem(st) for st in sentence]
    cleaned_contents = stem_s

    #calculate TF for the news
    count_words = len(cleaned_contents)

    freq_dict ={}
    for i in cleaned_contents:
        if i in freq_dict:
            freq_dict[i] +=1
        else:
            freq_dict[i] = 1

    TF_scores={}
    for tempDict in freq_dict:
        score = freq_dict[tempDict]/count_words
        TF_scores[tempDict]=score
    #print('TF score is :', TF_scores['appl'])
    return TF_scores['appl']


#we use api to get apple's  news from techcrunch website

url = ('https://newsapi.org/v2/everything?'
       'q=Apple&'
       'sources=techcrunch&'
       'from=2018-10-20&'
        'apiKey=91e82f8ffec54728b84ae822b453ed8e'
        )
response = requests.get(url)
js=response.json()
js

#return a list of url
article = js['articles']
url_list = []
for i in range(len(article)):
    #print (article[i]['url'])
    url_list.append(article[i]['url'])

#to extract one text from url
text_list=[]
Dic = {}
for i in url_list:

    r=requests.get(i).text
    soup = BeautifulSoup(r,'lxml')
    soup.find('p').getText()
    soup.getText()
    p_tags=soup.find_all('p')
    for p in p_tags[:]:
        text_list.append(p.text)

    content = ''.join(text_list)
    #print (i,calculate_TF(content))
    Dic[i] = calculate_TF(content)


s = [(k, Dic[k]) for k in sorted(Dic, key=Dic.get, reverse=True)]
max_TF=s[0]
# print ("max TF is ", s[0])

#wordclound
text_list=[]
r=requests.get(max_TF[0]).text
soup = BeautifulSoup(r,'lxml')
soup.find('p').getText()
soup.getText()
p_tags=soup.find_all('p')
for p in p_tags[:]:
    text_list.append(p.text)

content = ''.join(text_list)
text = content

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text)
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)

# Display the generated image:
plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('word_cloud_test.png')

list_of_images=os.listdir()
for each in list_of_images[:]: # filelist[:] makes a copy of filelist.
    if not(each.endswith(".png")):
        list_of_images.remove(each)

static_image_route = '/static/'

# crawl stock price from Yahoo finance
header = {'Connection': 'keep-alive',
           'Expires': '-1',
           'Upgrade-Insecure-Requests': '1',
           'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) \
           AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36'
           }

# return (header, crumb[0], response)

def get_YahooFinance(stock):
    url = 'https://finance.yahoo.com/quote/{}/history'.format(stock)
    with requests.session():
        response = requests.get(url, headers=header)

    return response

def load_historical_data(stock, interval='1d', day_begin='2015-01-01', day_end='2018-10-24'):
    day_begin_stamp = int(mktime(datetime.strptime(day_begin, '%Y-%m-%d').timetuple()))
    day_end_stamp = int(mktime(datetime.strptime(day_end, '%Y-%m-%d').timetuple()))

    response = get_YahooFinance(stock)
    soup = BeautifulSoup(response.text, 'lxml')
    crumb = re.findall('"CrumbStore":{"crumb":"(.+?)"}', str(soup))

    with requests.session():
        url = 'https://query1.finance.yahoo.com/v7/finance/download/' \
              '{stock}?period1={day_begin}&period2={day_end}&interval={interval}&events=history&crumb={crumb}' \
              .format(stock=stock, day_begin=day_begin_stamp, day_end=day_end_stamp, interval=interval, crumb=crumb[0])

        website = requests.get(url, headers=header, cookies=response.cookies)

    conn = sqlite3.connect('stock_temp.db')
    c = conn.cursor()
    try:
        c.execute('DROP TABLE stock')
    except:
        pass
    c.execute('CREATE TABLE stock(Date TEXT, Open REAL, High REAL, Low REAL, Close REAL, Adj_Close REAL, Volume INTEGER)')

    try:
        for each in website.text.split('\n')[1:-1]:
            row = each.split(',')
            c.execute('INSERT INTO stock VALUES(?, ?, ?, ?, ?, ?, ?)', tuple(row))
    except:
        pass
    c.execute('SELECT * from stock')
    df = pd.DataFrame(c.fetchall(), columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
    return df

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Apple Inc. historical price, real time price, and sentiment analysis'),

    # Date selection for historical price
    html.H2('Select date range for historical price'),
    html.Div([dcc.DatePickerRange(
        id='date-picker',
        min_date_allowed=datetime(1988, 8, 2),
        max_date_allowed=datetime.now().date(),
        initial_visible_month=datetime(2017, 1, 1),
        start_date=datetime(2017, 1, 1),
        end_date=datetime(2018, 10, 24)
        )]),

    # historical price
    html.Div([
        dcc.Graph(id='historical-price')
        ], style={'width': '49%', 'display': 'inline-block'}),

    # real-time price
    html.Div([
        dcc.Graph(
            id='live-graph',
            animate=True
        ),
        dcc.Interval(
            id='graph-update',
            interval=3*1000  # 5 seconds lively update
        )
        ], style={'width': '49%', 'display': 'inline-block'}),
    html.Div([
        html.H2('Sentiment analysis?'),
        dcc.Dropdown(
            id='image-dropdown',
            options=[{'label': i, 'value': i} for i in list_of_images],
            value=list_of_images[0]
        ),
        html.Img(id='image')
    ])

])

# update historical price
@app.callback(
    Output('historical-price', 'figure'),
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),]
)
def show_historical_graph(start_date, end_date):
    df = load_historical_data('AAPL', day_begin=start_date, day_end=end_date)
    return {
        'data': [go.Scatter(
                x=df['Date'],
                y=df['Close'],
                name='Historical stock price',
                mode='lines+markers',
                marker={
                    'size':1
                }
            )],
        'layout': {
            'margin': {'l': 30, 'b': 20, 'r': 30, 't': 30},
            'yaxis': {'type': 'linear'},
            'xaxis': {'showgrid': False},
            'title': 'Historical Price'
        }
    }

# update real-time price
X = deque(maxlen=20)
Y = deque(maxlen=20)

@app.callback(
    Output('live-graph', 'figure'),
    events = [Event('graph-update', 'interval')]  # calls Interval
)
def show_realtime_price():
    response = get_YahooFinance('AAPL')
    soup = BeautifulSoup(response.text, 'lxml')
    live_price = float(soup.find(attrs={'class':"Trsdu(0.3s) Fw(b) Fz(36px) Mb(-4px) D(ib)"}).text)

    global X
    global Y

    def display_time(time):
        return '%d:%d:%d'%(time.hour, time.minute, time.second)

    X.append(display_time(datetime.now()))
    Y.append(live_price)

    data = go.Scatter(
        x=list(X),
        y=list(Y),
        name='Scatter',
        mode='lines+markers'
    )

    return {'data':[data],
            'layout': {
                'xaxis': dict(range=[X[0], display_time(datetime.now()+timedelta(seconds=60))]),
                'yaxis': dict(range=[Y[-1]*0.99, Y[-1]*1.01]),
                'title': 'Real-time Price',
                #'height': 580
                        }
            }

@app.callback(
    Output('image', 'src'),
    [Input('image-dropdown', 'value')])
def update_image_src(value):
    return static_image_route + value

@app.server.route('{}<image_path>.png'.format(static_image_route))
def serve_image(image_path):
    image_name = '{}.png'.format(image_path)
    if image_name not in list_of_images:
        raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
    return flask.send_from_directory(os.path.curdir, image_name)

if __name__ == '__main__':
    app.run_server(debug=True)
