import requests
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm
import numpy as np
import udkanbun
import re

lzh=udkanbun.load()

###########################
####### 한국고전DB ########
###########################

# click 값에 따라 자료 리스트를 만드는 함수
def extract_list(고전번역서_ck, 고전원문_ck, 한국문집총간_ck, 한국고전총간_ck,
                 조선왕조실록_ck, 신역_조선왕조실록_ck, 승정원일기_ck, 일성록_ck, 해제_ck) :
    list_total = []
    if 고전번역서_ck == 1 :
        list_total.append('BT_AA')
    elif 고전원문_ck == 1 :
        list_total.append('GO_AA')
    elif 한국문집총간_ck == 1 : 
        list_total.append('MO_AA')
    elif 한국고전총간_ck == 1 :
        list_total.append('KP_AA')
    elif 조선왕조실록_ck == 1 :
        list_total.append('JT_AA')
    elif 신역_조선왕조실록_ck == 1 :
        list_total.append('JR_AA')
    elif 승정원일기_ck == 1 :
        list_total.append('ST_AA')
    elif 일성록_ck == 1 :
        list_total.append('IT_AA')
    elif 해제_ck == 1 : 
        list_total.append('BT_HJ')

    return list_total

#### Raw 데이터 만들기 ####
def korean_search(keyword, secld, start = 0, rows = 1000) :
    url = f'https://db.itkc.or.kr/openapi/search?secId={secld}&keyword={keyword}&start={start}&rows={rows}'

    req_str = requests.get(url).text

    totalCount = ET.fromstring(req_str).findall('.//header//field')[4].text
    totalCount = int(totalCount)

    nums = totalCount // rows +1

    row_list = []
    for num in tqdm(range(nums)):
        url = f'https://db.itkc.or.kr/openapi/search?secId={secld}&keyword={keyword}&start={start+num*1000}&rows={rows}'

        req_str = requests.get(url).text
        xtree = ET.fromstring(req_str).findall('.//result//doc')

        for nodes in xtree:
            field = {}
            for row in nodes.findall('field'):
                field[row.attrib['name']] = row.text
            row_list.append(field)

    df_raw = pd.DataFrame(row_list)

    df_raw.rename(columns = {"검색필드" : "기록", "DCI_s" : "URL"}, inplace = True)

    df = df_raw[['서명','기록', '간행년', '저자', '저자생년', '저자몰년']]

    df_datas = df_raw[['서명', '기록', '간행년', '저자', 'URL']]

    for i in range(len(df_datas)):
        url = "https://db.itkc.or.kr/dir/item?itemId=JT#dir/node?dataId=" + df_datas['URL'][i][0:27]
        df_datas['URL'][i] = url

    return df, df_datas ### URL이 포함된 데이터프레임은 df_datas 입니다.


### 시계열 감성 분석 : 시대에 따른 키워드에 대한 감성 분석 ###
from datetime import datetime
from transformers import pipeline
import plotly.express as px

sentiment_pipeline = pipeline('sentiment-analysis', model='bert-base-chinese')

# 신경 쓰지 않아도 되는 함수 : 데이터 형변환 함수
def convert_year_to_datetime(year):
    try:
        return datetime(int(year), 1, 1)
    except (ValueError, TypeError):
        return None

# 신경 쓰지 않아도 되는 함수 : 감성분석 점수 추출 함수
def get_sentiment_score(text):
    try:
        result = sentiment_pipeline(text)
        return result[0]['score'] if result else None
    except Exception as e:
        print(f"Error processing text: {text}. Error: {e}")
        return None

# 시대에 따른 키워드에 대한 감성 변화 그래프를 제시하는 함수 : 문장 길이 때문에 일부 누락이 생기는 것 존재 주의
def time_series_data(df) : # 여기에 들어가는 데이터 프레임은 korean_search를 통해 만들어진 것 중 두번째 데이터 프레임 사용 (ex. data[1])
    df['저자생년'] = df['저자생년'].astype(int)
    df['저자몰년'] = df['저자몰년'].astype(int)

    df.loc[df['간행년'] == '미상', '간행년'] = (df['저자생년'] + df['저자몰년']) / 2

    df = df[df['간행년'].str.contains('[가-힣]') != 1]

    df.dropna(subset=['간행년'], inplace=True)

    df['간행년'] = df['간행년'].astype(int)

    df = df.sort_values('간행년')
    
    df['sentence'] = df['기록'].map(lambda x:[e for e in x.split('。') if '</em>' in e])
    
    df['sentence'] = df['sentence'].map(lambda x:[re.sub('\<em class\=\"hl1\"\>(.+)\<\/em\>','\g<1>',e) for e in x])

    df['sentiment'] = df['sentence'].apply(get_sentiment_score)
    df['간행년'] = df['간행년'].apply(convert_year_to_datetime)

    df_for_sent = df[['간행년', 'sentiment']].groupby("간행년")

    df_mean = df_for_sent.mean()

    fig = px.line(df_mean, y='sentiment', title='키워드에 대한 시간에 따른 평가')
    fig.update_layout(
        title={
            'text': "키워드에 대한 시간에 따른 평가",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='간행년',
        yaxis_title='키워드 평가 수치화',
        legend_title='Legend'
    )

    fig.show()

### GPT 함수 : text에 질문을 넣을 경우 그에 따른 결과 출력 ###
import openai

def gpt_supporter(text):
    openai.api_key = "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"

    system_message_01 = text + "한글로 답변해주세요."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message_01}
        ],
        max_tokens=100
    )

    text = response.choices[0].message.content.strip()
    return text

### 빈도 분석 / 네트워크 분석 / 연관어 분석 ###
from tqdm import tqdm
from collections import Counter
import itertools

import nltk
from nltk import collocations

from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from itertools import combinations
from operator import itemgetter
import matplotlib.pyplot as plt
import networkx as nx
import udkanbun

lzh=udkanbun.load()
tqdm.pandas()

# 신경 안 써도 되는 함수 : 형태소에 따른 토큰화 함수
def tokenize(sentence,allow_pos=[]):
    try:
        s = lzh(sentence)
        if allow_pos != []:
            res = [t.form for t in s if t.upos in allow_pos]
        else:
            res = [t.form for t in s]
        return res
    except AttributeError as e:
        print(f"Error processing sentence: {sentence}. Error: {e}")
        return []

import plotly.graph_objects as go

def frequency_analysis(df) : # 여기 데이터 프레임에는 korean_search를 통해 만들어진 데이터 프레임 중 첫번째를 사용 (ex. data[0])
    df['token'] = df['기록'].progress_map(lambda x:tokenize(x,['NOUN','PROPN','VERB','ADV', 'ADJ']))
    
    cnt = Counter(list(itertools.chain(*df['token'].tolist())))
    frequency = pd.DataFrame(cnt.most_common(10))
    frequency.rename(columns={0 : "단어1", 1 : "빈도1"}, inplace=True)
    
    token_list = list(itertools.chain(*df['token'].tolist()))
    bgs = nltk.bigrams(token_list)
    fdist= nltk.FreqDist(bgs)
    fd= fdist.items()
    fd_df = pd.DataFrame(fd, columns =['단어2', '빈도2'])

    fd_df=fd_df.sort_values('빈도2', ascending = False)
    fd_df.reset_index(drop = True, inplace = True)
    fd_df = fd_df.head(10)

    frequency = pd.concat([frequency, fd_df], axis=1)

    frequency.index = frequency.index + 1

    frequency['단어 2'] = frequency['단어2'].apply(lambda x: ', '.join(x) if isinstance(x, tuple) else x)

    fig1 = go.Figure(data=[go.Bar(x=frequency['단어1'], y=frequency['빈도1'])])

    fig1.update_layout(
        title={
            'text': "키워드 포함 맥락 내 단어 출현 빈도 1",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='단어 1',
        yaxis_title='빈도',
        legend_title='Legend'
    )

    fig2 = go.Figure(data=[go.Bar(x=frequency['단어 2'], y=frequency['빈도2'])])

    fig2.update_layout(
        title={
            'text': "키워드 포함 맥락 내 단어 출현 빈도 2",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='단어 2',
        yaxis_title='빈도',
        legend_title='Legend'
    )

    fig1.show()
    fig2.show()

    return frequency

### 연관어 분석 ###
def build_doc_term_mat(doc_list):
    vectorizer = CountVectorizer(tokenizer=str.split, max_features=10)
    dt_mat = vectorizer.fit_transform(doc_list)
    vocab = vectorizer.get_feature_names()
    return dt_mat, vocab

def build_word_cooc_mat(dt_mat):
    co_mat = dt_mat.T * dt_mat
    co_mat.setdiag(0)
    return co_mat.toarray()

def get_word_sim_mat(co_mat):
    sim_mat = pdist(co_mat, metric='cosine')
    sim_mat = squareform(sim_mat)
    return sim_mat

def get_sorted_word_sims(sim_mat, vocab):
    sims = []
    for i, j in combinations(range(len(vocab)), 2):
        if sim_mat[i, j] == 0:
            continue
        sims.append((vocab[i], vocab[j], sim_mat[i, j]))
    mat_to_list = sorted(sims, key=itemgetter(2), reverse=True)
    return mat_to_list

### 네트워크 분석 ###
def build_word_sim_network(mat_to_list, minimum_span=False):
    G = nx.Graph()
    NUM_MAX_WORDS = 30
    for word1, word2, sim in mat_to_list[:NUM_MAX_WORDS]:
        G.add_edge(word1, word2, weight=sim)
    if minimum_span:
        return nx.minimum_spanning_tree(G)
    else:
        return G

def draw_network(G):
    weights = nx.get_edge_attributes(G,'weight').values()
    width = [weight / max(weights)*3 for weight in weights]

    nx.draw_networkx(G,
        pos=nx.kamada_kawai_layout(G),
        node_size=500,
        node_color="blue",
        font_color="white",
        font_family='NanumBarunGothic',
        with_labels=True,
        font_size=5,
        width=width)
    plt.axis("off")