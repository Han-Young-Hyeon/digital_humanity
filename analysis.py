###################################
####### 한국고전DB 한문버전 ########
###################################

import requests
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm
import numpy as np
import udkanbun
import re
pd.options.mode.chained_assignment = None

lzh=udkanbun.load()

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = 'gulim.ttc'
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rcParams['font.family'] = font_name

# click 값에 따라 전체 데이터를 가져오는 함수
def total_data(keyword, 고전원문_ck, 한국문집총간_ck, 한국고전총간_ck) :
    book = 0
    if 고전원문_ck == 1 :
        book = 'GO_AA'
    elif 한국문집총간_ck == 1 : 
        book = 'MO_AA'
    elif 한국고전총간_ck == 1 :
        book = 'KP_AA'

    df = korean_search(keyword = keyword, secld = book)
    total_data = df[0]
    total_data_with_url = df[1]

    return total_data, total_data_with_url # total_data_with_url UI 상에 들어갈 데이터 프레임

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

    df = pd.DataFrame(row_list)

    df.rename(columns = {"검색필드" : "기록", "DCI_s" : "URL"}, inplace = True)

    df['Sentence Raw'] = df['기록'].map(lambda x:[e for e in x.split('。') if '</em>' in e])
    df['Sentence Raw'] = df['Sentence Raw'].map(lambda x: [re.sub(r'<[^>]+>', '', e) for e in x])

    df_datas = df[['서명', 'Sentence Raw', '간행년', 'URL']]
    df_datas.rename({'Sentence Raw' : '기록'}, inplace=True)

    for i in range(len(df_datas)):
        url = "https://db.itkc.or.kr/dir/item?itemId=JT#dir/node?dataId=" + str(df_datas['URL'][i])[0:27]
        df_datas['URL'][i] = url

    return df, df_datas

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

# 시대에 따른 키워드에 대한 감성 변화 그래프를 제시하는 함수
def time_series_data(df) : # 여기에 들어가는 데이터 프레임은 total_data를 통해 만들어진 것 중 첫번째 데이터 프레임 사용 (ex. data[0])
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

    # fig = px.line(df_mean, y='sentiment', title='키워드에 대한 시간에 따른 평가')
    # fig.update_layout(
    #     title={
    #         'text': "키워드에 대한 시간에 따른 평가",
    #         'y':0.9,
    #         'x':0.5,
    #         'xanchor': 'center',
    #         'yanchor': 'top'
    #     },
    #     xaxis_title='간행년',
    #     yaxis_title='키워드 평가 수치화',
    #     legend_title='Legend'
    # )

    # fig.show()
    return df_mean

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

###
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

### 빈도분석 : 그래프 생성 ###
# 신경 안 써도 되는 함수 : 형태소에 따른 토큰화 함수
def tokenize(sentence,allow_pos=[]):
    try :
        s = lzh(sentence)
        if allow_pos !=[]:
            res = [t.form+'/'+t.upos.lower() for t in s if t.upos in allow_pos]
        else:
            res = [t.form+'/'+t.upos.lower() for t in s]
        return res
    except AttributeError as e:
        print(f"Error processing sentence: {sentence}. Error: {e}")
        return []

import plotly.graph_objects as go

def frequency_analysis(df) : # 여기 데이터 프레임에는 total_data를 통해 만들어진 데이터 프레임 중 첫번째를 사용 (ex. data[0])
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

    if len(fd_df) > 10 :
        fd_df = fd_df.head(10)
    else :
        fd_df = fd_df

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

    return frequency, df ## 첫번째 데이터프레임은 같이 등장하는 빈도가 높은 단어를 선별하여 이후 연관어 분석에 활용하기 위한 데이터 / 두 번째는 token 확보 데이터

### 워드 클라우드 생성 ###
from wordcloud import WordCloud
from io import BytesIO
import base64

def tokenize_wordcloud(sentence,allow_pos=[]):
  s = lzh(sentence)
  if allow_pos !=[]:
    res = [t.form for t in s if t.upos in allow_pos]
  else:
    res = [t.form for t in s]
  return res

def generate_wordcloud_image(df) : # total data에서 만들어진 것 중 첫번째 데이터프레임이 df
    df['token_2'] = df['기록'].progress_map(lambda x:tokenize_wordcloud(x,['NOUN','PROPN','VERB','ADV', 'ADJ']))

    counts=Counter(list(itertools.chain(*df['token_2'].tolist())))
    tags = counts.most_common(40)

    font_path = 'C:/Windows/Fonts/gulim.ttc'
    wc = WordCloud(font_path=font_path, background_color="white", relative_scaling=0.2) ### font source need
    cloud = wc.generate_from_frequencies(dict(tags))

    # 이미지를 바이트로 변환하여 반환
    img_buffer = BytesIO()
    cloud.to_image().save(img_buffer, format="PNG")
    img_str = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    return img_str

### 연관어 및 네트워크 분석에 필요한 함수 : 신경 쓰지 않으셔도 됩니다. ###
def build_doc_term_mat(doc_list):
    vectorizer = CountVectorizer(tokenizer=str.split)
    dt_mat = vectorizer.fit_transform(doc_list)
    vocab = vectorizer.get_feature_names_out()
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
    weights = nx.get_edge_attributes(G, 'weight').values()
    width = [weight / max(weights) * 3 for weight in weights]

    pos = nx.kamada_kawai_layout(G)

    # Create a new figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw the network graph using NetworkX
    nx.draw_networkx(G,
                     pos=pos,
                     node_size=1500,
                     node_color="blue",
                     font_color="white",
                     font_family='gulim',
                     with_labels=True,
                     font_size=10,
                     width=width,
                     ax=ax)

    # Save the figure to a BytesIO buffer
    network_buffer = BytesIO()
    fig.savefig(network_buffer, format="PNG")
    plt.close(fig)  # Close the figure

    # Encode the image to base64
    network_str = base64.b64encode(network_buffer.getvalue()).decode("utf-8")

    return network_str

### 연관어 분석 ###
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def create_heatmap(df_sim):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the heatmap using Seaborn
    sns.heatmap(df_sim, annot=True, fmt=".2f", cmap="YlGnBu", square=True, ax=ax)
    ax.set_title("빈출 단어 간 연관성 분석")

    # Save the figure to a BytesIO buffer
    img_buffer = BytesIO()
    FigureCanvas(fig).print_png(img_buffer)

    # Encode the image to base64
    img_str = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

    return img_str

def cosine_relate(df1, df2) : ### 여기 들어갈 데이터프레임은 각각 frequency_analysis으로부터 나온 첫번째 및 두번째 데이터프레임
    separated_list = []
    for item in df1['단어 2']:
        separated_list.extend(item.split(', '))

    separated_list =  list(set(separated_list))

    token_tag_list = df2['token'].map(lambda x: ' '.join(x)).tolist()

    vect = CountVectorizer(tokenizer=str.split)
    document_term_matrix = vect.fit_transform(token_tag_list)  # 문서-단어 행렬

    tf = pd.DataFrame(document_term_matrix.toarray(), columns=vect.get_feature_names_out())

    vect = TfidfVectorizer(tokenizer=str.split)
    tfvect = vect.fit_transform(token_tag_list)  # 문서-단어 행렬

    tfidf_df = pd.DataFrame(tfvect.toarray(), columns = vect.get_feature_names_out())

    tf = tf[separated_list]
    tfidf_df = tfidf_df[separated_list]

    tfidf=[]
    for col in tfidf_df.columns :
        tfidf.append(tfidf_df[col].sum())

    df_tfidf=pd.DataFrame(list(zip(tfidf_df.columns,tfidf)), columns=['words', 'tfidf_score'])
    df_tfidf=df_tfidf.sort_values('tfidf_score', ascending=False)
    df_tfidf.reset_index(drop=True, inplace=True)

    dt_matrix, vocab_list= build_doc_term_mat(token_tag_list)
    co_matrix_raw = build_word_cooc_mat(dt_matrix)

    df_co = pd.DataFrame(co_matrix_raw, columns=vocab_list, index=vocab_list)
    df_co = df_co.loc[separated_list, separated_list]
    co_matrix = np.matrix(df_co)

    sim_matrix = get_word_sim_mat(co_matrix)
    df_sim = pd.DataFrame(sim_matrix, columns=separated_list, index=separated_list)

    pd.set_option('mode.use_inf_as_na', True)

    font_path = 'C:/Windows/Fonts/gulim.ttc'
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams['font.family'] = font_name

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_sim, annot=True, fmt=".2f", cmap="YlGnBu", square=True)
    plt.title("빈출 단어 간 연관성 분석")

    img_str = create_heatmap(df_sim)

    return df_tfidf, co_matrix_raw, token_tag_list, img_str



### 네트워크 분석 ###
def network_analysis(correldata) : ## correldata는 cosine_relate의 함숫값입니다.
    token_tag_list = correldata[2]
    dt_matrix, vocab_list= build_doc_term_mat(token_tag_list)
    co_matrix = build_word_cooc_mat(dt_matrix)

    matrix_to_list = get_sorted_word_sims(co_matrix, vocab_list)
    G = build_word_sim_network(matrix_to_list)
    network_image=draw_network(G)

    close_centrality = nx.closeness_centrality(G)
    deg_centrality = nx.degree_centrality(G)
    bet_centrality = nx.betweenness_centrality(G, normalized = True, endpoints = False)

    close_df= pd.DataFrame(close_centrality.items(), columns=['token1', 'close_centrality'])
    deg_df = pd.DataFrame(deg_centrality.items(), columns=['token2', 'degree_centrality'])
    bet_df=pd.DataFrame(bet_centrality.items(), columns=['token3', 'between_centrality'])

    cent_df = pd.concat([close_df, deg_df, bet_df], axis=1)

    fig1 = go.Figure(data=[go.Bar(x=cent_df['token1'], y=cent_df['close_centrality'])])

    fig1.update_layout(
        title={
            'text': "근접 중심성 지표",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='단어',
        yaxis_title='중심도',
        legend_title='Legend'
    )

    fig2 = go.Figure(data=[go.Bar(x=cent_df['token2'], y=cent_df['degree_centrality'])])

    fig2.update_layout(
        title={
            'text': "연결 중심성",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='단어',
        yaxis_title='중심도',
        legend_title='Legend'
    )

    fig3 = go.Figure(data=[go.Bar(x=cent_df['token3'], y=cent_df['between_centrality'])])

    fig3.update_layout(
        title={
            'text': "매개 중심성",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='단어',
        yaxis_title='중심도',
        legend_title='Legend'
    )

    return pd.concat([close_df, deg_df, bet_df], axis=1), network_image

###################################
####### 한국고전DB 한글버전 ########
###################################

import requests
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm
import numpy as np
import re

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
pd.options.mode.chained_assignment = None

font_path = 'gulim.ttc'
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rcParams['font.family'] = font_name

# click 값에 따라 전체 데이터를 가져오는 함수
def total_data_kr(keyword, 고전번역서_ck, 조선왕조실록_ck, 신역_조선왕조실록_ck, 승정원일기_ck, 일성록_ck) :
    book = 0
    if 조선왕조실록_ck == 1 :
        book = 'JT_AA'
    elif 신역_조선왕조실록_ck == 1 :
        book = 'JR_AA'
    elif 승정원일기_ck == 1 :
        book = 'ST_AA'
    elif 일성록_ck == 1 :
        book = 'IT_AA'
    elif 고전번역서_ck == 1 :
        book = 'BT_AA'

    df = korean_search_kr(keyword = keyword, secld = book)
    total_data = df[0]
    total_data_with_url = df[1]

    return total_data, total_data_with_url # total_data_with_url UI 상에 들어갈 데이터 프레임

#### Raw 데이터 만들기 ####
def korean_search_kr(keyword, secld, start = 0, rows = 1000) :
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

    df = pd.DataFrame(row_list)

    df.rename(columns = {"검색필드" : "기록", "DCI_s" : "URL"}, inplace = True)

    df['Sentence Raw'] = df['기록'].map(lambda x:[e for e in x.split('。') if '</em>' in e])
    df['Sentence Raw'] = df['Sentence Raw'].map(lambda x: [re.sub(r'<[^>]+>', '', e) for e in x])

    df_datas = df[['서명', 'Sentence Raw', '간행년', 'URL']]
    df_datas.rename({'Sentence Raw' : '기록'}, inplace=True)

    for i in range(len(df_datas)):
        url = "https://db.itkc.or.kr/dir/item?itemId=JT#dir/node?dataId=" + str(df_datas['URL'][i])[0:27]
        df_datas['URL'][i] = url

    return df, df_datas

### 시계열 감성 분석 : 시대에 따른 키워드에 대한 감성 분석 ###
from datetime import datetime
from transformers import pipeline
import plotly.express as px

sentiment_pipeline_kr = pipeline('sentiment-analysis', model='monologg/kobert')

# 신경 쓰지 않아도 되는 함수 : 감성분석 점수 추출 함수
def get_sentiment_score_kr(text):
    try:
        result = sentiment_pipeline_kr(text)
        return result[0]['score'] if result else None
    except Exception as e:
        print(f"Error processing text: {text}. Error: {e}")
        return None

# 시대에 따른 키워드에 대한 감성 변화 그래프를 제시하는 함수
def time_series_data_kr(df) : # 여기에 들어가는 데이터 프레임은 total_data를 통해 만들어진 것 중 첫번째 데이터 프레임 사용 (ex. data[0])
    df = df[df['간행년'].str.contains('[가-힣]') != 1]

    df.dropna(subset=['간행년'], inplace=True)

    df['간행년'] = df['간행년'].astype(int)

    df = df.sort_values('간행년')
    
    df['sentence'] = df['기록'].map(lambda x:[e for e in x.split('。') if '</em>' in e])
    
    df['sentence'] = df['sentence'].map(lambda x:[re.sub('\<em class\=\"hl1\"\>(.+)\<\/em\>','\g<1>',e) for e in x])

    df['sentiment'] = df['sentence'].apply(get_sentiment_score_kr)
    df['간행년'] = df['간행년'].apply(convert_year_to_datetime)

    df_for_sent = df[['간행년', 'sentiment']].groupby("간행년")

    df_mean = df_for_sent.mean()

    # fig = px.line(df_mean, y='sentiment', title='키워드에 대한 시간에 따른 평가')
    # fig.update_layout(
    #     title={
    #         'text': "키워드에 대한 시간에 따른 평가",
    #         'y':0.9,
    #         'x':0.5,
    #         'xanchor': 'center',
    #         'yanchor': 'top'
    #     },
    #     xaxis_title='간행년',
    #     yaxis_title='키워드 평가 수치화',
    #     legend_title='Legend'
    # )

    # fig.show()

    return df_mean

###
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
from konlpy.tag import Komoran

### 빈도분석 : 그래프 생성 ###
# 신경 안 써도 되는 함수 : 형태소에 따른 토큰화 함수
komoran = Komoran()

# Tokenization function for Korean text
def tokenize_kr(sentence, allow_pos=[]):
    try:
        tokens = komoran.pos(sentence)
        if allow_pos != []:
            res = [f'{token[0]}/{token[1]}' for token in tokens if token[1] in allow_pos]
        else:
            res = [f'{token[0]}/{token[1]}' for token in tokens]
        return res
    except Exception as e:
        print(f"Error processing sentence: {sentence}. Error: {e}")
        return []
    
import plotly.graph_objects as go

def frequency_analysis_kr(df) : # 여기 데이터 프레임에는 total_data를 통해 만들어진 데이터 프레임 중 첫번째를 사용 (ex. data[0])
    df['token'] = df['기록'].progress_apply(lambda x: tokenize_kr(x, ['NNG', 'NNP', 'VV', 'VA', 'MAG', 'MAJ']))
    
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

    if len(fd_df) > 10 :
        fd_df = fd_df.head(10)
    else :
        fd_df = fd_df

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

    # fig1.show()
    # fig2.show()

    return frequency, df ## 첫번째 데이터프레임은 같이 등장하는 빈도가 높은 단어를 선별하여 이후 연관어 분석에 활용하기 위한 데이터 / 두 번째는 token 확보 데이터

### 워드 클라우드 생성 ###
from wordcloud import WordCloud

def tokenize_wordcloud_kr(sentence, allow_pos=[]):
    try :
        tokens = komoran.pos(sentence)
        if allow_pos != []:
            res = [token[0] for token in tokens if token[1] in allow_pos]
        else:
            res = [token[0] for token in tokens]
        return res
    except Exception as e:
        print(f"Error processing sentence: {sentence}. Error: {e}")
        return []

def generate_wordcloud_image_kr(df) : # total data에서 만들어진 것 중 첫번째 데이터프레임이 df
    df['token'] = df['기록'].progress_apply(lambda x: tokenize_wordcloud_kr(x, ['NNG', 'NNP', 'VV', 'VA', 'MAG', 'MAJ']))

    counts=Counter(list(itertools.chain(*df['token'].tolist())))
    tags = counts.most_common(40)

    font_path = 'gulim.ttc'
    wc = WordCloud(font_path=font_path, background_color="white", relative_scaling=0.2) ### font source need
    cloud = wc.generate_from_frequencies(dict(tags))

    # 이미지를 바이트로 변환하여 반환
    img_buffer = BytesIO()
    cloud.to_image().save(img_buffer, format="PNG")
    img_str = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    return img_str

### 연관어 분석 ###
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import seaborn as sns

def cosine_relate_kr(df1, df2) : ### 여기 들어갈 데이터프레임은 각각 frequency_analysis으로부터 나온 첫번째 및 두번째 데이터프레임
    separated_list = []
    for item in df1['단어 2']:
        separated_list.extend(item.split(', '))

    separated_list =  list(set(separated_list))

    separated_list = [name.lower() for name in separated_list]

    token_tag_list = df2['token'].map(lambda x: ' '.join(x)).tolist()

    vect = CountVectorizer(tokenizer=str.split)
    document_term_matrix = vect.fit_transform(token_tag_list)  # 문서-단어 행렬

    tf = pd.DataFrame(document_term_matrix.toarray(), columns=vect.get_feature_names_out())

    vect = TfidfVectorizer(tokenizer=str.split)
    tfvect = vect.fit_transform(token_tag_list)  # 문서-단어 행렬

    tfidf_df = pd.DataFrame(tfvect.toarray(), columns = vect.get_feature_names_out())

    tf = tf[separated_list]
    tfidf_df = tfidf_df[separated_list]

    tfidf=[]
    for col in tfidf_df.columns :
        tfidf.append(tfidf_df[col].sum())

    df_tfidf=pd.DataFrame(list(zip(tfidf_df.columns,tfidf)), columns=['words', 'tfidf_score'])
    df_tfidf=df_tfidf.sort_values('tfidf_score', ascending=False)
    df_tfidf.reset_index(drop=True, inplace=True)

    dt_matrix, vocab_list= build_doc_term_mat(token_tag_list)
    co_matrix_raw = build_word_cooc_mat(dt_matrix)

    df_co = pd.DataFrame(co_matrix_raw, columns=vocab_list, index=vocab_list)
    df_co = df_co.loc[separated_list, separated_list]
    co_matrix = np.matrix(df_co)

    sim_matrix = get_word_sim_mat(co_matrix)
    df_sim = pd.DataFrame(sim_matrix, columns=separated_list, index=separated_list)

    pd.set_option('mode.use_inf_as_na', True)

    font_path = 'gulim.ttc'
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams['font.family'] = font_name

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_sim, annot=True, fmt=".2f", cmap="YlGnBu", square=True)
    plt.title("빈출 단어 간 연관성 분석")
    img_str = create_heatmap(df_sim)

    return df_tfidf, co_matrix_raw, token_tag_list, img_str

### 네트워크 분석 ###
def network_analysis_kr(correldata) : ## correldata는 cosine_relate의 함숫값입니다.
    token_tag_list = correldata[2]
    dt_matrix, vocab_list= build_doc_term_mat(token_tag_list)
    co_matrix = build_word_cooc_mat(dt_matrix)

    matrix_to_list = get_sorted_word_sims(co_matrix, vocab_list)
    G = build_word_sim_network(matrix_to_list)
    network_image=draw_network(G)

    close_centrality = nx.closeness_centrality(G)
    deg_centrality = nx.degree_centrality(G)
    bet_centrality = nx.betweenness_centrality(G, normalized = True, endpoints = False)

    close_df= pd.DataFrame(close_centrality.items(), columns=['token1', 'close_centrality'])
    deg_df = pd.DataFrame(deg_centrality.items(), columns=['token2', 'degree_centrality'])
    bet_df=pd.DataFrame(bet_centrality.items(), columns=['token3', 'between_centrality'])

    cent_df = pd.concat([close_df, deg_df, bet_df], axis=1)

    fig1 = go.Figure(data=[go.Bar(x=cent_df['token1'], y=cent_df['close_centrality'])])

    fig1.update_layout(
        title={
            'text': "근접 중심성 지표",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='단어',
        yaxis_title='중심도',
        legend_title='Legend'
    )

    fig2 = go.Figure(data=[go.Bar(x=cent_df['token2'], y=cent_df['degree_centrality'])])

    fig2.update_layout(
        title={
            'text': "연결 중심성",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='단어',
        yaxis_title='중심도',
        legend_title='Legend'
    )

    fig3 = go.Figure(data=[go.Bar(x=cent_df['token3'], y=cent_df['between_centrality'])])

    fig3.update_layout(
        title={
            'text': "매개 중심성",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='단어',
        yaxis_title='중심도',
        legend_title='Legend'
    )

    # fig1.show()
    # fig2.show()
    # fig3.show()

    return pd.concat([close_df, deg_df, bet_df], axis=1), network_image