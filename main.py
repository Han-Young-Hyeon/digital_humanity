import requests
import pandas as pd
import re
import time
import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm
import numpy as np
import re
import udkanbun

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

# 시대에 따른 키워드에 대한 감성 변화 그래프를 제시하는 함수 : 문장 길이 때문에 일부 누락이 생기는 것 존재 주의
def time_series_data(df) :
    df['저자생년'] = df['저자생년'].astype(int)
    df['저자몰년'] = df['저자몰년'].astype(int)

    df.loc[df['간행년'] == '미상', '간행년'] = (df['저자생년'] + df['저자몰년']) / 2

    df = df[df['간행년'].str.contains('[가-힣]') != 1]

    df.dropna(subset=['간행년'], inplace=True)

    df['간행년'] = df['간행년'].astype(int)

    df = df.sort_values('간행년')
    
    df['sentiment'] = df['기록'].apply(get_sentiment_score)
    df['간행년'] = df['간행년'].apply(convert_year_to_datetime)
    df = df[(df['간행년'] < 2000)&(df['간행년'] > 300)]

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

### GPT 함수 : text에 질문을 넣을 경우 그에 따른 결과 출력
import openai

def gpt_supporter(text):
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