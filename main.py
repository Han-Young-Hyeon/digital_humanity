from flask import Flask, render_template, request
from analysis import *

app = Flask(__name__)

### Rendering Pages

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/1_page1.html')
def page1_1():
    return render_template('1_page1.html')

@app.route('/2_page1.html')
def page2_1():
    return render_template('2_page1.html')

@app.route('/2_page2.html', methods=['POST'])
def result():
    if request.method == 'POST':
        고전원문_ck = int(request.form.get('selection1', 0))
        고전번역서_ck = int(request.form.get('selection2', 0))
        조선왕조실록_ck = int(request.form.get('selection3', 0))
        신역_조선왕조실록_ck = int(request.form.get('selection4', 0))
        승정원일기_ck = int(request.form.get('selection5', 0))
        일성록_ck = int(request.form.get('selection6', 0))
        한국문집총간_ck = int(request.form.get('selection7', 0))
        한국고전총간_ck = int(request.form.get('selection8', 0))
        
        keyword = str(request.form.get('keyword'))

        if(고전원문_ck==1):
            data_type_is_character=1
        elif(고전번역서_ck==1):
            data_type_is_character=0
        elif(조선왕조실록_ck==1):
            data_type_is_character=0
        elif(신역_조선왕조실록_ck==1):
            data_type_is_character=0
        elif(승정원일기_ck==1):
            data_type_is_character=0
        elif(일성록_ck==1):
            data_type_is_character=0
        elif(한국문집총간_ck==1):
            data_type_is_character=1
        elif(한국고전총간_ck==1):
            data_type_is_character=1
        else:
            data_type_is_character=0

        if(data_type_is_character==1): # 고전DB 한자 버전
            data_type_is_character=True
            result_data = total_data(keyword, 고전원문_ck, 한국문집총간_ck, 한국고전총간_ck)
            result_df = result_data[1].head().to_html(index=False)
            time_series_data_result = time_series_data(result_data[0])
            wordcloud_image = generate_wordcloud_image(result_data[0]) 
            frequency_result, _ = frequency_analysis(result_data[0])
            frequency_result1, frequency_result2 = frequency_analysis(result_data[0])
            correl = cosine_relate(frequency_result1, frequency_result2)
            _, _, _, heatmap_image = cosine_relate(frequency_result1, frequency_result2)
            network_data, network_image= network_analysis(correl)
        else: #고전DB 한글 버전
            result_data = total_data_kr(keyword, 고전번역서_ck, 조선왕조실록_ck, 신역_조선왕조실록_ck, 승정원일기_ck, 일성록_ck)
            result_df = result_data[1].head().to_html(index=False)
            time_series_data_result = time_series_data_kr(result_data[0])
            wordcloud_image = generate_wordcloud_image_kr(result_data[0]) 
            frequency_result, _ = frequency_analysis_kr(result_data[0])
            frequency_result1, frequency_result2 = frequency_analysis_kr(result_data[0])
            correl = cosine_relate_kr(frequency_result1, frequency_result2)
            _, _, _, heatmap_image = cosine_relate_kr(frequency_result1, frequency_result2)
            network_data, network_image= network_analysis_kr(correl)

        return render_template('2_page2.html',
                               result_df=result_df, 
                               time_series_data_result=time_series_data_result,
                               wordcloud_image=wordcloud_image, 
                               frequency_result=frequency_result,
                               heatmap_image=heatmap_image,
                               network_data=network_data,
                               network_image=network_image
                               )

@app.route('/3_page1.html')
def page3_1():
    return render_template('3_page1.html')

@app.route('/3_page2.html')
def page3_2():
    return render_template('3_page2.html')

@app.before_request
def before_request():
    if request.method == 'POST' and request.is_json:
        request.json_data = request.get_json()

if __name__ == '__main__':
    app.run(debug=True, port=7777)
