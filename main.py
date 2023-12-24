from flask import Flask, render_template, request
from analysis import *
import bleach

app = Flask(__name__)

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

        if(data_type_is_character==1): # 한국고전종합DB 한문 버전
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
        else: # 한국고전종합DB 한글 버전
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

@app.route('/3_page2.html', methods=['POST'])
def page3_2():
    if request.method == 'POST':
        # TEXT="""효종 승하 후, 현종 즉위 직후 효종의 의붓 모후[1]인 장렬왕후가 행해야 했던 상례 격식을 두고 서인과 남인 간에 여러 차례 격렬하게 벌어졌던 학술적, 정치적, 사회적 논의를 말한다.[2] 1차(기해예송 1659년)는 효종이 사망한 이후, 모후인 장렬왕후가 입어야 하는 상복의 '규례'를 두고 일어난 논쟁이다. 그리고 2차(갑인예송 1674년)는 장렬왕후가 며느리 되는 인선왕후에 대해 상복을 몇 년 입어야 하느냐로 벌인 논쟁이다.(현종 15년).역사에 대해 잘 모르는 사람들에게는 무의미한 논쟁으로 취급되지만, 조선시대에는 유교가 정치, 사회, 생활의 기준이 되었기 때문에 매우 중요한 문제였다. 또한 왕조국가의 특성상 국왕의 정통성과 직결된 문제였기 때문에 현실 정치와 크게 엮인 중대한 논란으로, 단순한 예법 논란이 아니었다. 이 논쟁은 당시 성리학을 기반으로 한 전제왕조 국가 조선에서 왕의 정통성이 걸린 중대한 논쟁이었고, 이후 환국과 연결되어 조선 정치사의 중요한 분기점이 되었다. 왕조 국가에서 왕의 정통성은 국기(國基)와 관련된 중대한 사안이다. 그리고 어느 나라든 국가를 책임지고 이끄는 지도자의 정당성은 매우 중요하며, 정치력은 정당성, 정통성에서 나온다는 말이 과언이 아니다. 그렇기에 그러한 정통성 논쟁이 파벌 간 무력 충돌이 아닌 무혈로 끝났단 점에서 예송논쟁은 조선시기 한국사 내에서 상당히 중요한 사건이라고 볼 수 있다."""
        # TEXT="""古之建功立事。奠安國家者。類皆以學術行誼爲根基。蓄積深厚。旣已達於爲邦。而又必判義利審輕重。其執持於平素者。如欛柄之在手。無所撓奪。用能出而當難。大有成就。不徒然也。苟無如是之學力。以一時意氣。幸而取雋焉。則終亦不免爲壞敗之歸。何則。以其本末虛實之辨異也。在昔穆廟之世。有月川李公者。居家敦孝友。立朝秉忠純。內外單盡。蔚有樹立。而尤奉公守法。不屈於形要。由是頗爲時議所左。棲遲郡邑。積數十年。歸則閉門却掃。捐去俗事。密以斯文自娛。上下千古。所得益富。而人顧不盡知也。及以治民著最。稍闢其仕塗。末乃進之諫長銓貳之列。則亦顯矣。時公少日同榻友。持國柄政。廣植黨與。而公介立標高。不少降色辭。逮海寇之縕。糾合義旅。激厲衆心。以單師遏大敵。以書生辦奇績。使兇酋死咋。不敢肆意蹂逞。已又天兵繼之。訖成中興之功。始落其角距者公也。觀其踞積藁欲自燼。倉卒熊魚之取舍。而熟講乎忘身殉國之義者。卽此可知。卒之天人順信。一擧事集。初非公所預期。則牛溪先生之以伏節死義許公者。信知言哉。當捷書之聞。主上聳喜。特擢卿秩。朝野廩廩有公輔之望。而時輩方深嫉牛溪。以公牛溪所扶獎。移怒而齕公。公亦不肯周旋於季孟之間。退處海曲。弗屑嚇腐。夷考公平生。蓋斤斤乎出處之必審。此其爲建功立事之本也。夫豈今人所能及哉。公卒之五年。追策宣武勳。屢贈至左議政。封府院君。諡忠穆。月川者。公所啓邑號也。公文章超絶。栗谷先生嘗以尉薦於朝。月汀尹公。又以高文振古聲稱之。當時文苑。固自有定論。如不佞者何敢更有僭述。只以世人但知公勳業之隆。而不知公有學有守。本領卓然。故竊附微顯闡幽之義。爲之說如此。以備論世者商確。崇禎紀元後再丙辰復月上浣。龍仁李宜顯。序。"""
        TEXT = bleach.clean(request.form.get('DATA_INPUT', type=str))

        한글_ck = int(request.form.get('selection_kr', 0))
        한자_ck = int(request.form.get('selection_ch', 0))

        if 한자_ck==1: # 사용자데이터 한문 버전
            user_word_cloud, user_df, df = text_frequently(TEXT)
            user_df_tfidf, user_heatmap_image = text_correlate(user_df, df)
        else: # 사용자데이터 한글 버전
            user_word_cloud, user_df, df = text_frequently_kr(TEXT)
            user_df_tfidf, user_heatmap_image = text_correlate_kr(user_df, df)

        return render_template('3_page2.html',
                               TEXT=TEXT,
                               user_word_cloud=user_word_cloud,
                               user_df=user_df,
                               df=df,
                               user_heatmap_image=user_heatmap_image,
                               user_df_tfidf=user_df_tfidf
                               )

@app.before_request
def before_request():
    if request.method == 'POST' and request.is_json:
        request.json_data = request.get_json()

if __name__ == '__main__':
    app.run(debug=True, port=7777)
