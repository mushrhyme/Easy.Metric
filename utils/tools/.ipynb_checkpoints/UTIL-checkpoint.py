import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import random
from copy import deepcopy
import base64
import colorsys
from io import BytesIO
import plotly.io as pio
from func.regression.response_optimizer import *
###########################################################################################################################
# 버튼
###########################################################################################################################
def convert_to_calculatable_df():
    cal_data_df = deepcopy(st.session_state.df)
    edited_data = st.session_state.change_df_data['edited_rows']
    for row, v in edited_data.items():
        for col, val in v.items():
            cal_data_df.loc[row, col] = val if val != "" else np.nan
    # df = cal_data_df.astype('float')
    for col in cal_data_df.columns:
        try:
            cal_data_df[col] = cal_data_df[col].astype(float)
        except:
            # 숫자로 변환할 수 없는 경우 원래 타입 유지
            pass
    if not cal_data_df.isna().all().all():
        cal_data_df = cal_data_df.dropna(axis=1, how='all') # 결측 열 삭제
        cal_data_df = cal_data_df.dropna(axis=0, how='all') # 결측 행 삭제
    return cal_data_df

def update_table():
    edited_data = st.session_state.change_df_data['edited_rows']
    for row, v in edited_data.items():
        for col, val in v.items():
            st.session_state.df.loc[row, col] = val

def update_colname(df):
    col = st.session_state.df.columns.tolist()
    for i, col_name in enumerate(df.columns):
        if i < len(st.session_state.col_name):
            st.session_state.col_name[i] = col_name
            col[i] = col_name
    st.session_state.df.columns = col

def init_df():
    st.session_state.number_of_newClick += 10
    if st.session_state.number_of_newClick > 50:
        st.session_state.number_of_newClick = 0
    st.session_state['col_name'] = ['col' + str(i+1) for i in range(st.session_state.number_of_newClick, st.session_state.number_of_newClick+st.session_state.number_of_columns)]
    st.session_state['df'] = pd.DataFrame(
        np.full((st.session_state.number_of_rows, st.session_state.number_of_columns), None), 
        columns=st.session_state.col_name
    )


def get_filename(filename):
    folder = f'DB/{st.session_state.id}({st.session_state.name})'
    return os.path.join(folder, filename+'.json')


def menu(anlysis=None, plot=None, run=None):
    if st.session_state['new'].button('새 데이터', use_container_width=True):
        init_df()
        st.rerun()
    if st.session_state['history'].button('열기', use_container_width=True):
        open_history()
    if st.session_state['Del'].button('삭제', use_container_width=True):
        Del()
    if st.session_state['save'].button('저장', use_container_width=True):
        save()
    if st.session_state['run'].button('분석하기', use_container_width=True):
        cal(anlysis, plot, run)
    if st.session_state['upload'].button('파일 업로드', use_container_width=True):
        upload_file()
    if st.session_state['login'].button('로그인', use_container_width=True):
        login()


@st.dialog(title='데이터 불러오기:')
def open_history():
    if st.session_state.id == '' or st.session_state.name == '':
        st.error("로그인을 먼저 해주세요!")
        timecount = st.empty()
        for i in range(3):
            timecount.write(f"{3-i}초 후에 창이 닫힙니다.")
            time.sleep(1)
        st.rerun()
    else:
        filelist = os.listdir(f'DB/{st.session_state.id}({st.session_state.name})')
        for i in range(len(filelist)):
            filelist[i] = filelist[i].split('.')[0]
        with st.container(border=True):
            selected = st.radio('데이터를 선택해주세요:', filelist)
            st.caption(f'선택된 데이터: {selected}')
        if st.button('open', use_container_width=True):
            # st.title('불러옵니다..')
            if selected is None:
                st.error("데이터를 먼저 저장해주세요.")
                return
            init_df()

            df = pd.read_json(get_filename(selected))
            st.session_state.df.iloc[:df.shape[0], :df.shape[1]] = df.values
            # update_table()
            update_colname(df)
            st.rerun()


@st.dialog(title='데이터 삭제:')
def Del():
    if st.session_state.id != '' and st.session_state.name != '':
        filelist = os.listdir(f'DB/{st.session_state.id}({st.session_state.name})')
        for i in range(len(filelist)):
            filelist[i] = filelist[i].split('.')[0]
        with st.container(border=True):
            selected = st.radio('데이터를 선택해주세요:', filelist)
            st.caption(f'선택된 데이터: {selected}')
        if st.button('삭제하기', use_container_width=True):
            os.remove(get_filename(selected))
            st.rerun()
    else:
        st.error("데이터를 삭제하기 전 로그인을 먼저 해주세요!")
        timecount = st.empty()
        for i in range(3):
            timecount.write(f"{3-i}초 후에 창이 닫힙니다.")
            time.sleep(1)
        st.rerun()


@st.dialog(title='데이터 저장:')
def save():
    if st.session_state.id != '' and st.session_state.name != '':
        df = convert_to_calculatable_df()
        st.write(df)
        filename = st.text_input('새로 저장할 데이터명을 입력해주세요:', )
        if st.button('저장하기', use_container_width=True):
            fullname = get_filename(filename)
            
            # 컬럼 중복 예방조치
            new_col = []
            for i, col in enumerate(df.columns.tolist()):
                if 'col' in col:
                    new_col.append(f'{filename}{i+1}')
                else:
                    new_col.append(col)
                    
            df.columns = new_col
            
            df.to_json(fullname, orient='records', force_ascii=False, indent=4)
            st.success('저장되었습니다.')
            st.rerun()
    else:
        st.error("데이터 저장 전 로그인을 먼저 해주세요!")
        timecount = st.empty()
        for i in range(3):
            timecount.write(f"{3-i}초 후에 창이 닫힙니다.")
            time.sleep(1)
        st.rerun()


@st.dialog(title='분석결과:', width='large')
def cal(anlysis, plot, run):
    df = convert_to_calculatable_df()
    if df.columns.tolist() == st.session_state.col_name:
        st.error("컬럼명을 변경하고 다시 분석하기를 클릭해주세요!")
        st.caption("분석하고 싶은 컬럼의 이름을 기본값이 아닌 다른 이름으로 변경해주세요.")
    else:
        run()
        if anlysis.__name__=="regression_analysis_cal":
            with st.expander("반응 최적화 도구", expanded=False):
                response_optimizer_run(df)
        
        

@st.dialog(title='데이터 업로드:')
def upload_file():
    uploaded_file = st.file_uploader("데이터를 선택해주세요:", type=["csv", "xlsx"])
    if uploaded_file is not None:
        init_df()
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)

        # 업로드된 데이터 삽입
        st.session_state.df.iloc[:df.shape[0], :df.shape[1]] = df.values
        update_table()
        update_colname(df)
        st.success("데이터가 성공적으로 업로드되었습니다.")
        st.rerun()


@st.dialog(title='Login')
def login():
    if st.session_state.id != '' and st.session_state.name != '':
        st.warning('로그아웃 하시겠습니까?')
        if st.button('Log Out', use_container_width=True):
            st.session_state.id = ''
            st.session_state.name = ''
            
            with st.form(key='login_page'):
                id_ = st.text_input('ID')
                name = st.text_input('name')
                submitted = st.form_submit_button('Submit')
                if submitted:
                    st.session_state.id = id_
                    st.session_state.name = name
                    st.rerun()
    else:
        with st.form(key='login_page'):
            id = st.text_input('ID')
            name = st.text_input('name')
            submitted = st.form_submit_button('Submit')
            if submitted:
                st.session_state.id = id
                st.session_state.name = name
                folder = os.path.join(os.getcwd(), "DB", f"{st.session_state.id}({st.session_state.name})")
                if not os.path.exists(folder):
                    os.makedirs(folder)
                st.rerun()


def table():
    # 데이터 입력 데이터 및 분석 설정값 선언
    bodycol1, bodycol2 = st.columns([22, 7])
    
    # 데이터 입력 데이터
    with bodycol1:
        with st.container(border=True):
            tbcol1, tbcol2 = st.columns([2, 20])
        with tbcol1:
            def change_col_name(i):
                update_table()
                col = st.session_state.df.columns.tolist()
                col[i] = st.session_state[f'new_col_name_{i}']
                st.session_state.df.columns = col
            for i in range(len(st.session_state.col_name)):
                st.text_input('행', label_visibility='collapsed', 
                                value=st.session_state.col_name[i], 
                                key=f'new_col_name_{i}',
                                on_change=change_col_name,
                                args=(i,))
        with tbcol2:
            st.data_editor(st.session_state.df,
                            num_rows='dynamic',
                            hide_index=True,
                            use_container_width=True,
                            height=667,
                            key=f'change_df_data')
    return bodycol2
            

# def get_image_download_link(fig, filename="histogram.png", text="Download"):
#     buf = BytesIO()
#     fig.write_image(buf, format="png", scale=2, width=1000, height=600)
#     buf.seek(0)
#     b64 = base64.b64encode(buf.read()).decode()
#     href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
#     return href


def get_image_download_link(
    fig, 
    filename="plotly_graph.png", 
    text="이미지 다운로드", 
    tooltip="배경 제거한 투명 PNG"
    ):
    # 그래프 크기와 배경 설정
    fig.update_layout(
        autosize=False,
        width=800,  # 원하는 너비로 조정
        height=600,  # 원하는 높이로 조정
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=80, r=80, t=100, b=80)  # 마진을 늘려 축 라벨과 제목이 겹치지 않도록 함
    )
    
    # PNG 형식으로 Plotly 그래프 변환 (크기 지정)
    img_bytes = pio.to_image(fig, format="png", scale=2, width=800, height=600)
    
    # base64로 인코딩
    b64 = base64.b64encode(img_bytes).decode()
    
    button_uuid = f"download_button_{filename.replace('.', '_')}"
    custom_css = f"""
        <style>
            #{button_uuid} {{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.25rem 0.75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
                cursor: pointer;
            }}
            #{button_uuid}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_uuid}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
            }}
            .tooltip {{
                position: relative;
                display: inline-block;
            }}
            .tooltip .tooltiptext {{
                visibility: hidden;
                width: 200px;
                background-color: #555;
                color: #fff;
                text-align: center;
                border-radius: 6px;
                padding: 5px;
                position: absolute;
                z-index: 1;
                bottom: 125%;
                left: 50%;
                margin-left: -100px;
                opacity: 0;
                transition: opacity 0.3s;
            }}
            .tooltip:hover .tooltiptext {{
                visibility: visible;
                opacity: 1;
            }}
        </style> """
    
    dl_link = custom_css + f'''
        <div class="tooltip">
            <a download="{filename}" id="{button_uuid}" href="data:image/png;base64,{b64}">{text}</a>
            <span class="tooltiptext">{tooltip}</span>
        </div><br></br>
    '''
    
    return dl_link

###########################################################################################################################
# 색 조정
###########################################################################################################################
def generate_bright_colors(n):
        hue_partition = 1.0 / n
        colors = []
        for i in range(n):
            hue = i * hue_partition
            saturation = random.uniform(0.7, 1.0)
            value = random.uniform(0.7, 1.0)
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
            colors.append(hex_color)
        return colors

def adjust_color_lightness(color, amount=0.5):
    try:
        c = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        c = colorsys.rgb_to_hls(*[x/255.0 for x in c])
        return colorsys.hls_to_rgb(c[0], max(0, min(1, amount*c[1])), c[2])
    except ValueError:
        return (1, 1, 1)


def generate_color_scale(base_color):
    return [
        [int(255*r), int(255*g), int(255*b)]
        for r, g, b in [adjust_color_lightness(base_color, amount) for amount in np.linspace(0.3, 1.3, 10)]
    ]


def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])


def color_set(columns):
    color_intensity = st.slider("색상 강도", 1, 10, 5)
    BASE_COLORS = [
                "#87CEEB", "#FF9AA2", "#000000", "#4169E1", "#FDFD96",
                "#FFB347", "#C7A0D9", "#FF9CE3", "#B0C4DE", "#A6D1D4"]

    COLOR_SCALES = [generate_color_scale(color) for color in BASE_COLORS]
    st.session_state.colors = [COLOR_SCALES[i % len(COLOR_SCALES)][color_intensity-1] for i in range(len(columns))]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col6, col7, col8, col9, col10 = st.columns(5)

    for i in range(len(columns)):
        if i == 0:
            col1.color_picker(f"{columns[i]}", 
                            rgb_to_hex(st.session_state.colors[i]), 
                            key=f"color_{i}")
        elif i == 1:
            col2.color_picker(f"{columns[i]}", rgb_to_hex(st.session_state.colors[i]), key=f"color_{i}")
        elif i == 2:
            col3.color_picker(f"{columns[i]}", rgb_to_hex(st.session_state.colors[i]), key=f"color_{i}")
        elif i == 3:
            col4.color_picker(f"{columns[i]}", rgb_to_hex(st.session_state.colors[i]), key=f"color_{i}")
        elif i == 4:
            col5.color_picker(f"{columns[i]}", rgb_to_hex(st.session_state.colors[i]), key=f"color_{i}")
        elif i == 5:
            col6.color_picker(f"{columns[i]}", rgb_to_hex(st.session_state.colors[i]), key=f"color_{i}")
        elif i == 6:
            col7.color_picker(f"{columns[i]}", rgb_to_hex(st.session_state.colors[i]), key=f"color_{i}")
        elif i == 7:
            col8.color_picker(f"{columns[i]}", rgb_to_hex(st.session_state.colors[i]), key=f"color_{i}")
        elif i == 8:
            col9.color_picker(f"{columns[i]}", rgb_to_hex(st.session_state.colors[i]), key=f"color_{i}")
        elif i == 9:
            col10.color_picker(f"{columns[i]}", rgb_to_hex(st.session_state.colors[i]), key=f"color_{i}")

def axis_set(title, x_label, y_label):
    st.session_state.title = st.text_input("제목", title)
    st.session_state.x_label = st.text_input("X축 제목", x_label)
    st.session_state.y_label = st.text_input("Y축 제목", y_label)