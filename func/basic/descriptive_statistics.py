from utils.tools.UTIL import *
from scipy import stats
from plotly.subplots import make_subplots

import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go


def descriptive_statistics_set():
    col = st.session_state.df.columns.tolist()
    st.selectbox(f"변수 선택", options=col, index=0, key='descriptive_statistics_col')
    st.multiselect('분석항목 선택',
                    options=['평균', '평균의 표준오차', '표준편차', '분산', '변동계수',
                            '합', '최소값', '최대값', '최빈값', '범위', 
                            '왜도', '첨도'],
                    default=['평균', '표준편차', '최소값', '최대값'],
                    key='descriptive_statistics_selected'
                    )
    st.divider()
    st.checkbox('히스토그램 그리기', key='histogram')
    st.checkbox('상자그림 그리기', key='boxplot')

def descriptive_statistics_cal(df):
    col = st.session_state.descriptive_statistics_col
    try:
        print_txts = []
        if '평균' in st.session_state.descriptive_statistics_selected:
            print_txts.append(f'평균 : {round(df[col].mean(), 3)}')
        if '평균의 표준오차' in st.session_state.descriptive_statistics_selected:
            print_txts.append(f'평균의 표준오차 : {round(df[col].sem(), 3)}')
        if '표준편차' in st.session_state.descriptive_statistics_selected:
            print_txts.append(f'표준편차 : {round(df[col].std(), 3)}')
        if '분산' in st.session_state.descriptive_statistics_selected:
            print_txts.append(f'분산 : {round(df[col].var(), 3)}')
        if '변동계수' in st.session_state.descriptive_statistics_selected:
            print_txts.append(f'변동계수 : {round(df[col].std() / df[col].mean(), 3)}')
        if '합' in st.session_state.descriptive_statistics_selected:
            print_txts.append(f'합 : {round(df[col].sum(), 3)}')
        if '최소값' in st.session_state.descriptive_statistics_selected:
            print_txts.append(f'최소값 : {round(df[col].min(), 3)}')
        if '최대값' in st.session_state.descriptive_statistics_selected:
            print_txts.append(f'최대값 : {round(df[col].max(), 3)}')
        if '최빈값' in st.session_state.descriptive_statistics_selected:
            print_txts.append(f'최빈값 : {round(df[col].mode()[0], 3)}')
        if '범위' in st.session_state.descriptive_statistics_selected:
            print_txts.append(f'범위 : {round(df[col].max() - df[col].min(), 3)}')
        if '왜도' in st.session_state.descriptive_statistics_selected:
            print_txts.append(f'왜도 : {round(df[col].skew(), 3)}')
        if '첨도' in st.session_state.descriptive_statistics_selected:
            print_txts.append(f'첨도 : {round(df[col].kurtosis(), 3)}')

        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 1, 1])
            for i, txt in enumerate(print_txts[:4]):
                c1.markdown(f'{i+1}. {txt}')
            for i, txt in enumerate(print_txts[4:8]):
                c2.markdown(f'{i+5}. {txt}')
            for i, txt in enumerate(print_txts[8:]):
                c3.markdown(f'{i+9}. {txt}')
    except TypeError:
        st.error(f"'{col}' 열에 숫자가 아닌 데이터가 포함되어 있어 평균을 계산할 수 없습니다. 데이터를 확인해주세요.")
    except Exception as e:
        st.error(f"평균 계산 중 오류가 발생했습니다: {str(e)}")
            
def descriptive_statistics_plot(df):
    with st.container(border=True):
        _, c, _ = st.columns([1, 12, 1])
        if st.session_state.histogram:
            column = st.session_state.descriptive_statistics_col
            fig = px.histogram(df, x=column, height=350, nbins=18)

            fig.update_traces(selector=dict(type='histogram'), 
                            marker=dict(line=dict(width=3, color='white')))

            mean, std = df[column].mean(), df[column].std()
            x = np.linspace(mean - 4*std, mean + 4*std, 200)
            y = stats.norm.pdf(x, mean, std)

            # 정규 분포 곡선 추가 (두 번째 y축 사용)
            fig.add_trace(go.Scatter(
                x=x, 
                y=y, 
                mode='lines', 
                name='Normal Distribution',
                line=dict(color='red', dash='solid'),
                yaxis='y2'
            ))

            fig.update_layout(
                title=f'Histogram of {column} with Normal Distribution',
                xaxis_title=column,
                yaxis_title='Frequency',
                yaxis2=dict(
                    title='Probability Density',
                    overlaying='y',
                    side='right',
                    range=[0, 1.1 * max(y)],  # y축 범위를 0에서 최대값보다 약간 위로 설정
                    showgrid=False
                ),
                xaxis=dict(range=[mean - 4*std, mean + 4*std], nticks=20),
                yaxis=dict(rangemode='nonnegative'),
                showlegend=False,
                legend=dict(x=1.1, y=1)  # 범례 위치 조정
            )

            c.plotly_chart(fig)
            
        if st.session_state.boxplot:
            column = st.session_state.descriptive_statistics_col
            fig = px.box(df, y=column, height=350)
            fig.update_traces(marker=dict(color='rgb(8,81,156)', 
                                        outliercolor='red', 
                                        line=dict(outliercolor='red')))
            fig.update_layout(title=f'Boxplot of {column}')
            c.plotly_chart(fig)

def descriptive_statistics_run():
    # 매우 중요 : 데이터프레임 가져오기 ------------
    df = convert_to_calculatable_df()
    # --------------------------------------
    try:
        col_type = df[st.session_state.descriptive_statistics_col].dtype
        if col_type == "object":
            st.error(f"{st.session_state.descriptive_statistics_col}는 수치형 변수가 아닙니다. 수치형 변수만 선택해주세요.")
        else:
            descriptive_statistics_cal(df)
            descriptive_statistics_plot(df)
    except:
        st.error(f"분석에 사용할 데이터가 없습니다. 올바른 변수를 선택해주세요.")


