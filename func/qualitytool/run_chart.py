# 런차트를 위한 기능 코드

from utils.tools.UTIL import *
from scipy import stats
from plotly.subplots import make_subplots

import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go


def run_chart_set():
    df = convert_to_calculatable_df()

    st.session_state.uniq_col = st.selectbox("단일 열", df.columns.tolist())
    st.session_state.subgroup_size = st.number_input("부분군 크기", value=1)

def run_chart_cal(df):
    pass

def run_chart_plot(df):
    data = df[st.session_state.uniq_col].values
    n = len(data)
    
    # 전체 평균선 계산
    overall_mean = np.mean(data)
    
    # 부분군 수 계산
    num_subgroups = n // st.session_state.subgroup_size
    
    # 부분군별 데이터 및 평균 계산
    subgroup_means = []
    scatter_x = []  # 산점도용 x 좌표
    scatter_y = []  # 산점도용 y 좌표
    
    for i in range(num_subgroups):
        start_idx = i * st.session_state.subgroup_size
        end_idx = start_idx + st.session_state.subgroup_size
        subgroup_data = data[start_idx:end_idx]
        
        # 부분군 평균 계산
        subgroup_means.append(np.mean(subgroup_data))
        
        # 각 데이터 포인트를 동일한 x 좌표(부분군 번호)에 수직으로 배치
        for value in subgroup_data:
            scatter_x.append(i + 1)  # 1부터 시작하는 부분군 번호
            scatter_y.append(value)
    
    # Plotly 그래프 생성
    fig = go.Figure()
    
    # 개별 데이터 포인트 추가 (산점도)
    fig.add_trace(go.Scatter(
        x=scatter_x,
        y=scatter_y,
        mode='markers',
        name='개별 데이터',
        marker=dict(size=8, color='blue', opacity=0.6)
    ))
    
    # 부분군 평균선 추가
    fig.add_trace(go.Scatter(
        x=list(range(1, num_subgroups + 1)),  # 1부터 시작하는 부분군 번호
        y=subgroup_means,
        mode='lines+markers',
        name='부분군 평균',
        line=dict(color='red', width=2),
        marker=dict(size=10, color='red')
    ))
    
    # 전체 평균선 추가
    fig.add_hline(
        y=overall_mean,
        line_dash="dash",
        line_color="green",
        annotation_text="전체 평균",
        annotation_position="bottom right"
    )
    
    # 그래프 레이아웃 설정
    fig.update_layout(
        title=f'Run Chart - {st.session_state.uniq_col}',
        xaxis=dict(
            title='부분군 번호',
            tickmode='linear',
            tick0=1,
            dtick=1,  # 1단위로 눈금 표시
            range=[0.5, num_subgroups + 0.5]  # x축 범위 조정
        ),
        yaxis_title=st.session_state.uniq_col,
        showlegend=True,
        hovermode='closest',
        template='simple_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def run_chart_run():  
    # 매우 중요 : 데이터프레임 가져오기 ------------
    df = convert_to_calculatable_df()
    # --------------------------------------
    
    # run_chart_cal(df)
    run_chart_plot(df)