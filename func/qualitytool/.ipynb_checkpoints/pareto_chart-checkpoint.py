# 파레토차트를 위한 기능 코드

from utils.tools.UTIL import *
from scipy import stats
from plotly.subplots import make_subplots

import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go


def pareto_chart_set():
    df = convert_to_calculatable_df()
    st.session_state.pareto_col = st.selectbox("분석할 변수", df.columns.tolist())
    st.session_state.min_percent = st.number_input(
        "나머지 결점 결합 기준 백분율",
        value=5.0,
        help="이 백분율 미만의 범주들은 '기타'로 결합됩니다."
    )

def transform_data_with_others(df, column, min_percent):
    # 원본 데이터프레임 복사
    df_copy = df.copy()
    
    # 빈도 및 백분율 계산
    value_counts = df_copy[column].value_counts()
    total_count = len(df_copy)
    percentages = (value_counts / total_count) * 100
    
    # 기준 백분율 미만인 카테고리들 식별
    small_categories = percentages[percentages < min_percent].index
    
    # 해당 카테고리들을 '기타'로 변경
    df_copy.loc[df_copy[column].isin(small_categories), column] = '기타'
    
    return df_copy

def pareto_chart_set():
    df = convert_to_calculatable_df()
    st.session_state.pareto_col = st.selectbox("분석할 변수", df.columns.tolist())
    st.session_state.min_percent = st.slider(
        "나머지 결점 결합 기준 백분율",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1,
        help="이 백분율 미만의 범주들은 '기타'로 결합됩니다."
    )

def pareto_chart_cal(df):
    # 원본 데이터의 빈도 계산
    value_counts = df.groupby(st.session_state.pareto_col).size()
    total_count = len(df)
    
    # 백분율 계산
    percentages = (value_counts / total_count) * 100
    st.write(percentages)
    # 기준값 미만 카테고리 식별
    small_categories = percentages[percentages < st.session_state.min_percent]
    large_categories = percentages[percentages >= st.session_state.min_percent]
    st.write(small_categories)
    # 결과 시리즈 생성
    result = large_categories.copy()
    if len(small_categories) > 0:
        result['기타'] = small_categories.sum()
    
    # 내림차순 정렬 (기타는 제외)
    if '기타' in result:
        others_value = result['기타']
        result = result[result.index != '기타'].sort_values(ascending=False)
        result['기타'] = others_value
    else:
        result = result.sort_values(ascending=False)
    
    # 빈도로 변환
    counts = (result / 100) * total_count
    counts = counts.round().astype(int)
    # st.write(result, counts)
    # 누적 백분율 계산
    cumulative = np.cumsum(counts) / total_count * 100
    
    return counts, cumulative

def pareto_chart_set():
    df = convert_to_calculatable_df()
    st.session_state.pareto_col = st.selectbox("분석할 변수", df.columns.tolist())
    st.session_state.min_percent = st.slider(
        "나머지 결점 결합 기준 백분율",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1,
        help="이 백분율 미만의 범주들은 '기타'로 결합됩니다."
    )

def pareto_chart_cal(df):
    # 원본 데이터의 빈도 계산
    value_counts = df.groupby(st.session_state.pareto_col).size()
    total_count = len(df)
    
    # 백분율 계산
    percentages = (value_counts / total_count) * 100
    
    # 기준값 미만 카테고리 식별
    small_categories = percentages[percentages < st.session_state.min_percent]
    large_categories = percentages[percentages >= st.session_state.min_percent]
    
    # 결과 시리즈 생성
    result = large_categories.copy()
    if len(small_categories) > 0:
        result['기타'] = small_categories.sum()
    
    # 내림차순 정렬 (기타는 제외)
    if '기타' in result:
        others_value = result['기타']
        result = result[result.index != '기타'].sort_values(ascending=False)
        result['기타'] = others_value
    else:
        result = result.sort_values(ascending=False)
    
    # 빈도로 변환
    counts = (result / 100) * total_count
    counts = counts.round().astype(int)
    
    # 누적 백분율 계산
    cumulative = np.cumsum(counts) / total_count * 100
    
    return counts, cumulative


def pareto_chart_plot(df):
    # 데이터 계산
    counts, cumulative = pareto_chart_cal(df)
    # 백분율 계산
    percentages = (counts / counts.sum()) * 100
    
    # 서브플롯 생성
    fig = make_subplots(specs=[[{"secondary_y": True}]], subplot_titles=[f"{st.session_state.pareto_col}의 Pareto 차트"])
    
    # 데이터 준비
    x_values = list(counts.index)
    y_values = list(counts.values)
    
    # 색상 팔레트 설정
    main_color = 'rgba(66, 133, 244, 0.7)'
    others_color = 'rgba(189, 189, 189, 0.7)'
    colors = [main_color if x != '기타' else others_color for x in x_values]
    
    # 막대 차트 추가
    fig.add_trace(
        go.Bar(
            x=x_values,
            y=y_values,
            name="빈도",
            marker=dict(
                color=colors,
                line=dict(
                    color='rgba(58, 62, 65, 0.3)',
                    width=1.5
                )
            ),
            hovertemplate="<b>%{x}</b><br>" +
                         "Count: %{y:,}<br>" +
                         "<extra></extra>"
        ),
        secondary_y=False
    )
    
    # 막대 아래 텍스트 추가 - 위치 조정
    for i, (count, perc, cum) in enumerate(zip(counts, percentages, cumulative)):
        # x 위치 계산 - 정확한 위치를 위해 인덱스 사용
        x_pos = i
        
        # Count 텍스트
        fig.add_annotation(
            x=x_pos,
            y=0,
            text=f'<b>{count:,}</b>',
            showarrow=False,
            yshift=-30,
            font=dict(size=11),
            xanchor='center',  # 중앙 정렬
            yanchor='top',
            xref='x',  # x축 참조 명시
            yref='y'   # y축 참조 명시
        )
        
        # Percentage 텍스트
        fig.add_annotation(
            x=x_pos,
            y=0,
            text=f'{perc:.1f}%',
            showarrow=False,
            yshift=-50,
            font=dict(size=11),
            xanchor='center',  # 중앙 정렬
            yanchor='top',
            xref='x',
            yref='y'
        )
        
        # Cumulative 텍스트
        fig.add_annotation(
            x=x_pos,
            y=0,
            text=f'누적 {cum:.1f}%',
            showarrow=False,
            yshift=-70,
            font=dict(size=11),
            xanchor='center',  # 중앙 정렬
            yanchor='top',
            xref='x',
            yref='y'
        )
    
    # 누적 백분율 선 추가
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=cumulative,
            name="누적 백분율",
            line=dict(
                color='rgb(255, 87, 34)',
                width=3,
                shape='spline'
            ),
            mode='lines+markers',
            marker=dict(
                size=8,
                symbol='circle',
                color='rgb(255, 87, 34)',
                line=dict(
                    color='white',
                    width=2
                )
            ),
            text=[f'{val:.1f}%' for val in cumulative],
            textposition='top center',
            hovertemplate="<b>%{x}</b><br>" +
                         "Cumulative: %{y:.1f}%<br>" +
                         "<extra></extra>"
        ),
        secondary_y=True
    )
    
    # 80% 기준선 추가
    fig.add_hline(
        y=80,
        line=dict(
            color='rgba(156, 39, 176, 0.3)',
            width=2,
            dash='dash'
        ),
        secondary_y=True
    )
    
    # 레이아웃 설정
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            xanchor='center',
            yanchor='top',
            y=1.02,
            x=1
        ),
        template='plotly_white',
        bargap=0.3,
        xaxis=dict(
            title=dict(
                text="",
                font=dict(size=14)
            ),
            type='category',
            tickfont=dict(size=12)
        ),
        margin=dict(t=120, b=120, l=60, r=60),  # 하단 여백 증가
        plot_bgcolor='white',
        height=700
    )
    
    # y축 설정
    max_y = max(y_values)
    fig.update_yaxes(
        title=dict(
            text="빈도",
            font=dict(size=14)
        ),
        tickfont=dict(size=12),
        gridcolor='rgba(0,0,0,0.1)',
        secondary_y=False,
        showgrid=True,
        range=[0, max_y * 1.1]
    )
    
    fig.update_yaxes(
        title=dict(
            text="누적 백분율 (%)",
            font=dict(size=14)
        ),
        tickformat='.1f',
        range=[0, 105],
        tickfont=dict(size=12),
        secondary_y=True,
        showgrid=False
    )
    
    # 차트 표시
    st.plotly_chart(fig, use_container_width=True)
    

def pareto_chart_run():
    # 데이터프레임 가져오기
    df = convert_to_calculatable_df()
    
    # 파레토 차트 표시
    pareto_chart_plot(df)