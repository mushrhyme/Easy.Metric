# 교호작용분석을 그리기 위한 기능 코드
from utils.tools.UTIL import *
import streamlit as st
import numpy as np
import pandas as pd

from scipy import stats
import statsmodels.api as sm
import math
import itertools

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

###########################################################################################################################
def interaction_analysis_set():
    tab1, tab2 = st.tabs(["분석", "그래프"])
    with tab1:
        df = convert_to_calculatable_df()
        st.session_state.target = st.selectbox("반응", df.columns.tolist())
        predictor = df.columns.tolist()
        predictor.remove(st.session_state.target)
        st.session_state.predictor = st.multiselect("요인", predictor)

        # 반응: 계량형 변수 확인
        if df[st.session_state.target].dtype=="object":
            st.error(f"{st.session_state.target}는 계량형 변수가 아닙니다. 계량형 변수만 선택해주세요.")
            return
        else:
            categorical_predictors = [col for col in st.session_state.predictor if df[col].dtype=="object"]
            non_categorical = [pred for pred in st.session_state.predictor if pred not in categorical_predictors]
            # 요인: 범주형 변수 확인
            if non_categorical:
                st.error(f"{', '.join(non_categorical)}는 범주형 변수가 아닙니다. 범주형 변수만 선택해주세요.")
                return
    with tab2:
        max_col = np.max([len(df[x].unique()) for x in st.session_state.predictor])
        color_set([f"색상{i}" for i in range(1, max_col+1)])

def interaction_analysis_cal(df):
    pass


def interaction_analysis_plot(df):
    if len(st.session_state.predictor) == 1:
        # 단일 요인일 경우 코드는 동일
        factor = st.session_state.predictor[0]
        fig = go.Figure()
        means = df.groupby(factor)[st.session_state.target].mean().reset_index()
        
        for i, level in enumerate(means[factor]):
            fig.add_trace(
                go.Scatter(
                    x=[level],
                    y=[means.loc[means[factor] == level, st.session_state.target].iloc[0]],
                    mode='lines+markers',
                    name=str(level),
                    line=dict(color=st.session_state[f"color_{i}"]),
                    marker=dict(color=st.session_state[f"color_{i}"], size=10),
                    showlegend=True
                )
            )
        
        fig.update_layout(
            height=600,
            width=800,
            title_text=f'{st.session_state.target}의 {factor}에 따른 변화',
            xaxis_title=factor,
            yaxis_title=st.session_state.target,
            showlegend=True,
            legend_title_text=factor,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05
            )
        )
    else:
        factor_combinations = list(itertools.combinations(st.session_state.predictor, 2))
        n_combinations = len(factor_combinations)
        
        n_rows = math.ceil(n_combinations / 2)
        n_cols = min(n_combinations, 2)
        
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols, 
            subplot_titles=[f'{factor1} vs {factor2}' for factor1, factor2 in factor_combinations],
            shared_yaxes=True,
            horizontal_spacing=0.15,
            vertical_spacing=0.2
        )
        
        y_min = df[st.session_state.target].min()
        y_max = df[st.session_state.target].max()
        
        # 범례 위치를 저장할 리스트
        legend_traces = [[] for _ in range(n_rows)]
        
        for row in range(1, n_rows + 1):
            for col in range(1, n_cols + 1):
                combination_index = (row - 1) * n_cols + (col - 1)
                if combination_index >= len(factor_combinations):
                    continue
                    
                factor1, factor2 = factor_combinations[combination_index]
                interaction_means = df.groupby([factor1, factor2])[st.session_state.target].mean().reset_index()
                
                for j, level in enumerate(sorted(df[factor1].unique())):
                    data = interaction_means[interaction_means[factor1] == level]
                    
                    # 메인 트레이스 (그래프)
                    trace = go.Scatter(
                        x=data[factor2], 
                        y=data[st.session_state.target], 
                        mode='lines+markers',
                        name=f'{factor1}={level}',
                        line=dict(color=st.session_state[f"color_{j}"]),
                        marker=dict(color=st.session_state[f"color_{j}"], size=8),
                        showlegend=False  # 메인 트레이스의 범례는 숨김
                    )
                    fig.add_trace(trace, row=row, col=col)
                    
                    # 범례용 트레이스 (첫 번째 열에만 추가)
                    if col == 1:
                        legend_trace = go.Scatter(
                            x=[None],
                            y=[None],
                            mode='lines+markers',
                            name=f'{factor1}={level}',
                            line=dict(color=st.session_state[f"color_{j}"]),
                            marker=dict(color=st.session_state[f"color_{j}"], size=8),
                            showlegend=True,
                            legendgroup=f"row_{row}",
                            legendgrouptitle_text=factor1
                        )
                        legend_traces[row-1].append(legend_trace)
                
                x_range = sorted(df[factor2].unique())
                x_padding = 0.5
                x_min, x_max = -x_padding, len(x_range) - 1 + x_padding
                
                fig.update_xaxes(
                    title_text=factor2, 
                    row=row, 
                    col=col, 
                    tickvals=list(range(len(x_range))), 
                    ticktext=x_range,
                    range=[x_min, x_max]
                )
                
                fig.update_yaxes(
                    title_text=st.session_state.target if col == 1 else '',
                    row=row, 
                    col=col,
                    range=[y_min - 1, y_max + 1]
                )
        
        subplot_height = 300
        total_height = subplot_height * n_rows
        
        # 범례용 트레이스 추가
        for row, row_traces in enumerate(legend_traces):
            y_position = 0.9 - (row * (1/n_rows))  # 각 행별 범례 위치
            for trace in row_traces:
                fig.add_trace(trace)
                # 각 행의 범례 위치 설정
                if trace['showlegend']:
                    trace.update(
                        legendrank=row * 1000,  # 범례 순서 보장
                    )
        
        # 레이아웃 업데이트
        fig.update_layout(
            height=total_height, 
            width=1000,
            title_text=f'{st.session_state.target}의 교호작용도',
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.95,
                xanchor="left",
                x=1.02,
                itemsizing='constant',
                tracegroupgap=240,  # 범례 그룹 간격 증가
                groupclick="toggleitem",
                font=dict(size=10)
            ),
            margin=dict(r=100)
        )
    
    st.plotly_chart(fig)
    
def interaction_analysis_run():  
    df = convert_to_calculatable_df()
    # --------------------------------------
    if len(st.session_state.predictor) > 0:
        interaction_analysis_cal(df)
        # --------------------------------------
        with st.container(border=True):
            interaction_analysis_plot(df)
    else:
        st.error("분석에 사용할 변수가 지정되지 않았습니다.")
