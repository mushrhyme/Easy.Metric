# 주효과분석을 위한 기능 코드
from utils.tools.UTIL import *
import streamlit as st
import numpy as np
import pandas as pd

from scipy import stats
import statsmodels.api as sm
import math

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


###########################################################################################################################
def main_effect_analysis_set():
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
        if len(st.session_state.predictor)>0:
            max_col = np.max([len(df[x].unique()) for x in st.session_state.predictor])
            color_set([f"색상{i}" for i in range(1, max_col+1)])
        
def main_effect_analysis_cal(df):
    pass

def main_effect_analysis_plot(df):
    n_rows = math.ceil(len(st.session_state.predictor) / 3)  # 3열로 변경
    n_cols = min(len(st.session_state.predictor), 3)  # 최대 3열로 변경
    
    fig = make_subplots(rows=n_rows, cols=n_cols, 
                        subplot_titles=[f'{factor}' for factor in st.session_state.predictor],
                        shared_yaxes=True,
                        horizontal_spacing=0.01,
                        vertical_spacing=0.2)
    
    overall_mean = df[st.session_state.target].mean()
    y_min = df[st.session_state.target].min()
    y_max = df[st.session_state.target].max()
    
    for i, factor in enumerate(st.session_state.predictor, 1):
        row = (i - 1) // 3 + 1  # 3열 기준으로 행 계산
        col = ((i - 1) % 3) + 1  # 3열 기준으로 열 계산
        
        means = df.groupby(factor)[st.session_state.target].mean().reset_index()     
        color_index = (i - 1) % len(st.session_state.predictor)
        fig.add_trace(
            go.Scatter(
                x=means[factor], 
                y=means[st.session_state.target], 
                mode='lines+markers',
                name=factor,
                line=dict(color=st.session_state[f"color_{color_index}"]),
                marker=dict(color=st.session_state[f"color_{color_index}"], size=10),
            ), row=row, col=col
        )
        
        x_range = df[factor].unique()
        x_padding = 0.5
        x_min, x_max = -x_padding, len(x_range) - 1 + x_padding
        
        fig.update_xaxes(title_text=factor, row=row, col=col, 
                         tickvals=list(range(len(x_range))), 
                         ticktext=x_range,
                         range=[x_min, x_max])
        
        fig.update_yaxes(title_text=st.session_state.target if col == 1 else '',
                         row=row, col=col,
                         range=[y_min - 1, y_max + 1])
        
        fig.add_shape(
            type="line",
            x0=x_min, x1=x_max,
            y0=overall_mean, y1=overall_mean,
            line=dict(color="gray", width=2, dash="dash"),
            row=row, col=col
        )
    
    subplot_height = 400
    total_height = subplot_height * n_rows + 100
    fig.update_layout(
        height=total_height, 
        width=1200,  # 3열 레이아웃에 맞게 너비 증가
        title_text=f'{st.session_state.target}의 주효과도',
        title_y=0.98,
        margin=dict(t=120),
        showlegend=False
    )
    
    fig.add_annotation(
        x=0.5, y=0.99,
        text=f"전체 평균: {overall_mean:.2f}",
        showarrow=False,
        xref="paper", yref="paper"
    )
    st.plotly_chart(fig)
    
def main_effect_analysis_run():  
    df = convert_to_calculatable_df()
    # --------------------------------------
    if len(st.session_state.predictor) > 0:
        main_effect_analysis_cal(df)
        # --------------------------------------
        with st.container(border=True):
            main_effect_analysis_plot(df)
    else:
        st.error("분석에 사용할 요인이 지정되지 않았습니다.")