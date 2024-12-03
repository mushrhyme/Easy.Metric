from utils.tools.UTIL import *
import streamlit as st
import numpy as np
import pandas as pd

from scipy import stats
import statsmodels.api as sm

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def normality_test_set():
    df = convert_to_calculatable_df()
    st.session_state.target = st.selectbox("변수", df.columns.tolist())
    st.session_state.significance_level = st.selectbox("유의수준", [0.05, 0.01, 0.1])
    
def normality_test_cal(df):
    # Anderson-Darling 검정
    ad_statistic, ad_critical_values, ad_significance_level = stats.anderson(df)
    ad_critical_value = ad_critical_values[np.where(ad_significance_level==st.session_state.significance_level*100)][0]   
    # Shapiro-Wilk 검정 (Ryan-Joiner 대신 사용)
    sw_statistic, sw_p_value = stats.shapiro(df)
    
    # Kolmogorov-Smirnov 검정
    ks_statistic, ks_p_value = stats.kstest(df, 'norm', args=(np.mean(df), np.std(df)))
    
    st.session_state.stats_df = pd.DataFrame({
        'Anderson-Darling 검정': [
            np.round(ad_statistic, 4), 
            np.round(ad_critical_value, 3), 
            '기각' if ad_statistic > ad_critical_value else '기각 X'],
        'Shapiro-Wilk 검정': [
            np.round(sw_statistic, 4), 
            np.round(sw_p_value, 3), 
            '기각' if sw_p_value < st.session_state.significance_level else '기각 X'],
        'Kolmogorov-Smirnov 검정': [
            np.round(ks_statistic, 4),
            np.round(ks_p_value, 3), 
            '기각' if ks_p_value < st.session_state.significance_level else '기각 X'], 
    }, index=["통계량", "P-값", "결론"])
    
def normality_test_plot(df):
    fig = go.Figure()
    
    # 데이터 정렬 및 이론적 분위수 계산
    sorted_df = np.sort(df)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(df)))
    
    # 산점도 추가
    fig.add_trace(go.Scatter(x=theoretical_quantiles, y=sorted_df, mode='markers', showlegend=False))
    
    # 기준선 추가
    min_val = min(theoretical_quantiles.min(), sorted_df.min())
    max_val = max(theoretical_quantiles.max(), sorted_df.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val], 
            y=[min_val, max_val], 
            mode='lines', 
            showlegend=False,
            line=dict(color='red')
        )
    )
   
    stats_text = f"평균: {np.mean(df):.3f}, 표준편차: {np.std(df):.3f}, N: {len(df)}"
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=1.05,
        showarrow=False,
        text=stats_text,
        font=dict(size=12),
        # bordercolor="black",
        # borderwidth=1,
        # borderpad=4,
        bgcolor="white",
        opacity=0.8
    )
    
    # 레이아웃 설정
    fig.update_layout(
        title=f'{st.session_state.target} 확률도',
        xaxis_title=st.session_state.target,
        yaxis_title='백분율',
        showlegend=True
    )
    st.plotly_chart(fig)

def normality_test_run():  
    df = convert_to_calculatable_df()[st.session_state.target]
    # --------------------------------------
    normality_test_cal(df)
    st.write("**정규성 검정**")
    st.write("귀무가설: 데이터가 정규 분포를 따릅니다.")
    st.write("대립가설: 데이터가 정규 분포를 따르지 않습니다.")
    st.write(f"유의수준: α={st.session_state.significance_level}")
    st.data_editor(st.session_state.stats_df, key="stats", use_container_width=True)
    # --------------------------------------
    with st.container(border=True):
        normality_test_plot(df)