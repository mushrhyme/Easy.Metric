# 쌍체 t검정을 위한 기능 선언 코드

from utils.tools.UTIL import *
from scipy import stats
from plotly.subplots import make_subplots

import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go


def paired_ttest_set():
    df = convert_to_calculatable_df()
    col = df.columns.tolist()
    st.info('두 표본의 데이터 개수는 동일해야 합니다.')
    st.session_state.sample1 = st.selectbox(f"표본1", options=col, index=0)
    st.session_state.sample2 = st.selectbox(f"표본2", options=col, index=1)
    st.session_state.significance_level = 1 - st.number_input(f"신뢰수준", value=95.0, step=1.0) / 100.0
    
    st.session_state.alternative = st.selectbox(f"대립 가설", ["차이 < 귀무가설에서의 차이", "차이 ≠ 귀무가설에서의 차이", "차이 > 귀무가설에서의 차이"])

def paired_ttest_cal(data1, data2, confidence_level, alternative):
    # 데이터 준비
    data1 = pd.Series(data1).dropna()
    data2 = pd.Series(data2).dropna()
    diff = data1 - data2
    
    # 기본 통계량 계산
    n = len(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    se_diff = std_diff / np.sqrt(n)
    
    # 대립가설에 따른 alternative 설정
    if alternative == "차이 < 귀무가설에서의 차이":
        alt = 'less'
    elif alternative == "차이 > 귀무가설에서의 차이":
        alt = 'greater'
    else:
        alt = 'two-sided'
    
    # T 검정 수행
    t_stat, p_value = stats.ttest_rel(data1, data2, alternative=alt)
    
    # 신뢰구간 계산
    df = n - 1
    confidence_level = 1 - st.session_state.significance_level
    if alt == 'two-sided':
        ci = stats.t.interval(confidence_level, df, loc=mean_diff, scale=se_diff)
    elif alt == 'less':
        ci = (-np.inf, stats.t.ppf(confidence_level, df, loc=mean_diff, scale=se_diff))
    else:  # 'greater'
        ci = (stats.t.ppf(1 - confidence_level, df, loc=mean_diff, scale=se_diff), np.inf)
    
    return {
        'n': n,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'se_diff': se_diff,
        't_stat': t_stat,
        'p_value': p_value,
        'ci': ci
    }

def paired_ttest_plot(results, data1, data2, sample1_name, sample2_name):
    diff = data1 - data2

    # 정규화 곡선 그래프
    fig1 = go.Figure()

    x_norm = np.linspace(results['mean_diff'] - 4*results['std_diff'], 
                        results['mean_diff'] + 4*results['std_diff'], 100)
    y_norm = stats.norm.pdf(x_norm, results['mean_diff'], results['std_diff'])

    fig1.add_trace(go.Scatter(x=x_norm, y=y_norm, mode='lines', name='정규 분포'))

    # 평균 차이 선
    fig1.add_vline(x=results['mean_diff'], line_dash="solid", line_color="red", annotation_text="평균 차이", annotation_position="top right")

    # 신뢰구간 선
    if np.isfinite(results['ci'][0]):
        fig1.add_vline(x=results['ci'][0], line_dash="dash", line_color="green", annotation_text="하한 CI", annotation_position="bottom left")
    if np.isfinite(results['ci'][1]):
        fig1.add_vline(x=results['ci'][1], line_dash="dash", line_color="green", annotation_text="상한 CI", annotation_position="bottom right")

    # 데이터1과 데이터2의 히스토그램 및 정규분포 곡선
    fig2 = go.Figure()

    # 데이터1 히스토그램 (빈도로 표시)
    hist1 = go.Histogram(x=data1, name=f"{sample1_name} 히스토그램", opacity=0.7, nbinsx=10)
    fig2.add_trace(hist1)

    # 데이터2 히스토그램 (빈도로 표시)
    hist2 = go.Histogram(x=data2, name=f"{sample2_name} 히스토그램", opacity=0.7, nbinsx=10)
    fig2.add_trace(hist2)

    # 데이터1 정규분포 곡선
    x1 = np.linspace(min(data1), max(data1), 100)
    y1 = stats.norm.pdf(x1, np.mean(data1), np.std(data1))
    fig2.add_trace(go.Scatter(x=x1, y=y1, mode='lines', 
                            name=f'{sample1_name} 정규분포', line=dict(color='red'),
                            yaxis='y2'))

    # 데이터2 정규분포 곡선
    x2 = np.linspace(min(data2), max(data2), 100)
    y2 = stats.norm.pdf(x2, np.mean(data2), np.std(data2))
    fig2.add_trace(go.Scatter(x=x2, y=y2, mode='lines', 
                            name=f'{sample2_name} 정규분포', line=dict(color='blue'),
                            yaxis='y2'))

    # 평균선 추가
    fig2.add_vline(x=np.mean(data1), line_dash="dash", line_color="red", annotation_text=f"{sample1_name} 평균", annotation_position="top right")
    fig2.add_vline(x=np.mean(data2), line_dash="dash", line_color="blue", annotation_text=f"{sample2_name} 평균", annotation_position="top left")

    # y축 범위 설정
    y_max_hist = max(max(hist1.y or [0]), max(hist2.y or [0]))
    y_max_norm = max(max(y1), max(y2)) * 1.1
    
    # data1, data2 에서 최빈값 찾기
    mode1 = data1.mode().values[0]
    mode2 = data2.mode().values[0]
    y_max_hist = max(mode1, mode2) * 1.2

    fig2.update_layout(
        title="쌍체 T 검정: 데이터 분포",
        xaxis_title="값",
        yaxis=dict(title="빈도", range=[0, y_max_hist]),
        yaxis2=dict(title="밀도", overlaying='y', side='right', range=[0, 1], showgrid=False),
        barmode='group',
        
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # 그래프 표시
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)

def paired_ttest_run():  
    df = convert_to_calculatable_df()
    # --------------------------------------
    data1 = df[st.session_state.sample1].dropna()
    data2 = df[st.session_state.sample2].dropna()
    
    if len(data1) != len(data2):
        st.error("두 샘플의 데이터 수가 같아야 합니다.")
    else:
        results = paired_ttest_cal(data1, data2, 
                                st.session_state.confidence_level, 
                                st.session_state.alternative)
        
        c1, c2, c3 = st.columns(3)
        c1.write(f"샘플 크기: {results['n']}")
        c1.write(f"평균 차이: {results['mean_diff']:.4f}")
        c1.write(f"표준편차 차이: {results['std_diff']:.4f}")
        c2.write(f"표준오차 차이: {results['se_diff']:.4f}")
        c2.write(f"T 통계량: {results['t_stat']:.4f}")
        c2.write(f"P-value: {results['p_value']:.4f}")
        c3.write(f"{st.session_state.confidence_level}% 신뢰구간: ({results['ci'][0]:.4f}, {results['ci'][1]:.4f})")
        
        paired_ttest_plot(results, data1, data2, st.session_state.sample1, st.session_state.sample2)