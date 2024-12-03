# Description: 단일비율 검증을 위한 함수들을 정의한 파일입니다.

from utils.tools.UTIL import *
from scipy import stats
from plotly.subplots import make_subplots

import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

def oneproportion_test_set():
    col = st.session_state.df.columns.tolist()
    st.session_state.sample = st.selectbox(f"표본",
                                         options=col,
                                         index=0)
    
    st.session_state.hyp_prop = st.number_input(f"가설 비율",
                                               min_value=0.0,
                                               max_value=1.0,
                                               value=0.5,
                                               step=0.1)
    
    st.session_state.confidence_level = st.number_input(f"신뢰수준",
                                                      value=95.0,
                                                      step=1.0)
    
    st.session_state.alternative = st.selectbox(f"대립 가설",
                                              ["비율 < 가설 비율",
                                               "비율 ≠ 가설 비율",
                                               "비율 > 가설 비율"])

def oneproportion_test_cal(df):
    sample_data = df[st.session_state.sample].dropna()
    
    # 기본 통계량 계산
    sample_size = len(sample_data)
    success_count = sum(sample_data)
    sample_prop = success_count / sample_size
    hypothesis_prop = float(st.session_state.hyp_prop)
    confidence_level = st.session_state.confidence_level / 100
    
    # 표준오차 계산
    standard_error = np.sqrt(hypothesis_prop * (1 - hypothesis_prop) / sample_size)
    
    # Z 통계량 계산
    z_stat = (sample_prop - hypothesis_prop) / standard_error
    
    # p-value 계산
    if st.session_state.alternative == "비율 < 가설 비율":
        p_value = stats.norm.cdf(z_stat)
    elif st.session_state.alternative == "비율 > 가설 비율":
        p_value = 1 - stats.norm.cdf(z_stat)
    else:  # "비율 ≠ 가설 비율"
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    # 신뢰구간 계산
    sample_se = np.sqrt(sample_prop * (1 - sample_prop) / sample_size)
    
    if st.session_state.alternative == "비율 < 가설 비율":
        ci_name = f"{confidence_level * 100}% 상한"
        ci = stats.norm.interval(confidence_level, loc=sample_prop, scale=sample_se)[1]
        ci_result = pd.DataFrame({
            '차이': [np.round(sample_prop - hypothesis_prop, 3)],
            ci_name: [np.round(ci, 3)]
        })
    elif st.session_state.alternative == "비율 > 가설 비율":
        ci_name = f"{confidence_level * 100}% 하한"
        ci = stats.norm.interval(confidence_level, loc=sample_prop, scale=sample_se)[0]
        ci_result = pd.DataFrame({
            '차이': [np.round(sample_prop - hypothesis_prop, 3)],
            ci_name: [np.round(ci, 3)]
        })
    else:  # "비율 ≠ 가설 비율"
        ci = stats.norm.interval(confidence_level, loc=sample_prop, scale=sample_se)
        ci_result = pd.DataFrame({
            '차이': [np.round(sample_prop - hypothesis_prop, 3)],
            f"{confidence_level * 100}% CI 하한": [np.round(ci[0], 3)],
            f"{confidence_level * 100}% CI 상한": [np.round(ci[1], 3)]
        })
    
    alpha = 1 - confidence_level
    
    # 가설 표시
    if st.session_state.alternative == "비율 < 가설 비율":
        st.markdown(f"#### 귀무가설: p = {hypothesis_prop}")
        st.markdown(f"#### 대립가설: p < {hypothesis_prop}")
        result = f"#### 결론: 유의수준 {alpha:.2f}에서 모비율이 {hypothesis_prop}보다 작다고 판단할 만한 통계적 근거가"
    elif st.session_state.alternative == "비율 > 가설 비율":
        st.markdown(f"#### 귀무가설: p = {hypothesis_prop}")
        st.markdown(f"#### 대립가설: p > {hypothesis_prop}")
        result = f"#### 결론: 유의수준 {alpha:.2f}에서 모비율이 {hypothesis_prop}보다 크다고 판단할 만한 통계적 근거가"
    else:
        st.markdown(f"#### 귀무가설: p = {hypothesis_prop}")
        st.markdown(f"#### 대립가설: p ≠ {hypothesis_prop}")
        result = f"#### 결론: 유의수준 {alpha:.2f}에서 모비율이 {hypothesis_prop}과 다르다고 판단할 만한 통계적 근거가"
    
    # 검정 결과 해석
    if p_value < alpha:
        st.success(f" 유의수준 {alpha:.2f}에서 귀무가설을 기각합니다.")
        st.markdown(result+" 있습니다.")
    else:
        st.error(f" 유의수준 {alpha:.2f}에서 귀무가설을 기각할 수 없습니다.")
        st.markdown(result+" 없습니다.")
    
    # 결과를 데이터프레임에 저장
    st.session_state.stats_df = pd.DataFrame({
        '표본': [st.session_state.sample],
        '표본 크기': [sample_size],
        '성공 횟수': [success_count],
        '표본 비율': [np.round(sample_prop, 3)],
        '표준 오차': [np.round(standard_error, 3)]
    })
    
    st.session_state.diff_df = ci_result
    
    st.session_state.test_df = pd.DataFrame({
        "Z-값": [np.round(z_stat, 3)],
        "P-값": [np.round(p_value, 3)]
    })


def oneproportion_test_run():
    df = convert_to_calculatable_df()
    # --------------------------------------
    oneproportion_test_cal(df)
    st.divider()
    st.write("**기술 통계량**")
    st.data_editor(st.session_state.stats_df, hide_index=True, key="stats", use_container_width=True)
    
    c1, c2 = st.columns(2)
    c1.write("**신뢰구간**")
    c1.data_editor(st.session_state.diff_df, hide_index=True, key="diff", use_container_width=True)
    c2.write("**검정**")
    c2.data_editor(st.session_state.test_df, hide_index=True, key="test", use_container_width=True)