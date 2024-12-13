# Description: 두 비율 검증을 위한 함수들을 정의한 파일입니다.

from utils.tools.UTIL import *
from scipy import stats
from plotly.subplots import make_subplots

import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go


def twoproportion_test_set():
    df = convert_to_calculatable_df()
    col = df.columns.tolist()
    if len(col) < 2:
        st.error("2개 이상의 열이 존재해야 합니다. 데이터를 확인해주세요.")
    else:
        st.session_state.test_type = "비율 검정"
        st.session_state.sample1 = st.selectbox(f"표본1", options=col, index=0)
        st.session_state.sample2 = st.selectbox(f"표본2", options=col, index=1)
        st.session_state.significance_level = 1 - st.number_input(f"신뢰수준", value=95.0, step=1.0) / 100.0
        st.session_state.null_diff = st.number_input(f"귀무 가설에서의 차이", value=0.0, step=0.01)
        st.session_state.alternative = st.selectbox(f"대립 가설", ["차이 < 귀무가설에서의 차이", "차이 ≠ 귀무가설에서의 차이", "차이 > 귀무가설에서의 차이"])

def twoproportion_test_cal(df):
    group_a = df[st.session_state.sample1]
    group_b = df[st.session_state.sample2]
    group_a = group_a.dropna()
    group_b = group_b.dropna()

    # 기술통계량 계산
    n_a = len(group_a)
    n_b = len(group_b)
    success_a = group_a.sum()
    success_b = group_b.sum()
    p_a = success_a / n_a
    p_b = success_b / n_b
    
    # 표준오차 계산
    p_pooled = (success_a + success_b) / (n_a + n_b)
    se_diff = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b))
    
    # 기술통계량 DataFrame 생성
    st.session_state.stats_df = pd.DataFrame({
        '표본': [st.session_state.sample1, st.session_state.sample2],
        '표본 크기': [n_a, n_b],
        '성공 횟수': [success_a, success_b],
        '표본 비율': [p_a, p_b],
        '표준 오차': [np.sqrt(p_a * (1-p_a) / n_a), np.sqrt(p_b * (1-p_b) / n_b)]
    })

    # 통계량 계산
    prop_diff = p_a - p_b
    z_stat = (prop_diff - st.session_state.null_diff) / se_diff
    confidence_level = (1 - st.session_state.significance_level) * 100

    # p-value 계산
    if st.session_state.alternative == "차이 > 귀무가설에서의 차이":
        p_value = 1 - stats.norm.cdf(z_stat)
        ci_name = f"차이에 대한 {confidence_level}% 하한"
        ci = prop_diff + stats.norm.ppf(st.session_state.significance_level) * se_diff
    elif st.session_state.alternative == "차이 < 귀무가설에서의 차이":
        p_value = stats.norm.cdf(z_stat)
        ci_name = f"차이에 대한 {confidence_level}% 상한"
        ci = prop_diff + stats.norm.ppf(1 - st.session_state.significance_level) * se_diff
    else:  # "차이 ≠ 귀무가설에서의 차이"
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        ci_name = f"차이에 대한 {confidence_level}% CI"
        ci = stats.norm.interval(confidence_level/100, prop_diff, se_diff)

    # 검정 결과 해석
    alpha = 1 - confidence_level/100

    if st.session_state.alternative == "차이 < 귀무가설에서의 차이":
        st.markdown(f"#### 귀무가설: p₁ - p₂ = {st.session_state.null_diff}")
        st.markdown(f"#### 대립가설: p₁ - p₂ < {st.session_state.null_diff}")
        result = f"#### 결론: 유의수준 {alpha:.2f}에서 모비율1이 모비율2보다 작다고 판단할 만한 통계적 근거가"
    elif st.session_state.alternative == "차이 > 귀무가설에서의 차이":
        st.markdown(f"#### 귀무가설: p₁ - p₂ = {st.session_state.null_diff}")
        st.markdown(f"#### 대립가설: p₁ - p₂ > {st.session_state.null_diff}")
        result = f"#### 결론: 유의수준 {alpha:.2f}에서 모비율1이 모비율2보다 크다고 판단할 만한 통계적 근거가"
    else:
        st.markdown(f"#### 귀무가설: p₁ - p₂ = {st.session_state.null_diff}")
        st.markdown(f"#### 대립가설: p₁ - p₂ ≠ {st.session_state.null_diff}")
        result = f"#### 결론: 유의수준 {alpha:.2f}에서 실제 두 모비율이 다르다고 판단할 만한 통계적 근거가"

    if p_value < alpha:
        st.success(f" 유의수준 {alpha:.2f}에서 귀무가설을 기각합니다.")
        st.markdown(result+" 있습니다.")
    else:
        st.error(f" 유의수준 {alpha:.2f}에서 귀무가설을 기각할 수 없습니다.")
        st.markdown(result+" 없습니다.")

    st.session_state.test_df = pd.DataFrame({
        'z-값': [np.round(z_stat, 3)],
        'P-값': [np.round(p_value, 3)]
    })

    if isinstance(ci, np.float64):
        st.session_state.diff_df = pd.DataFrame({
            '차이': [np.round(prop_diff, 3)],
            ci_name: [np.round(ci, 3)]
        })
    else:
        st.session_state.diff_df = pd.DataFrame({
            '차이': [np.round(prop_diff, 3)],
            ci_name: [np.round(ci[0], 3)],
            '': [np.round(ci[1], 3)]
        })


def twoproportion_test_plot(df):
    pass
    

def twoproportion_test_run():  
    df = convert_to_calculatable_df()
    proportion_test_cal(df)
    st.divider()
    st.write("**기술 통계량**")
    st.data_editor(st.session_state.stats_df, hide_index=True, key="stats", use_container_width=True)

    c1, c2 = st.columns(2)
    c1.write("**신뢰구간**")
    c1.data_editor(st.session_state.diff_df, hide_index=True, key="diff", use_container_width=True)
    c2.write("**검정**")
    c2.data_editor(st.session_state.test_df, hide_index=True, key="test", use_container_width=True)
