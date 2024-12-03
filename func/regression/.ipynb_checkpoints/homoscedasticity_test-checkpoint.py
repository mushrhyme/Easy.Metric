from utils.tools.UTIL import *
import streamlit as st
import numpy as np
import pandas as pd

from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def ci_sd(data, confidence=0.95):
    n = len(data)
    df = n - 1
    sd = np.std(data, ddof=1)
    chi_lower = stats.chi2.ppf((1-confidence)/2, df)
    chi_upper = stats.chi2.ppf(1-(1-confidence)/2, df)
    ci_lower = np.sqrt(df * sd**2 / chi_upper)
    ci_upper = np.sqrt(df * sd**2 / chi_lower)
    return sd, (ci_lower, ci_upper)


def homoscedasticity_test_set():
    df = convert_to_calculatable_df()
    st.session_state.target = st.selectbox("반응변수", df.columns.tolist())
    predictor = df.columns.tolist()
    predictor.remove(st.session_state.target)
    st.session_state.predictor = st.selectbox("예측변수", predictor)
    st.session_state.significance_level = st.selectbox("유의수준", [0.05, 0.01, 0.1])

def homoscedasticity_test_cal(df):
    confidence_level = (1 - st.session_state.significance_level) * 100

    def calc_ci(group):
        n = len(group)
        std = group.std()
        se = std / np.sqrt(2 * (n - 1))
        df = n - 1
        ci = stats.t.interval(0.95, df, loc=std, scale=se)
        return pd.Series(
            {'N': n, '표준편차': std, f'{confidence_level}% CI 하한': ci[0], f'{confidence_level}% CI 상한': ci[1]})

    # 그룹별 통계량 계산
    grouped = df.groupby(st.session_state.predictor)[st.session_state.target]
    pivot = grouped.apply(calc_ci).reset_index()
    st.session_state.base_df = pd.pivot_table(
        pivot,
        index=st.session_state.predictor,
        columns=list(set(pivot) - set(df.columns)),
        values=[st.session_state.target], aggfunc="first")

    groups = [group[st.session_state.target].values for name, group in df.groupby(st.session_state.predictor)]
    levene_statistic, levene_pvalue = stats.levene(*groups)
    st.session_state.stats_df = pd.DataFrame({
        "방법": ["Levene 검정"],
        "검정통계량":levene_statistic ,
        "P-값": levene_pvalue
    })


def homoscedasticity_test_plot(df):
    pass

def homoscedasticity_test_run():  
    # 매우 중요 : 데이터프레임 가져오기 ------------
    df = convert_to_calculatable_df()
    print(st.session_state.predictor)
    df = df[[st.session_state.target, st.session_state.predictor]]
    # --------------------------------------
    if len(df[st.session_state.target].dropna(axis=0, how="all")) == len(
            df[st.session_state.predictor].dropna(axis=0, how="all")):
        homoscedasticity_test_cal(df)
        homoscedasticity_test_plot(df)
        st.write("**방법**")
        st.write("귀무가설: 모든 분산이 동일합니다.")
        st.write("대립가설: 하나 이상의 분산이 다릅니다.")
        st.write(f"유의수준: α={st.session_state.significance_level}")
        st.write(f"**표준 편차의 {int((1 - st.session_state.significance_level) * 100)}% Bonferroni 신뢰 구간**")
        st.data_editor(st.session_state.base_df, key="base", use_container_width=True)

        st.write("**검정**")
        st.data_editor(st.session_state.stats_df, key="stats", hide_index=True, use_container_width=True)
        # --------------------------------------
    else:
        st.error(f"{st.session_state.target}열과 {st.session_state.predictor}열의 길이는 동일해야 합니다.")