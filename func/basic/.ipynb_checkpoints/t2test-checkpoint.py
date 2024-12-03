from utils.tools.UTIL import *
from scipy import stats
from plotly.subplots import make_subplots

import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go


def t2test_set():
    df = convert_to_calculatable_df()
    col = df.columns.tolist()
    st.session_state.sample1 = st.selectbox(f"표본1", options=col, index=0)
    st.session_state.sample2 = st.selectbox(f"표본2", options=col, index=1)
    st.session_state.significance_level = 1 - st.number_input(f"신뢰수준", value=95.0, step=1.0) / 100.0
    st.session_state.null_diff = st.number_input(f"귀무 가설에서의 차이", value=0.05, step=0.01)
    st.session_state.alternative = st.selectbox(f"대립 가설", ["차이 < 귀무가설에서의 차이", "차이 ≠ 귀무가설에서의 차이", "차이 > 귀무가설에서의 차이"])
    st.session_state.equal_var = st.checkbox(f"등분산 가정")

def t2test_cal(df):
    group_a = df[st.session_state.sample1]
    group_b = df[st.session_state.sample2]

    group_a = group_a.dropna()
    group_b = group_b.dropna()
    
    #### 기술통계량
    desc_a = group_a.describe()
    desc_b = group_b.describe()
    
    n_a, n_b = desc_a['count'], desc_b['count']
    # 평균, 표준편차
    mean_a, mean_b = desc_a['mean'], desc_b['mean']
    std_a, std_b = desc_a['std'], desc_b['std']
    # 표본오차
    sem_a = std_a / np.sqrt(n_a)
    sem_b = std_b / np.sqrt(n_b)
    st.session_state.stats_df = pd.DataFrame({
        '표본': [st.session_state.sample1, st.session_state.sample2],
        'N': [desc_a['count'], desc_b['count']],
        '평균': [desc_a['mean'], desc_b['mean']],
        '표준 편차': [desc_a['std'], desc_b['std']],
        '평균의 표준 오차': [sem_a, sem_b]
    })

    #### 2-표본 t-검정
    # 등분산 가정 시
    if st.session_state.equal_var:
        comp = sm.stats.CompareMeans(sm.stats.DescrStatsW(group_a), sm.stats.DescrStatsW(group_b))
        t_stat, _, degree_of_freedom = comp.ttest_ind(usevar ="pooled")
    # 이분산 가정 시
    else:
        comp = sm.stats.CompareMeans(sm.stats.DescrStatsW(group_a), sm.stats.DescrStatsW(group_b))
        t_stat, _, degree_of_freedom = comp.ttest_ind(usevar="unequal")

    # 자유도 정수형 처리 (Minitab 특징)
    degree_of_freedom = int(degree_of_freedom)

    #### P값, 신뢰구간
    mean_diff = desc_a['mean'] - desc_b['mean']
    se_diff = np.sqrt(sem_a**2 + sem_b**2)
    confidence_level = (1 - st.session_state.significance_level)*100
    if st.session_state.alternative == "차이 > 귀무가설에서의 차이":
        p_value = 1 - stats.t.cdf(t_stat, df=degree_of_freedom)
        ci_name = f"차이에 대한 {confidence_level}% 하한"
        ci = mean_diff + stats.t.ppf(st.session_state.significance_level, degree_of_freedom) * se_diff
    elif st.session_state.alternative == "차이 < 귀무가설에서의 차이":
        p_value = stats.t.cdf(t_stat, df=degree_of_freedom)
        ci_name = f"차이에 대한 {confidence_level}% 상한"
        ci = mean_diff - stats.t.ppf(st.session_state.significance_level, degree_of_freedom) * se_diff
    elif st.session_state.alternative == "차이 ≠ 귀무가설에서의 차이":
        p_value = 2 * stats.t.cdf(-abs(t_stat), df=degree_of_freedom)
        ci_name = f"차이에 대한 {confidence_level}% CI"
        ci = stats.t.interval(1-st.session_state.significance_level, degree_of_freedom, loc=mean_diff, scale=se_diff)

    st.session_state.ttest_df = pd.DataFrame({
        "T-값": [np.round(t_stat, 3)],
        "DF": degree_of_freedom,
        "P-값": [np.round(p_value, 3)]
    })

    if isinstance(ci, np.float64):
        st.session_state.diff_df = pd.DataFrame({
                '차이': [np.round(mean_diff, 3)],
                ci_name: [np.round(ci, 3)],
            })
    else:
        st.session_state.diff_df = pd.DataFrame({
                '차이': [np.round(mean_diff, 3)],
                ci_name: [np.round(ci[0], 3)],
                '': [np.round(ci[1], 3)]
        }) 
        
def t2test_plot(df):
    colname1, colname2 = st.session_state.sample1, st.session_state.sample2
    melted_df = df[[colname1, colname2]].melt(var_name='Group', value_name='Value')

    fig = make_subplots(
        rows=2, cols=1,  # 1열 2행으로 변경
        subplot_titles=(f"{colname1}, {colname2}의 개별 값 그림", f"{colname1}, {colname2} 상자 그림"),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )

    strip_fig = px.strip(melted_df, x='Group', y='Value', stripmode='overlay', color='Group')
    for trace in strip_fig.data:
        fig.add_trace(trace, row=1, col=1)  # 첫 번째 행에 추가

    group_means = melted_df.groupby('Group')['Value'].mean().reset_index()
    fig.add_trace(
        go.Scatter(
            x=group_means['Group'], y=group_means['Value'], 
            mode='lines+markers', name='Mean', line=dict(color='red'), showlegend=True
        ), 
        row=1, col=1  # 첫 번째 행에 추가
    )

    box_fig = px.box(melted_df, x='Group', y='Value', notched=False, color='Group', labels={'Group': 'Sample'})
    for trace in box_fig.data:
        fig.add_trace(trace, row=2, col=1)  # 두 번째 행에 추가

    fig.add_trace(
        go.Scatter(
            x=group_means['Group'], y=group_means['Value'], 
            mode='lines+markers', name='Mean', 
            line=dict(color='black'), showlegend=True, 
        ), 
        row=2, col=1  # 두 번째 행에 추가
    )

    fig.update_layout(title_text="", height=800)  # 높이를 조정하여 두 그래프를 잘 표시
    
    _, plot2, _ = st.columns([1, 8, 1])
    plot2.plotly_chart(fig)
        
def t2test_run():
    df = convert_to_calculatable_df()
    # --------------------------------------
    t2test_cal(df)
    st.write("**기술 통계량**")
    st.data_editor(st.session_state.stats_df, hide_index=True, key="stats", use_container_width=True)

    c1, c2 = st.columns(2)
    c1.write("**차이 추정치**")
    c1.data_editor(st.session_state.diff_df, hide_index=True, key="diff", use_container_width=True)
    c2.write("**두 표본 T-검정**")
    c2.data_editor(st.session_state.ttest_df, hide_index=True, key="ttest", use_container_width=True)
    
    with st.container(border=True):
        t2test_plot(df)