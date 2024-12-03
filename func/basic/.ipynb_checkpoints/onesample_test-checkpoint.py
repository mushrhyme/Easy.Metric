# Description: 1샘플 T테스트를 위한 함수들을 정의한 파일입니다.

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


def onesample_test_set():
    col = st.session_state.df.columns.tolist()
    st.session_state.test_type = st.selectbox("방법", options=["t검정", "z검정"], index=0)
    if st.session_state.test_type =="z검정":
        st.session_state.pop_std = st.number_input("알려진 표준편차")
    st.session_state.sample = st.selectbox(f"표본1",
                                            options=col,
                                            index=0)

    st.session_state.avg = st.text_input(f"가설 평균",)

    st.session_state.confidence_level = st.number_input(f"신뢰수준",
                                                        value=95.0,
                                                        step=1.0)

    st.session_state.alternative = st.selectbox(f"대립 가설",
                                                ["평균 < 가설 평균",
                                                "평균 ≠ 가설 평균",
                                                "평균 > 가설 평균"])

def onesample_test_cal(df):
    sample_data = df[st.session_state.sample]
    sample_data = sample_data.dropna()

    hypothesis_mean = float(st.session_state.avg)
    confidence_level = st.session_state.confidence_level / 100

    # 기술통계량 계산
    sample_mean = np.mean(sample_data)
    sample_std = np.std(sample_data, ddof=1)
    sample_size = len(sample_data)
    standard_error = sample_std / np.sqrt(sample_size)
    mean_diff = sample_mean - hypothesis_mean
    if st.session_state.test_type == 't검정':
        # 1 샘플 t 검정 수행
        
        degree_of_freedom = sample_size - 1
        test_name = 'T'
        
        # 대립가설에 따른 신뢰구간 계산
        if st.session_state.alternative == "평균 < 가설 평균":
            stat, p_value = stats.ttest_1samp(sample_data, popmean=hypothesis_mean, alternative="less")
            ci_name = f"{confidence_level * 100}% 상한"
            ci = stats.t.interval(confidence_level, degree_of_freedom, loc=sample_mean, scale=standard_error)[1]
            ci_result = pd.DataFrame({
                '차이': [np.round(sample_mean - hypothesis_mean, 3)],
                ci_name: [np.round(ci, 3)]
            })
        elif st.session_state.alternative == "평균 > 가설 평균":
            stat, p_value = stats.ttest_1samp(sample_data, popmean=hypothesis_mean, alternative="greater")
            ci_name = f"{confidence_level * 100}% 하한"
            ci = stats.t.interval(confidence_level, degree_of_freedom, loc=sample_mean, scale=standard_error)[0]
            ci_result = pd.DataFrame({
                '차이': [np.round(sample_mean - hypothesis_mean, 3)],
                ci_name: [np.round(ci, 3)]
            })
        else:  # "차이 ≠ 귀무가설에서의 차이"
            stat, p_value = stats.ttest_1samp(sample_data, popmean=hypothesis_mean, alternative="two-sided")
            ci_name = f"{confidence_level * 100}% CI"
            ci = stats.t.interval(confidence_level, degree_of_freedom, loc=sample_mean, scale=standard_error)
            ci_result = pd.DataFrame({
                '차이': [np.round(sample_mean - hypothesis_mean, 3)],
                f"{confidence_level * 100}% CI 하한": [np.round(ci[0], 3)],
                f"{confidence_level * 100}% CI 상한": [np.round(ci[1], 3)]
            })
    else:  # 'z검정'
        # Z 통계량 계산
        population_std = float(st.session_state.pop_std)
        z_stat = (sample_mean - hypothesis_mean) / (population_std / np.sqrt(sample_size))

        # p-value 계산
        if st.session_state.alternative == "평균 < 가설 평균":
            p_value = stats.norm.cdf(z_stat)
        elif st.session_state.alternative == "평균 > 가설 평균":
            p_value = 1 - stats.norm.cdf(z_stat)
        else:  # "차이 ≠ 귀무가설에서의 차이"
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        stat = z_stat
        degree_of_freedom = "N/A"
        test_name = 'Z'
        
        # 대립가설에 따른 신뢰구간 계산
        if st.session_state.alternative == "평균 < 가설 평균":
            ci_name = f"{confidence_level * 100}% 상한"
            ci = stats.norm.interval(confidence_level, loc=sample_mean, scale=population_std / np.sqrt(sample_size))[1]
            ci_result = pd.DataFrame({
                '차이': [np.round(sample_mean - hypothesis_mean, 3)],
                ci_name: [np.round(ci, 3)]
            })
        elif st.session_state.alternative == "평균 > 가설 평균":
            ci_name = f"{confidence_level * 100}% 하한"
            ci = stats.norm.interval(confidence_level, loc=sample_mean, scale=population_std / np.sqrt(sample_size))[0]
            ci_result = pd.DataFrame({
                '차이': [np.round(sample_mean - hypothesis_mean, 3)],
                ci_name: [np.round(ci, 3)]
            })
        else:  # "차이 ≠ 귀무가설에서의 차이"
            ci_name = f"{confidence_level * 100}% CI"
            ci = stats.norm.interval(confidence_level, loc=sample_mean, scale=population_std / np.sqrt(sample_size))
            ci_result = pd.DataFrame({
                '차이': [np.round(sample_mean - hypothesis_mean, 3)],
                f"{confidence_level * 100}% CI 하한": [np.round(ci[0], 3)],
                f"{confidence_level * 100}% CI 상한": [np.round(ci[1], 3)]
            })
            
    alpha = 1 - confidence_level
    # st.markdown(f":green-background[{feature}]의 p-값 = :blue-background[{p_value:.4f}] < :red-background[{st.session_state.alpha_out}] = α")
    if st.session_state.alternative == "평균 < 가설 평균":
        st.markdown(f"#### 귀무가설: μ = {st.session_state.avg}")
        st.markdown(f"#### 대립가설: μ < {st.session_state.avg}")
        result = f"#### 결론: 유의수준 {alpha:.2f}에서 모집단1의 평균이 {st.session_state.avg}보다 작다고 판단할 만한 통계적 근거가"
    elif st.session_state.alternative == "평균 > 가설 평균":
        st.markdown(f"#### 귀무가설: μ = {st.session_state.avg}")
        st.markdown(f"#### 대립가설: μ > {st.session_state.avg}")
        result = f"#### 결론: 유의수준 {alpha:.2f}에서 모집단1의 평균이 {st.session_state.avg}보다 크다고 판단할 만한 통계적 근거가"
    else:
        st.markdown(f"#### 귀무가설: μ = {st.session_state.avg}")
        st.markdown(f"#### 대립가설: μ ≠ {st.session_state.avg}")
        result = f"#### 결론: 유의수준 {alpha:.2f}에서 모집단1의 평균이 {st.session_state.avg}과 다르다고 판단할 만한 통계적 근거가"


    if st.session_state.test_type == 'z검정':
        st.write(f"사용된 모집단 표준편차: {population_std:.3f}")

    # 검정 결과 해석
    
    if p_value < alpha:
        st.success(f" 유의수준 {alpha:.2f}에서 귀무가설을 기각합니다.")
        st.markdown(result+" 있습니다.")
    else:
        st.error(f" 유의수준 {alpha:.2f}에서 귀무가설을 기각할 수 없습니다.")
        st.markdown(result+" 없습니다.")

    # 결과를 데이터프레임에 저장
    st.session_state.test_df = pd.DataFrame({
        f"{test_name}-값": [np.round(stat, 3)],
        "DF": [degree_of_freedom],
        "P-값": [np.round(p_value, 3)]
    })

    st.session_state.stats_df = pd.DataFrame({
        '표본': [st.session_state.sample],
        '표본 크기': [sample_size],
        '표본 평균': [np.round(sample_mean, 3)],
        '표준 편차': [np.round(sample_std, 3)],
        '평균의 표준 오차': [np.round(standard_error, 3)]
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

    st.session_state.test_df = pd.DataFrame({
        f"{'t' if st.session_state.test_type == 't검정' else 'z'}-값": [np.round(stat, 3)],
        "DF": degree_of_freedom,
        "P-값": [np.round(p_value, 3)]
    })


def onesample_test_plot(df):
    colname = st.session_state.sample
    sample_data = df[colname].dropna()
    hypothesis_mean = float(st.session_state.avg)
    
    plot_df = pd.DataFrame({
        'Value': sample_data
    })
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f"{colname}의 개별 값 그림", f"{colname}의 상자 그림"),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    strip_fig = px.strip(plot_df, y='Value', stripmode='overlay')
    for trace in strip_fig.data:
        trace.showlegend = False
        fig.add_trace(trace, row=1, col=1)
    
    # Add sample mean line to strip plot
    sample_mean = sample_data.mean()
    fig.add_trace(
        go.Scatter(
            x=[None], 
            y=[sample_mean],
            mode='lines',
            line=dict(color='red', width=2),
            name='표본 평균',
            showlegend=True
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[hypothesis_mean],
            mode='lines',
            line=dict(color='blue', width=2),
            name='가설 평균',
            showlegend=True
        ),
        row=1, col=1
    )

    fig.add_hline(y=sample_mean, line_color="red", line_width=2, row=1, col=1)
    fig.add_hline(y=hypothesis_mean, line_color="blue", line_width=2, row=1, col=1)
    
    # Box plot
    box_fig = px.box(plot_df, y='Value', notched=False)
    for trace in box_fig.data:
        trace.showlegend = False
        fig.add_trace(trace, row=2, col=1)

    fig.add_hline(y=sample_mean, line_color="red", line_width=2, row=2, col=1)
    fig.add_hline(y=hypothesis_mean, line_color="blue", line_width=2, row=2, col=1)

    fig.update_layout(
        title_text="",
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            orientation="v"
        ),
        margin=dict(r=100)  
    )

    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)
    
    _, plot_col, _ = st.columns([1, 8, 1])
    with plot_col:
        st.plotly_chart(fig, use_container_width=True)

def onesample_test_run():  
    df = convert_to_calculatable_df()
    # --------------------------------------
    onesample_test_cal(df)
    st.divider()
    st.write("**기술 통계량**")
    st.data_editor(st.session_state.stats_df, hide_index=True, key="stats", use_container_width=True)

    c1, c2 = st.columns(2)
    c1.write("**신뢰구간**")
    c1.data_editor(st.session_state.diff_df, hide_index=True, key="diff", use_container_width=True)
    c2.write("**검정**")
    c2.data_editor(st.session_state.test_df, hide_index=True, key="test", use_container_width=True)

    with st.expander("결과 그래프", expanded=False):
        # with st.container(border=True):
        onesample_test_plot(df)