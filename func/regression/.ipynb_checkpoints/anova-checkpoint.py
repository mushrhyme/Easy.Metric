# 분산분석을 위한 기능 코드
from utils.tools.UTIL import *
from func.regression.regression_analysis import *
import streamlit as st
import numpy as np
import pandas as pd

from scipy import stats
import statsmodels.api as sm


import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def run_one_way_anova(df):
    n, p = df.shape
    confidence_level = (1 - st.session_state.significance_level)*100
    groups = [group for _, group in df.groupby(st.session_state.predictor)[st.session_state.target]]

    # 일원분산분석 수행
    try:
        f_value, p_value = stats.f_oneway(*groups)
    except ValueError:
        f_value, p_value = np.nan, np.nan

    # 총 제곱합(SST) 계산
    grand_mean = df[st.session_state.target].mean()
    SST = np.sum((df[st.session_state.target] - grand_mean)**2)

    # 그룹간 제곱합(SSB) 계산
    SSB = np.sum([len(group) * (group.mean() - grand_mean)**2 for group in groups])

    # 그룹내 제곱합(SSW) 계산
    SSW = np.nan if np.round(SST - SSB, 10) == 0 else SST - SSB

    # 자유도 계산
    df_between = len(groups) - 1
    df_within = n - len(groups)
    df_total = n - 1

    # 평균제곱(MS) 계산
    MSB = np.round(SSB / df_between, 3)
    MSW = np.round(SSW / df_within, 3) if not np.isnan(SSW) else np.nan

    # ANOVA 표 생성
    st.session_state.anova_df = pd.DataFrame({
        "DF": [df_between, df_within, df_total],
        "Adj SS": [SSB, SSW, SST],
        "Adj MS": [MSB, MSW, ""],
        "F-값": [np.round(f_value, 3), "", ""],
        "P-값": [np.round(p_value, 3), "", ""]
    }
    , index=[st.session_state.predictor, "오차", "총계"]
    ).round(3).fillna("*")

    # 4. 모형 요약
    s = np.sqrt(MSW)
    r_squared = SSB / SST
    adj_r_squared = 1 - (1 - r_squared) * (df_total / df_within) if df_within != 0 else np.nan

    # 예측 R-제곱 계산 (단순화된 방법, 실제로는 교차 검증을 사용해야 함)
    pred_r_squared = 1 - ((1 - r_squared) * ((df_total + 1) / df_within)) if df_within != 0 else np.nan

    st.session_state.reg_df = pd.DataFrame({
        "S": [s],
        "R-제곱": f"{np.round(r_squared*100, 2)}%",
        "R-제곱(수정)": f"{np.round(max(0,adj_r_squared)*100, 2)}%",
        "R-제곱(예측)": f"{np.round(max(0, pred_r_squared)*100, 2)}%"
    }).fillna("*")


    # 평균 및 신뢰구간 계산
    level_stats = df.groupby(st.session_state.predictor)[st.session_state.target].agg(["count", "mean", "std"]).round(3).reset_index()
    level_stats.columns = [st.session_state.predictor, "N", "평균", "표준편차"]
    pooled_std = np.sqrt(MSW) # 합동 표준편차 계산

    # 신뢰 수준에 따른 t-값 계산
    t_value = stats.t.ppf(1-(st.session_state.significance_level/2), df_within)
    margin_of_error  = t_value * pooled_std / np.sqrt(level_stats["N"])
    level_stats[f"{confidence_level}% CI 하한"] = level_stats["평균"] - margin_of_error
    level_stats[f"{confidence_level}% CI 상한"] = level_stats["평균"] + margin_of_error
    st.session_state.factor_df  = pd.DataFrame(level_stats).fillna("*")

def draw_interval_plot(df):
    confidence_level = (1 - st.session_state.significance_level)*100

    fig = go.Figure()

    # 요인 값을 추출하고 정렬
    factors = sorted(df[st.session_state.predictor].unique())

    # 요인 값을 0부터 시작하는 정수 인덱스에 매핑
    factor_to_index = {factor: index for index, factor in enumerate(factors)}

    for i, row in st.session_state.factor_df.iterrows():
        factor = row[st.session_state.predictor]
        mean = row["평균"]
        ci_lower = row[f"{confidence_level}% CI 하한"]
        ci_upper = row[f"{confidence_level}% CI 상한"]

        # x 좌표를 요인의 인덱스로 변경
        x_coord = factor_to_index[factor]

        hover_text = f"{confidence_level}% 중위수에 대한 구간 그림<br>" \
                     f"추정치 = {mean}<br>" \
                     f"구간 = ({ci_lower}, {ci_upper})<br>" \
                     f"N = {row['N']}"

        # 평균값 점 추가
        fig.add_trace(go.Scatter(
            x=[x_coord],
            y=[mean],
            mode="markers",
            name=f"{factor} 평균",
            marker=dict(size=10, color="blue"),
            showlegend=False,
            hoverinfo="text",
            hovertext=f"{mean:.3f}"
        ))

        # 신뢰구간 선 추가 (수직선)
        fig.add_trace(go.Scatter(
            x=[x_coord, x_coord, x_coord],
            y=[ci_lower, ci_upper, None],
            mode="lines",
            name=f"{factor} {confidence_level}% CI",
            line=dict(color="blue", width=2),
            showlegend=False,
            hoverinfo="text",
            hovertext=hover_text
        ))

        # 신뢰구간 끝에 수평선 추가
        whisker_width = 0.2  # 수평선의 너비 조절
        fig.add_trace(go.Scatter(
            x=[x_coord-whisker_width/2, x_coord+whisker_width/2, None, x_coord-whisker_width/2, x_coord+whisker_width/2],
            y=[ci_lower, ci_lower, None, ci_upper, ci_upper],
            mode="lines",
            line=dict(color="blue", width=2),
            showlegend=False,
            hoverinfo="text",
            hovertext=hover_text
        ))

    fig.update_layout(
        title=f"{st.session_state.target} vs {st.session_state.predictor}의 구간 그림",
        xaxis=dict(
            title=st.session_state.predictor,
            tickmode="array",
            tickvals=list(factor_to_index.values()),
            ticktext=list(factor_to_index.keys()),
            tickangle=-45,
        ),
        yaxis_title=st.session_state.target,
        showlegend=False,
        hovermode="closest"
    )
    st.plotly_chart(fig)


def draw_individual_value_plot(df):
    fig = go.Figure()

    # 요인 값을 추출하고 정렬
    factors = sorted(df[st.session_state.predictor].unique())

    # 요인 값을 0부터 시작하는 정수 인덱스에 매핑
    factor_to_index = {factor: index for index, factor in enumerate(factors)}

    # 평균값을 저장할 리스트
    mean_values = []

    for factor in factors:
        factor_data = df[df[st.session_state.predictor] == factor]
        x_coord = factor_to_index[factor]

        # 해당 수준의 값이 2개 이상인 경우에만 개별 값 표시
        if len(factor_data) > 1:
            # 개별 값 추가
            jittered_x = [x_coord] * len(factor_data) + np.random.normal(0, 0.1, len(factor_data))
            fig.add_trace(go.Scatter(
                x=jittered_x,
                y=factor_data[st.session_state.target],
                mode="markers",
                name=f"{factor} 개별값",
                marker=dict(size=8, opacity=0.5),
                showlegend=False,
                hoverinfo="y",
                hovertemplate="%{y:.3f}<extra></extra>"
            ))

        # 평균값 계산 및 저장
        mean_value = factor_data[st.session_state.target].mean()
        mean_values.append((x_coord, mean_value, len(factor_data)))

    # 평균값 점과 꺾은선 추가
    mean_x, mean_y, counts = zip(*mean_values)
    hover_text = [f"{y:.3f}" if count == 1 else f"평균: {y:.3f}" for y, count in zip(mean_y, counts)]

    fig.add_trace(go.Scatter(
        x=mean_x,
        y=mean_y,
        mode="lines+markers",
        name="평균값",
        line=dict(color="blue", width=2),
        marker=dict(size=8, color="blue"),
        showlegend=False,
        hoverinfo="text",
        hovertext=hover_text,
        hoverlabel=dict(namelength=0)
    ))

    fig.update_layout(
        title=f"{st.session_state.target} vs {st.session_state.predictor}의 개별 값 그림",
        xaxis=dict(
            title=st.session_state.predictor,
            tickmode="array",
            tickvals=list(factor_to_index.values()),
            ticktext=list(factor_to_index.keys()),
            tickangle=-45,
        ),
        yaxis_title=st.session_state.target,
        showlegend=True,
        margin=dict(b=100)
    )
    st.plotly_chart(fig)

def draw_box_plot(df):
    fig = go.Figure()
    means = []

    # 요인을 정렬합니다.
    factors = sorted(df[st.session_state.predictor].unique())

    for factor in factors:
        factor_data = df[df[st.session_state.predictor] == factor]
        fig.add_trace(go.Box(
            y=factor_data[st.session_state.target],
            name=str(factor),  # factor를 문자열로 변환
            boxpoints=False,
            fillcolor="lightblue",
            line_color="lightblue",
            showlegend=False,
        ))
        # 평균값 계산 및 저장
        mean = factor_data[st.session_state.target].mean()
        means.append(mean)

    # 평균값을 점으로 표시하고 선으로 연결
    fig.add_trace(go.Scatter(
        x=factors,
        y=means,
        mode="lines+markers",
        name="평균",
        line=dict(color="blue", width=1),
        marker=dict(color="blue", size=7),
        hoverinfo="text",
        hovertext=[f"{mean:.2f}" for mean in means],
        showlegend=False,
    ))

    fig.update_layout(
        title=f"{st.session_state.target} 상자 그림",
        xaxis_title=st.session_state.predictor,
        yaxis_title=st.session_state.target,
        xaxis=dict(type='category', categoryorder='array', categoryarray=factors, tickangle=-45,),
    )

    st.plotly_chart(fig)

###########################################################################################################################
def anova_set():
    df = convert_to_calculatable_df()
    st.session_state.target = st.selectbox("반응변수", df.columns.tolist())
    predictor = df.columns.tolist()
    predictor.remove(st.session_state.target)
    st.session_state.predictor = st.selectbox("예측변수", predictor)
    st.session_state.significance_level = st.selectbox("유의수준", [0.05, 0.01, 0.1])

def anova_cal(df):
    run_one_way_anova(df)

def anova_plot(df):
    # 구간 그림
    if len(list(set(df[st.session_state.predictor]))) > 45:
        st.error("구간 그림을 그릴 수 없습니다. 구간이 45개보다 많아서 구간 그림을 읽을 수 없습니다.")
    else:
        draw_interval_plot(df)
    # 개별 값 그림
    draw_individual_value_plot(df)

    # 상자 그림
    draw_box_plot(df)

    # 잔차그림
    group_means = df.groupby(st.session_state.predictor)[st.session_state.target].mean()
    residuals = df.apply(lambda row: row[st.session_state.target] - group_means[row[st.session_state.predictor]], axis=1)
    fitted_values = df[st.session_state.predictor].map(group_means)
    draw_reg_plot(residuals, fitted_values)


def anova_run():
    df = convert_to_calculatable_df()
    # --------------------------------------
    if len(df[st.session_state.target].dropna(axis=0, how="all"))==len(df[st.session_state.predictor].dropna(axis=0, how="all")):
        anova_cal(df)
        st.write("**방법**")
        st.write("귀무가설: 모든 평균이 동일합니다.")
        st.write("대립가설: 평균이 모두 같지 않습니다.")
        st.write(f"유의수준: α={st.session_state.significance_level}")
        st.info("분석을 위해 분산이 같다고 가정되었습니다.")

        st.write("**요인 정보**")
        factor_level = list(set(df[st.session_state.predictor]))
        st.data_editor(pd.DataFrame({
            "요인": [st.session_state.predictor],
            "수준": len(factor_level),
            "값": ", ".join(map(str, factor_level))
        }), hide_index=True, use_container_width=True)
        st.write("**분산 분석**")
        st.data_editor(st.session_state.anova_df, key="anova", use_container_width=True)
        st.write("**모형 요약**")
        st.data_editor(st.session_state.reg_df, key="reg", hide_index=True, use_container_width=True)
        st.write("**평균**")
        st.data_editor(st.session_state.factor_df, key="factor", hide_index=True, use_container_width=True)
        # --------------------------------------
        # with st.container(border=True):
        with st.expander("결과 그래프", expanded=False):
            anova_plot(df)
    else:
        st.error(f"{st.session_state.target}열과 {st.session_state.predictor}열의 길이는 동일해야 합니다.")