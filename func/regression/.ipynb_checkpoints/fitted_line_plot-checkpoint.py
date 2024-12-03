# 적합선 그래프를 그리기 위한 기능 코드
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

def draw_fitted_line_plot(df):
    y = df[st.session_state.target]

    # 첫 번째 예측 변수 선택 (x축으로 사용)
    x_var = st.session_state.predictor[0]
    X_plot = st.session_state.X[x_var]
    np.random.seed(2024)
    jitter = np.random.normal(0, 0.015 * (X_plot.max() - X_plot.min()), size=len(X_plot))
    hover_text = [f"{st.session_state.target}: {y:.4f}<br>{x_var}: {x:.4f}" for x, y in zip(X_plot, y)]
    X_plot += jitter
    
    # 정렬된 X 값으로 새로운 데이터프레임 생성
    X_range = pd.DataFrame({x_var: np.linspace(st.session_state.X[x_var].min(), st.session_state.X[x_var].max(), 100)})

    # 다른 변수들의 평균값으로 설정
    for var in st.session_state.predictor:
        if var != x_var:
            X_range[var] = st.session_state.X[var].mean()

    # 다항식 항, 상호작용 항 처리
    for col in st.session_state.X.columns:
        if ":" in col or "^" in col:
            terms = col.split(":")
            X_range[col] = np.prod([X_range[term.split("^")[0]] ** (int(term.split("^")[1]) if "^" in term else 1) for term in terms], axis=0)

    # 예측
    X_range = sm.add_constant(X_range)
    y_pred_range = st.session_state.model.predict(X_range)
    
    # 신뢰구간과 예측구간 계산
    n, p = len(df), len(st.session_state.model.params)
    t_value = stats.t.ppf(0.975, n - p)
    std_error = np.sqrt(st.session_state.model.mse_resid)
    
    # 공분산 행렬 계산
    cov_matrix = st.session_state.model.cov_params()
    x_design = X_range.values
    var_pred = np.sum(x_design * np.dot(cov_matrix, x_design.T).T, axis=1)
    
    # 신뢰구간, 예측구간
    ci_err = t_value * np.sqrt(var_pred)
    pi_err = t_value * np.sqrt(std_error**2 + var_pred)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=X_plot, 
        y=y, 
        mode="markers", 
        name="Data Points", 
        marker=dict(color='rgba(0,0,0,0.6)', size=5),
        text=hover_text,
        showlegend=False,
        hoverinfo='text'
    ))
    fig.add_trace(go.Scatter(x=X_range[x_var], y=y_pred_range, mode="lines", name=f"{st.session_state.degree}차 적합선", line=dict(color="red")))
    
    
    fig.add_trace(go.Scatter(x=X_range[x_var], y=y_pred_range+ci_err, mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=X_range[x_var], y=y_pred_range-ci_err, mode="lines", line=dict(width=0), 
            fillcolor="rgba(255, 0, 0, 0.1)", 
            fill="tonexty", name="95% CI"
        )
    )
    fig.add_trace(go.Scatter(x=X_range[x_var], y=y_pred_range+ci_err, mode="lines", name="CI 경계", line=dict(color="red", width=1, dash="dash")))
    fig.add_trace(go.Scatter(x=X_range[x_var], y=y_pred_range-ci_err, mode="lines", showlegend=False, line=dict(color="red", width=1, dash="dash")))

    fig.add_trace(go.Scatter(x=X_range[x_var], y=y_pred_range+pi_err, mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=X_range[x_var], y=y_pred_range-pi_err, mode="lines", line=dict(width=0), 
            fillcolor="rgba(0, 0, 255, 0.1)", 
            fill="tonexty", name="95% PI"
        )
    )
    fig.add_trace(go.Scatter(x=X_range[x_var], y=y_pred_range+pi_err, mode="lines", name="PI 경계", line=dict(color="blue", width=1, dash="dash")))
    fig.add_trace(go.Scatter(x=X_range[x_var], y=y_pred_range-pi_err, mode="lines", showlegend=False, line=dict(color="blue", width=1, dash="dash")))
    
    fig.update_layout(
        title=f"{st.session_state.degree}차 회귀분석",
        xaxis_title=x_var, 
        yaxis_title=st.session_state.target,
        legend=dict(
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
        ),
        margin=dict(r=150) 
    )
    st.plotly_chart(fig)


###########################################################################################################################
def fitted_line_plot_set():
    df = convert_to_calculatable_df()
    st.session_state.target = st.selectbox("반응변수", df.columns.tolist())            
    predictor = df.columns.tolist()
    predictor.remove(st.session_state.target)
    st.session_state.predictor = st.selectbox("예측변수", predictor)

    # 반응: 계량형 변수 확인
    if df[st.session_state.target].dtype=="object":
        st.error(f"{st.session_state.target}는 계량형 변수가 아닙니다. 계량형 변수만 선택해주세요.")
        return
    else:
        # 요인: 계량형 변수 확인
        if df[st.session_state.predictor].dtype=="object":
            st.error(f"{st.session_state.predictor}는 계량형 변수가 아닙니다. 계량형 변수만 선택해주세요.")
            return
    
    st.session_state.degree = int(st.radio("회귀 모형 유형", ["1차", "2차", "3차"])[0])
    predictor = st.session_state.predictor
    st.session_state.interactions = [":".join([predictor] * i) for i in range(2, st.session_state.degree + 1)]
    st.session_state.predictor = [predictor]
    with st.popover("옵션"):
        st.session_state.residuals = st.selectbox("사용할 잔차", ["정규 잔차", "표준화 잔차", "외적 스튜던트화 잔차"])

def fitted_line_plot_cal(df):
    run_regression(df)
    run_anova(df)


def fitted_line_plot_plot(df):
    draw_fitted_line_plot(df)
    if st.session_state.residuals == "정규 잔차":
        draw_reg_plot(st.session_state.model.resid, st.session_state.model.fittedvalues)
    elif st.session_state.residuals == "표준화 잔차":
        draw_reg_plot(st.session_state.model.resid_pearson, st.session_state.model.fittedvalues)
    elif st.session_state.residuals == "외적 스튜던트화 잔차":
        draw_reg_plot(st.session_state.model.outlier_test()["student_resid"], st.session_state.model.fittedvalues)


def fitted_line_plot_run():  
    df = convert_to_calculatable_df()
    # --------------------------------------
    if len(df[st.session_state.target].dropna(axis=0, how="all"))==len(df[st.session_state.predictor].dropna(axis=0, how="all")):
        fitted_line_plot_cal(df)
        st.write("**회귀방정식**")
        st.markdown(st.session_state.equation)
        st.write("**계수**")
        st.data_editor(st.session_state.coeff_df, key="coeff", use_container_width=True)
        st.write("**모형 요약**")
        st.data_editor(st.session_state.reg_df, hide_index=True, key="reg", use_container_width=True)
        st.write("**분산 분석**")
        st.data_editor(st.session_state.anova_df, key="anova", use_container_width=True)
        # --------------------------------------
        with st.container(border=True):
            fitted_line_plot_plot(df)        
    else:
        st.error(f"{st.session_state.target}열과 {st.session_state.predictor}열의 길이는 동일해야 합니다.")