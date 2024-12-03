# 게이지R&R을 위한 기능 코드

from utils.tools.UTIL import *
import scipy as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from pypetb import RnR
from plotly.subplots import make_subplots

import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go


def draw_var_comp():
    fig = go.Figure()
    comp = ['총 Gage R&R', '반복성', '재현성', '부품-대-부품']
    fig.add_trace(
        go.Bar(
            x=comp,
            y=st.session_state.var_comp_df['%기여(분산 성분)'], 
            name='% 기여',
            # marker_color=st.session_state.bar_color1,
            legendgroup="group1",
            showlegend=True
        )
    )
    fig.add_trace(
        go.Bar(
            x=comp,
            y=st.session_state.std_comp_df['%연구 변동(%SV)'], 
            name='% 연구 변동',
            # marker_color=st.session_state.bar_color2,
            legendgroup="group1",
            showlegend=True
        )
    )    
    fig.update_layout(
        title='변동 성분',
        barmode='group',
    )
    fig.update_yaxes(title="백분율", showgrid=True)
    st.plotly_chart(fig, use_container_width=True)
  
def draw_meas_by_part(df):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[st.session_state.part], 
            y=df[st.session_state.measurement], 
            mode='markers',
            name='반응',
            marker=dict(color='blue', size=8, opacity=0.5),
            # legendgroup="group2",
            showlegend=False
        )
    )
    
    # 평균을 이은 선 그리기
    part_means = df.groupby(st.session_state.part)[st.session_state.measurement].mean()
    fig.add_trace(
        go.Scatter(
            x=part_means.index, 
            y=part_means.values, 
            mode='lines+markers',
            name='평균',
            line=dict(color='red', width=2),
            # legendgroup="group2",
            showlegend=False
        )
    )
    fig.update_layout(
        title="부품에 의한 반응",
    )
    fig.update_xaxes(
        title="부품",
        tickmode='linear', 
        tick0=1, dtick=1, 
        showgrid=True, 
    )
    fig.update_yaxes(showgrid=True)
    st.plotly_chart(fig, use_container_width=True)

def draw_R_ctrl_chart(df):
    fig = go.Figure()
    r_values = df.groupby([st.session_state.operator, st.session_state.part])[st.session_state.measurement].agg(lambda x: x.max() - x.min())
    r_mean = r_values.mean()
    
    ucl = r_mean * 2
    lcl = 0
    
    fig.add_trace(
        go.Scatter(
            y=r_values.values, 
            mode='lines+markers',
            name='R Values',
            # legendgroup="group3",
            # showlegend=True
        )
    )
    
    fig.add_hline(y=r_mean, line_dash="dash", line_color="green", annotation_text="Mean")
    fig.add_hline(y=ucl, line_dash="dash", line_color="red", annotation_text="UCL")
    fig.add_hline(y=lcl, line_dash="dash", line_color="red", annotation_text="LCL")
    
    # y축 범위 조정
    y_min = max(0, lcl * 0.9) 
    y_max = ucl * 1.1
    
    # y축 눈금값 설정
    tick_step = (y_max - y_min) / 5  # 6게 눈금 생성
    tick_step = np.ceil(tick_step * 10) / 10  # 눈금 간격 소수점 첫째 자리 올림
    y_ticks = np.arange(np.floor(y_min * 10) / 10, y_max, tick_step)
    fig.update_layout(
        title="R 관리도",
        annotations=[
            dict(x=1.02, y=ucl/y_max, xref="paper", yref="paper", 
                 text=f"UCL: {ucl:.2f}", showarrow=False, 
                 xanchor="left", yanchor="middle", font=dict(color="red")),
            dict(x=1.02, y=r_mean/y_max, xref="paper", yref="paper", 
                 text=f"Mean: {r_mean:.2f}", showarrow=False, 
                 xanchor="left", yanchor="middle", font=dict(color="green")),
            dict(x=1.02, y=lcl/y_max, xref="paper", yref="paper", 
                 text=f"LCL: {lcl:.2f}", showarrow=False, 
                 xanchor="left", yanchor="middle", font=dict(color="red")),
        ],
        margin=dict(r=150)  # Increase right margin to accommodate labels
    )
    fig.update_xaxes(title="부품", showgrid=True)
    fig.update_yaxes(
        showgrid=True, 
        range=[y_min, y_max],
        tickmode='array',
        tickvals=y_ticks,
        tickformat='.2f'
    )
    st.plotly_chart(fig, use_container_width=True)
    
def draw_meas_by_operator(df):
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            x=df[st.session_state.operator],
            y=df[st.session_state.measurement],
            showlegend=False
        )
    )
    
    operator_means = df.groupby(st.session_state.operator)[st.session_state.measurement].mean()
    fig.add_trace(
        go.Scatter(
            x=operator_means.index,
            y=operator_means.values,
            mode='markers',
            name='Mean',
            marker=dict(color='red', size=10, symbol='diamond'),
            showlegend=False
        )
    )
    fig.update_layout(
        title="측정자에 의한 반응",
    )
    fig.update_xaxes(title="측정자")
    st.plotly_chart(fig, use_container_width=True)


def draw_xbar_ctrl_chart(df):
    fig = go.Figure()
    xbar_values = df.groupby([st.session_state.operator, st.session_state.part])[st.session_state.measurement].mean()
    grand_mean = xbar_values.mean()
    
    std_dev = xbar_values.std()
    ucl = grand_mean + 3 * std_dev
    lcl = grand_mean - 3 * std_dev
    
    fig.add_trace(
        go.Scatter(
            y=xbar_values.values, 
            mode='lines+markers',
            name='Xbar Values'
        )
    )
    
    fig.add_hline(y=grand_mean, line_dash="dash", line_color="green", annotation_text="Mean")
    fig.add_hline(y=ucl, line_dash="dash", line_color="red", annotation_text="UCL")
    fig.add_hline(y=lcl, line_dash="dash", line_color="red", annotation_text="LCL")
    
    # y축 범위 조정
    y_min = max(0, lcl * 0.9) 
    y_max = ucl * 1.1
    
    # y축 눈금값 설정
    tick_step = (y_max - y_min) / 5  # 6게 눈금 생성
    tick_step = np.ceil(tick_step * 10) / 10  # 눈금 간격 소수점 첫째 자리 올림
    y_ticks = np.arange(np.floor(y_min * 10) / 10, y_max, tick_step)
    fig.update_layout(
        title="X bar 관리도",
        annotations=[
            dict(x=1.02, y=ucl/y_max, xref="paper", yref="paper", 
                 text=f"UCL: {ucl:.2f}", showarrow=False, 
                 xanchor="left", yanchor="middle", font=dict(color="red")),
            dict(x=1.02, y=grand_mean/y_max, xref="paper", yref="paper", 
                 text=f"Mean: {grand_mean:.2f}", showarrow=False, 
                 xanchor="left", yanchor="middle", font=dict(color="green")),
            dict(x=1.02, y=lcl/y_max, xref="paper", yref="paper", 
                 text=f"LCL: {lcl:.2f}", showarrow=False, 
                 xanchor="left", yanchor="middle", font=dict(color="red")),
        ],
        margin=dict(r=150)  # Increase right margin to accommodate labels
    )
    fig.update_xaxes(title="부품", showgrid=True)
    fig.update_yaxes(
        showgrid=True, 
        range=[y_min, y_max],
        tickmode='array',
        tickvals=y_ticks,
        tickformat='.2f'
    )
    st.plotly_chart(fig, use_container_width=True)

def draw_interaction(df):
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    interaction_means = df.groupby([st.session_state.part, st.session_state.operator])[st.session_state.measurement].mean().unstack()

    for i, operator in enumerate(interaction_means.columns):
        fig.add_trace(
            go.Scatter(
                x=interaction_means.index,
                y=interaction_means[operator],
                mode='lines+markers',
                name=f'측정자 {str(int(operator))}',
                line=dict(color=colors[i % len(colors)], width=2),
            )
        )
    fig.update_layout(
        title="부품-대-측정자 교호작용",
        hovermode='x unified'
    )
    fig.update_xaxes(
        title="부품",
        tickmode='linear', 
        tick0=1, dtick=1, 
        showgrid=True, 
        # gridwidth=1, gridcolor='lightgray'
    )
    fig.update_yaxes(showgrid=True)
    st.plotly_chart(fig, use_container_width=True)


###########################################################################################################################
def gage_RnR_set():
    df = convert_to_calculatable_df()
    st.session_state.part = st.selectbox("시료 번호", df.columns.tolist())
    st.session_state.operator = st.selectbox("측정 시스템", df.columns.tolist())
    st.session_state.measurement = st.selectbox("측정 데이터", df.columns.tolist())
    st.session_state.studyvar = st.number_input("연구 변동(표준 편차 수)", value=6)
    if st.toggle("공정 공차"):
        option1 = st.radio("", ["최소한 하나의 규격 한계 입력", "규격 상한-규격 하한"], label_visibility="collapsed")
        if option1 == "최소한 하나의 규격 한계 입력":
            st.session_state.lsl = st.number_input("규격 하한")
            st.session_state.usl = st.number_input("규격 상한")
            st.session_state.tolerance = st.session_state.usl-st.session_state.lsl
        elif option1 == "규격 상한-규격 하한":
            st.session_state.tolerance = st.number_input("", value=st.session_state.usl-st.session_state.lsl, label_visibility="collapsed")


def gage_RnR_cal(df):
    df = pd.DataFrame({
        "Part": df[st.session_state.part],
        "Operator": df[st.session_state.operator],
        "Measurement": df[st.session_state.measurement]
    })

    # 이원 분산 분석 수행
    model = ols("Measurement ~ C(Part) + C(Operator) + C(Part):C(Operator)", data=df).fit()
    anova_results = sm.stats.anova_lm(model)
    
    # 반복성 분산 분석표에 추가
    SS_Total = np.sum((df["Measurement"] - np.mean(df["Measurement"])) ** 2)
    SS_Repeatability = anova_results.loc["Residual", "sum_sq"]
    df_Repeatability = anova_results.loc["Residual", "df"]
    
    # 반복성에 대한 분산 계산
    MS_Repeatability = SS_Repeatability / df_Repeatability
    
    # F값 수정
    MS_Interaction = anova_results.loc["C(Part):C(Operator)", "sum_sq"] / anova_results.loc["C(Part):C(Operator)", "df"]
    
    # Part의 F값 수정
    anova_results.loc["C(Part)", "F"] = anova_results.loc["C(Part)", "mean_sq"] / MS_Interaction
    
    # Operator의 F값 수정
    anova_results.loc["C(Operator)", "F"] = anova_results.loc["C(Operator)", "mean_sq"] / MS_Interaction
    
    # p-value 재계산
    anova_results.loc["C(Part)", "PR(>F)"] = 1 - stats.stats.f.cdf(anova_results.loc["C(Part)", "F"], 
                                                            anova_results.loc["C(Part)", "df"], 
                                                            anova_results.loc["C(Part):C(Operator)", "df"])
    
    anova_results.loc["C(Operator)", "PR(>F)"] = 1 - stats.stats.f.cdf(anova_results.loc["C(Operator)", "F"], 
                                                                anova_results.loc["C(Operator)", "df"], 
                                                                anova_results.loc["C(Part):C(Operator)", "df"])
    
    anova_results.loc["Total"] = [anova_results.df.sum(), SS_Total, np.nan, np.nan, np.nan]
    anova_results.index = ["부품", "측정자", "부품:측정자", "반복성", "총계"]
    st.session_state.anova_w_inter_df = anova_results

    ###############################################################################
    # 교호작용이 없는 이원분산분석
    model = ols("Measurement ~ C(Part) + C(Operator)", data=df).fit()
    anova_results = sm.stats.anova_lm(model)  
    
    # 반복성(잔차) 분산 분석표에 추가
    SS_Total = np.sum((df["Measurement"] - np.mean(df["Measurement"])) ** 2)
    SS_Repeatability = anova_results.loc["Residual", "sum_sq"]
    df_Repeatability = anova_results.loc["Residual", "df"]
    
    anova_results.loc["Total"] = [anova_results.df.sum(), SS_Total, np.nan, np.nan, np.nan]
    anova_results.index = ["부품", "측정자", "반복성", "총계"]
    st.session_state.anova_wo_inter_df = anova_results

    ###############################################################################
    # 분산 성분 기여도
    RnRModel=RnR.RnRNumeric(
        mydf_Raw=df,
        mydict_key={"1":"Operator","2":"Part","3":"Measurement"},
        mydbl_tol=st.session_state.tolerance
        )
    RnRModel.RnRSolve()
    var_comp = RnRModel.RnR_varTable()
    var_comp.index = ["총 Gage R&R", "반복성", "재현성", "측정자", "부품:측정자", "부품-대-부품", "총 변동"]
    
    # 음수 분산 성분을 0으로 변경
    var_comp["Variance"] = np.maximum(var_comp["Variance"], 0)
    
    # 총 변동 재계산 (음수 성분의 절댓값을 더함)
    # abs_negative_sum = abs(var_comp.loc[var_comp["Variance"] < 0, "Variance"].sum())
    # var_comp.loc["총 변동", "Variance"] += abs_negative_sum
    total_variance = var_comp.loc["총 Gage R&R", "Variance"] + var_comp.loc["부품-대-부품", "Variance"]
    var_comp.loc["총 변동", "Variance"] = total_variance
    
    # 기여도 재계산
    var_comp["% Contribution"] = var_comp["Variance"] / total_variance * 100

    var_comp.columns = ["분산 성분", "%기여(분산 성분)"]
    ndc = int(np.floor(np.sqrt(2 * var_comp.loc["부품-대-부품", "분산 성분"] / var_comp.loc["총 변동", "분산 성분"])))
    st.session_state.ndc = 1 if ndc < 1 else ndc
    st.session_state.var_comp_df = var_comp

    ###############################################################################
    # 연구 변동
    # std_comp = RnRModel.RnR_SDTable()
    std_comp = pd.DataFrame()
    std_comp["표준편차(SD)"] = np.sqrt(var_comp["분산 성분"])
    
    std_comp["연구 변동(6*SD)"] = 6 * std_comp["표준편차(SD)"]
    total_studyvar = std_comp.loc["총 변동", "연구 변동(6*SD)"]
    
    std_comp["%연구 변동(%SV)"] = (std_comp["연구 변동(6*SD)"] / total_studyvar) * 100
    
    if st.session_state.tolerance is not None:
        std_comp["%공차(SV/공차)"] = (std_comp["연구 변동(6*SD)"] / st.session_state.tolerance) * 100
    
    std_comp.index = ["총 Gage R&R", "반복성", "재현성", "측정자", "부품:측정자", "부품-대-부품", "총 변동"]
    st.session_state.std_comp_df = std_comp


def gage_RnR_plot(df):
    col1, col2 = st.columns(2)
    
    with col1:
        draw_var_comp()
    
    with col2:
        draw_meas_by_part(df)

    col3, col4 = st.columns(2)
    
    with col3:
        draw_R_ctrl_chart(df)
    
    with col4:
        draw_meas_by_operator(df)

    col5, col6 = st.columns(2)
    
    with col5:
        draw_xbar_ctrl_chart(df)
    
    with col6:
        draw_interaction(df)


def gage_RnR_run():  
    df = convert_to_calculatable_df()
    # --------------------------------------
    gage_RnR_cal(df)
    st.write("**교호작용이 있는 이원분산분석**")
    st.data_editor(st.session_state.anova_w_inter_df, key="anova_w")
    st.write("**교호작용이 없는 이원분산분석**")
    st.data_editor(st.session_state.anova_wo_inter_df, key="anova_wo")
    st.write("**Gage R&R**")
    st.data_editor(st.session_state.var_comp_df, key="var_comp")
    if st.session_state.tolerance is not None:
        st.write(f"공정 공차: {str(st.session_state.tolerance)}")
    st.data_editor(st.session_state.std_comp_df, key="std_comp")
    st.write(f"구별 범주의 수: {str(st.session_state.ndc)}")
    # --------------------------------------
    gage_RnR_plot(df)