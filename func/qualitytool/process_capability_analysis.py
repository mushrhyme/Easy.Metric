# 공정능력분석을 위한 기능 코드

from utils.tools.UTIL import *
from scipy import stats, special
from plotly.subplots import make_subplots

import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

def calculate_d2(n, simulations=100000):
        samples = np.random.standard_normal((simulations, n))
        ranges = np.ptp(samples, axis=1)
        return np.mean(ranges)


def calculate_d3(n, simulations=100000):
    samples = np.random.standard_normal((simulations, n))
    ranges = np.ptp(samples, axis=1)
    return np.std(ranges)


def pooled_std_method(data, subgroup_size):    
    n = len(data)
    squared_diff_sum = np.sum((data - data.mean())**2)
    
    if subgroup_size == 1:
        # 이동 범위 계산
        moving_range = np.abs(data.diff().dropna())
        mr_bar = moving_range.mean()
        return mr_bar / 1.128
    else:
        # 불편화상수 계산 (감마 함수 사용)
        c4 = np.sqrt(2 / (subgroup_size - 1)) * special.gamma(subgroup_size / 2) / special.gamma((subgroup_size - 1) / 2)
        return np.sqrt(squared_diff_sum / (n - 1)) / c4

###########################################################################################################################
def process_capability_analysis_set():
    df = convert_to_calculatable_df()

    tab1, tab2 = st.tabs(["분석", "그래프"])
    with tab1:
        st.session_state.uniq_col = st.selectbox("단일 열", df.columns.tolist())
        st.session_state.subgroup_size = st.number_input("부분군 크기", value=1)
        st.session_state.lsl = st.number_input("규격 하한", value=None)
        st.session_state.usl = st.number_input("규격 상한", value=None)

        with st.popover("옵션"):
            st.session_state.target = st.number_input("목표값(표에 Cpm 추가X)", value=None)
            st.session_state.k = st.number_input("공정 능력 통계에 K x σ 공차 사용", value=6, step=1)
            # st.radio("표시", ["PPM", "백분율"])
            # st.radio("", ["공정 능력 통계량(Cp, PpXL)", "벤치마크 Z(σ수준 X E)"], label_visibility="collapsed")
            
            # option1 = st.radio("변환", ["변환 없음", "Box-Cox 누승 변환", "Johnson 변환(전체 산포 분석만 X J)"])
            # if option1 == "Box-Cox 누승 변환":
            #     with st.container(border=True):
            #         option2 = st.radio("", ["최적 λ 사용", "λ = 0", "λ = 0.5", "기타(-5와 5 사이의 값 입력 X H)"], label_visibility="collapsed")
            #         if option2 =="기타(-5와 5 사이의 값 입력 X H)":
            #             st.session_state.lambd = st.number_input("", min_value=-5, max_value=5, label_visibility="collapsed")
            # elif option1 == "Johnsom 변환(전체 산포 분석만 X J)":
            #     st.session_state.p_value = st.number_input("최량 적합을 선택하기 위한 P-값", value=0.10)
            
            # if st.checkbox("신뢰구간 포함"):
            #     with st.container(border=True):
            #         st.session_state.confidence_level = st.number_input("신뢰 수준", value=95.0, step=1.0)
            #         st.session_state.confidence_interval = st.selectbox("신뢰 구간", ["단측", "양측"])

    with tab2:
        color_set([st.session_state.uniq_col, "전체 ", "군내", "규격"])
        axis_set("", "", "")

def process_capability_analysis_cal(df):
    df = df[st.session_state.uniq_col]

    # 기본 통계량 계산
    mean = df.mean()
    overall_std = df.std()
    within_std = pooled_std_method(df, st.session_state.subgroup_size)
    target = st.session_state.target or 0

    def calc_z_scores(std):
        z_lsl = (mean - st.session_state.lsl) / std
        z_usl = (st.session_state.usl - mean) / std
        p_total = stats.norm.cdf(-z_lsl) + stats.norm.cdf(-z_usl)
        z_bench = -stats.norm.ppf(p_total)
        return np.round([z_bench, z_lsl, z_usl], 3)

    def calc_capability(std, is_overall=True):
        prefix = 'P' if is_overall else 'C'
        cp = (st.session_state.usl - st.session_state.lsl) / (st.session_state.k * std)
        cpl = (mean - st.session_state.lsl) / (st.session_state.k / 2 * std)
        cpu = (st.session_state.usl - mean) / (st.session_state.k / 2 * std)
        cpk = min(cpl, cpu)

        result = dict(zip(
            [f'{prefix}p', f'{prefix}PL', f'{prefix}PU', f'{prefix}pk'],
            np.round([cp, cpl, cpu, cpk], 3)
        ))

        if is_overall:
            cpm = cp / np.sqrt(1 + ((mean - target) / std) ** 2) if target is not None else '*'
            result['Cpm'] = np.round(cpm, 3) if target is not None else '*'

        return result

    def calc_ppm(data=None, std=None):
        if data is not None:  # 관측 PPM
            below = (data < st.session_state.lsl).sum() / len(data) * 1e6
            above = (data > st.session_state.usl).sum() / len(data) * 1e6
        else:  # 기대 PPM
            z_lsl = (st.session_state.lsl - mean) / std
            z_usl = (st.session_state.usl - mean) / std
            below = stats.norm.cdf(z_lsl) * 1e6
            above = (1 - stats.norm.cdf(z_usl)) * 1e6
        return below, above, below + above

    # 전체 및 군내 공정 능력 계산
    z_overall = dict(zip(['Z.Bench', 'Z.LSL', 'Z.USL'], calc_z_scores(overall_std)))
    z_within = dict(zip(['Z.Bench', 'Z.LSL', 'Z.USL'], calc_z_scores(within_std)))

    overall_caps = {**z_overall, **calc_capability(overall_std, True)}
    within_caps = {**z_within, **calc_capability(within_std, False)}

    # PPM 계산
    obs_ppm = calc_ppm(data=df)
    exp_ppm = calc_ppm(std=overall_std)
    within_ppm = calc_ppm(std=within_std)

    # 결과 데이터프레임 생성
    st.session_state.process_data_df = pd.DataFrame({
        "규격 하한": st.session_state.lsl,
        "목표값": "*" if target is None else target,
        "규격 상한": st.session_state.usl,
        "표본 평균": np.round(mean, 3),
        "표본 N": len(df),
        "표준 편차(전체)": np.round(overall_std, 3),
        "표준 편차(군내)": np.round(within_std, 3),
    }, index=["공정 데이터"]).T

    st.session_state.overall_capt_df = pd.DataFrame(overall_caps, index=["공정 능력"]).T
    st.session_state.within_capt_df = pd.DataFrame(within_caps, index=["공정 능력"]).T

    st.session_state.perf_df = pd.DataFrame({
        "관측 성능": obs_ppm,
        "기대 성능(전체)": exp_ppm,
        "기대 성능(군내)": within_ppm
    }, index=["PPM < 규격 하한", "PPM > 규격 상한", "PPM 총계"]).round(2)


def process_capability_analysis_plot(df):
    df = df[st.session_state.uniq_col]
    overall_mean = df.mean()
    overall_std = df.std()
    within_std = pooled_std_method(df, st.session_state.subgroup_size)

    # KDE 계산을 위한 범위 설정
    range_extent = max(
        abs(st.session_state.lsl - overall_mean),
        abs(st.session_state.usl - overall_mean),
        abs(df.min() - overall_mean),
        abs(df.max() - overall_mean)
    )
    line_start = overall_mean - range_extent * 1.5
    line_end = overall_mean + range_extent * 1.5
    x = np.linspace(line_start, line_end, 1000)
    overall_kde = stats.norm.pdf(x, df.mean(), overall_std)
    within_group_kde = stats.norm.pdf(x, df.mean(), within_std)

    # 히스토그램 빈 계산 (1 단위로)
    min_value = np.floor(df.min())
    max_value = np.ceil(df.max())
    bins = np.arange(min_value, max_value + 2) - 0.5  # +2 to include the last bin
    hist, bin_edges = np.histogram(df, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 최대 빈도와 최대 밀도 계산
    max_freq = max(hist)
    max_density = max(max(overall_kde), max(within_group_kde))

    # 축 범위 조정을 위한 스케일 팩터 계산
    scale_factor = max_freq / max_density

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=hist,
            name="히스토그램",
            opacity=0.7,
            showlegend=False,
            hovertemplate="막대 구간: %{x:.1f} - %{customdata:.1f}<br>빈도: %{y}<extra></extra>",
            customdata=bin_edges[1:],
            width=1,
            marker_color=st.session_state["color_0"]
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=x, y=overall_kde * scale_factor, mode="lines", name="전체", line=dict(color=st.session_state["color_1"]), hoverinfo="skip"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=x, y=within_group_kde * scale_factor, mode="lines", name="군내", line=dict(color=st.session_state["color_2"], dash="dash"), hoverinfo="skip"),
        secondary_y=False,
    )

    fig.add_vline(
        x=st.session_state.lsl,
        line_dash="dot",
        line_color=st.session_state["color_3"],
        annotation_text="규격 하한",
        annotation=dict(
            yref="paper",
            y=1,
            yanchor="bottom",
            showarrow=False,
            font=dict(color=st.session_state["color_3"])
        )
    )
    fig.add_vline(
        x=st.session_state.usl,
        line_dash="dot",
        line_color=st.session_state["color_3"],
        annotation_text="규격 상한",
        annotation=dict(
            yref="paper",
            y=1,
            yanchor="bottom",
            showarrow=False,
            font=dict(color=st.session_state["color_3"])
        )
    )
    if st.session_state.target is not None:
        fig.add_vline(
            x=st.session_state.target,
            line_dash="dot",
            line_color="green",
            annotation_text="목표값",
            annotation=dict(
                yref="paper",
                y=1,
                yanchor="bottom",
                showarrow=False,
                font=dict(color="green")
            )
        )
    
    # y축 범위 설정
    y_range = [0, max_freq * 1.1]  # 10% 여유 추가
    
    fig.update_layout(
        title=st.session_state.title,
        hovermode="x unified",
        yaxis=dict(range=y_range),
        yaxis2=dict(range=[0, max_density * 1.1], overlaying='y', scaleratio=1, scaleanchor='y'),
    )
    fig.update_xaxes(title=st.session_state.x_label)
    fig.update_yaxes(title=st.session_state.y_label, secondary_y=False)
    fig.update_yaxes(secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)
    # st.markdown(get_image_download_link(fig), unsafe_allow_html=True)


def process_capability_analysis_run():  
    df = convert_to_calculatable_df()
    # --------------------------------------
    if st.session_state.lsl is None or st.session_state.usl is None:
        st.write("규격 상한 또는 규격 하한을 선택하지 않았습니다.")
    else:
        st.markdown(f'**{st.session_state.uniq_col}의 공정 능력 보고서**')
        process_capability_analysis_cal(df)
        # --------------------------------------
        process_capability_analysis_plot(df)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**공정데이터**")
            st.data_editor(st.session_state.process_data_df, key="process_data")
        with col2:
            st.write("**전체 공정 능력**")
            st.data_editor(st.session_state.overall_capt_df, key="overall_capt")
        with col3:
            st.write("**잠재적(군내) 공정 능력**")
            st.data_editor(st.session_state.within_capt_df, key="within_capt")    
        
        st.write("**성능**")
        st.data_editor(st.session_state.perf_df, key="perf")
        st.write(f"실제 공정 산포가 {st.session_state.k} 시그마로 표시됩니다.")

    
    
    