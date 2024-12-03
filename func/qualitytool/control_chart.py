# 파레토차트를 위한 기능 코드

from utils.tools.UTIL import *
from scipy import stats
from plotly.subplots import make_subplots

import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go

def get_control_chart_constants(n):
    """
    부분군 크기에 따른 관리도 상수 계산
    n > 10인 경우 근사식 사용
    """
    if n <= 10:
        # 기존의 상수표 사용
        constants_xbar_r = {
            2: {'A2': 1.880, 'D3': 0, 'D4': 3.267},
            3: {'A2': 1.023, 'D3': 0, 'D4': 2.574},
            4: {'A2': 0.729, 'D3': 0, 'D4': 2.282},
            5: {'A2': 0.577, 'D3': 0, 'D4': 2.114},
            6: {'A2': 0.483, 'D3': 0, 'D4': 2.004},
            7: {'A2': 0.419, 'D3': 0.076, 'D4': 1.924},
            8: {'A2': 0.373, 'D3': 0.136, 'D4': 1.864},
            9: {'A2': 0.337, 'D3': 0.184, 'D4': 1.816},
            10: {'A2': 0.308, 'D3': 0.223, 'D4': 1.777}
        }
        constants_xbar_s = {
            2: {'A3': 2.659, 'B3': 0, 'B4': 3.267},
            3: {'A3': 1.954, 'B3': 0, 'B4': 2.568},
            4: {'A3': 1.628, 'B3': 0, 'B4': 2.266},
            5: {'A3': 1.427, 'B3': 0, 'B4': 2.089},
            6: {'A3': 1.287, 'B3': 0.030, 'B4': 1.970},
            7: {'A3': 1.182, 'B3': 0.118, 'B4': 1.882},
            8: {'A3': 1.099, 'B3': 0.185, 'B4': 1.815},
            9: {'A3': 1.032, 'B3': 0.239, 'B4': 1.761},
            10: {'A3': 0.975, 'B3': 0.284, 'B4': 1.716}
        }
        return constants_xbar_r.get(n), constants_xbar_s.get(n)
    else:
        # n > 10 일 때의 근사식
        # 중심극한정리에 기반한 계산
        c4 = np.sqrt(2 / (n - 1)) * gamma(n/2) / gamma((n-1)/2)
        
        # X-bar R 상수 계산
        d2 = 2 * gamma((n+1)/2) / gamma(n/2) / np.sqrt(np.pi/2)
        A2 = 3 / (d2 * np.sqrt(n))
        D3 = max(0, 1 - 3 * np.sqrt(1 - c4**2))
        D4 = 1 + 3 * np.sqrt(1 - c4**2)
        
        # X-bar S 상수 계산
        A3 = 3 / (c4 * np.sqrt(n))
        B3 = max(0, 1 - 3 * np.sqrt(1 - c4**2))
        B4 = 1 + 3 * np.sqrt(1 - c4**2)
        
        xbar_r_constants = {'A2': A2, 'D3': D3, 'D4': D4}
        xbar_s_constants = {'A3': A3, 'B3': B3, 'B4': B4}
        
        return xbar_r_constants, xbar_s_constants

def xbar_r_plot(df):
    # 상수 정의
    
    subgroup_size = st.session_state.subgroup_size
    predictors = st.session_state.predictor
    
    const, _ = get_control_chart_constants(subgroup_size)
    # 각 변수별로 차트 생성
    for predictor in predictors:
        # streamlit 컨테이너 생성
        with st.container(border=True):
            st.subheader(f"{predictor}의 Xbar-R 관리도")
            
            # 데이터 준비
            n_subgroups = len(df) // subgroup_size
            data = df[predictor].values[:n_subgroups * subgroup_size]
            data_matrix = data.reshape(-1, subgroup_size)
            
            # 계산
            xbar = np.mean(data_matrix, axis=1)
            ranges = np.ptp(data_matrix, axis=1)
            
            grand_mean = np.mean(xbar)
            r_bar = np.mean(ranges)
            
            # 관리 한계선 계산
            xcl = grand_mean
            xucl = grand_mean + const['A2'] * r_bar
            xlcl = grand_mean - const['A2'] * r_bar
            
            rcl = r_bar
            rucl = const['D4'] * r_bar
            rlcl = const['D3'] * r_bar
            
            # Xbar-R 차트 생성
            fig = make_subplots(
                rows=2, 
                cols=1,
                subplot_titles=(f'{predictor} - X-bar Chart', 
                              f'{predictor} - R Chart'),
                vertical_spacing=0.25
            )
            
            # X-bar 차트
            fig.add_trace(
                go.Scatter(
                    y=xbar,
                    mode='lines+markers',
                    name='X-bar',
                    line=dict(color='blue')
                ),
                row=1,
                col=1
            )
            
            # X-bar 관리한계선
            for value, name, color in [
                (xucl, 'UCL', 'red'),
                (xcl, 'CL', 'green'),
                (xlcl, 'LCL', 'red')
            ]:
                fig.add_trace(
                    go.Scatter(
                        y=[value] * len(xbar),
                        mode='lines',
                        name=name,
                        line=dict(color=color, dash='dash')
                    ),
                    row=1,
                    col=1
                )
            
            # R 차트
            fig.add_trace(
                go.Scatter(
                    y=ranges,
                    mode='lines+markers',
                    name='Range',
                    line=dict(color='blue')
                ),
                row=2,
                col=1
            )
            
            # R 관리한계선
            for value, name, color in [
                (rucl, 'UCL', 'red'),
                (rcl, 'CL', 'green'),
                (rlcl, 'LCL', 'red')
            ]:
                fig.add_trace(
                    go.Scatter(
                        y=[value] * len(ranges),
                        mode='lines',
                        name=name,
                        line=dict(color=color, dash='dash')
                    ),
                    row=2,
                    col=1
                )
            
            # 레이아웃 업데이트
            fig.update_layout(
                height=600,
                showlegend=False,
                margin=dict(t=30)
            )
            
            # y축 레이블 업데이트
            fig.update_yaxes(title_text="표본 평균", row=1, col=1)
            fig.update_yaxes(title_text="표본 범위", row=2, col=1)
            fig.update_xaxes(title_text="표본", row=2, col=1)
            
            # 차트 표시
            st.plotly_chart(fig, use_container_width=True)



def xbar_s_plot(df):
    subgroup_size = st.session_state.subgroup_size
    predictors = st.session_state.predictor
    
    _, const = get_control_chart_constants()
    
    # 각 변수별로 차트 생성
    for predictor in predictors:
        with st.container(border=True):
            st.subheader(f"{predictor}의 Xbar-S 관리도")
            
            # 데이터 준비
            n_subgroups = len(df) // subgroup_size
            data = df[predictor].values[:n_subgroups * subgroup_size]
            data_matrix = data.reshape(-1, subgroup_size)
            
            # 계산
            xbar = np.mean(data_matrix, axis=1)
            s = np.std(data_matrix, axis=1, ddof=1)  # ddof=1 for sample standard deviation
            
            grand_mean = np.mean(xbar)
            s_bar = np.mean(s)
            
            # 관리 한계선 계산
            xcl = grand_mean
            xucl = grand_mean + const['A3'] * s_bar
            xlcl = grand_mean - const['A3'] * s_bar
            
            scl = s_bar
            sucl = const['B4'] * s_bar
            slcl = const['B3'] * s_bar
            
            # Xbar-S 차트 생성
            fig = make_subplots(
                rows=2, 
                cols=1,
                subplot_titles=(f'{predictor} - X-bar Chart', 
                              f'{predictor} - S Chart'),
                vertical_spacing=0.25
            )
            
            # X-bar 차트
            fig.add_trace(
                go.Scatter(
                    y=xbar,
                    mode='lines+markers',
                    name='X-bar',
                    line=dict(color='blue')
                ),
                row=1,
                col=1
            )
            
            # X-bar 관리한계선
            for value, name, color in [
                (xucl, 'UCL', 'red'),
                (xcl, 'CL', 'green'),
                (xlcl, 'LCL', 'red')
            ]:
                fig.add_trace(
                    go.Scatter(
                        y=[value] * len(xbar),
                        mode='lines',
                        name=name,
                        line=dict(color=color, dash='dash')
                    ),
                    row=1,
                    col=1
                )
            
            # S 차트
            fig.add_trace(
                go.Scatter(
                    y=s,
                    mode='lines+markers',
                    name='Standard Deviation',
                    line=dict(color='blue')
                ),
                row=2,
                col=1
            )
            
            # S 관리한계선
            for value, name, color in [
                (sucl, 'UCL', 'red'),
                (scl, 'CL', 'green'),
                (slcl, 'LCL', 'red')
            ]:
                fig.add_trace(
                    go.Scatter(
                        y=[value] * len(s),
                        mode='lines',
                        name=name,
                        line=dict(color=color, dash='dash')
                    ),
                    row=2,
                    col=1
                )
            
            # 레이아웃 업데이트
            fig.update_layout(
                height=600,
                showlegend=False,
                margin=dict(t=30)
            )
            
            # y축 레이블 업데이트
            fig.update_yaxes(title_text="표본 평균", row=1, col=1)
            fig.update_yaxes(title_text="표본 표준편차", row=2, col=1)
            fig.update_xaxes(title_text="표본", row=2, col=1)
            
            # 차트 표시
            st.plotly_chart(fig, use_container_width=True)

def imr_plot(df):
    # 상수 정의
    E2 = 2.66  # Moving Range 관리도 상수
    D3 = 0     # Moving Range 관리도 하한 상수
    D4 = 3.27  # Moving Range 관리도 상한 상수
    
    predictors = st.session_state.predictor
    
    # 각 변수별로 차트 생성
    for predictor in predictors:
        with st.container(border=True):
            st.subheader(f"{predictor}의 I-MR 관리도")
            
            # 데이터 준비
            data = df[predictor].values
            
            # 이동범위(Moving Range) 계산
            moving_range = np.abs(np.diff(data))
            
            # 계산
            indv_mean = np.mean(data)
            mr_bar = np.mean(moving_range)
            
            # 관리 한계선 계산 (Individual Chart)
            icl = indv_mean
            iucl = indv_mean + E2 * mr_bar
            ilcl = indv_mean - E2 * mr_bar
            
            # 관리 한계선 계산 (Moving Range Chart)
            mrcl = mr_bar
            mrucl = D4 * mr_bar
            mrlcl = D3 * mr_bar
            
            # I-MR 차트 생성
            fig = make_subplots(
                rows=2, 
                cols=1,
                subplot_titles=(f'{predictor} - Individual Chart', 
                              f'{predictor} - Moving Range Chart'),
                vertical_spacing=0.25
            )
            
            # Individual 차트
            fig.add_trace(
                go.Scatter(
                    y=data,
                    mode='lines+markers',
                    name='Individual Value',
                    line=dict(color='blue')
                ),
                row=1,
                col=1
            )
            
            # Individual 관리한계선
            for value, name, color in [
                (iucl, 'UCL', 'red'),
                (icl, 'CL', 'green'),
                (ilcl, 'LCL', 'red')
            ]:
                fig.add_trace(
                    go.Scatter(
                        y=[value] * len(data),
                        mode='lines',
                        name=name,
                        line=dict(color=color, dash='dash')
                    ),
                    row=1,
                    col=1
                )
            
            # Moving Range 차트
            fig.add_trace(
                go.Scatter(
                    y=moving_range,
                    mode='lines+markers',
                    name='Moving Range',
                    line=dict(color='blue')
                ),
                row=2,
                col=1
            )
            
            # Moving Range 관리한계선
            for value, name, color in [
                (mrucl, 'UCL', 'red'),
                (mrcl, 'CL', 'green'),
                (mrlcl, 'LCL', 'red')
            ]:
                fig.add_trace(
                    go.Scatter(
                        y=[value] * len(moving_range),
                        mode='lines',
                        name=name,
                        line=dict(color=color, dash='dash')
                    ),
                    row=2,
                    col=1
                )
            
            # 레이아웃 업데이트
            fig.update_layout(
                height=600,
                showlegend=False,
                margin=dict(t=30)
            )
            
            # y축 레이블 업데이트
            fig.update_yaxes(title_text="개별값", row=1, col=1)
            fig.update_yaxes(title_text="이동범위", row=2, col=1)
            fig.update_xaxes(title_text="관측값", row=2, col=1)
            
            # 차트 표시
            st.plotly_chart(fig, use_container_width=True)



###########################################################################################################################
def control_chart_set():
    df = convert_to_calculatable_df()
    st.session_state.option = st.selectbox("관리도 유형", ["Xbar-R", "Xbar-S", "I-MR"])
    st.session_state.predictor = st.multiselect("요인", df.columns.tolist())
    st.session_state.subgroup_size = st.number_input("부분군 크기", value=1)
    if st.session_state.subgroup_size < 2:
        st.error("부분군 크기는 2 이상이어야 합니다.")
    elif st.session_state.subgroup_size > 8 and st.session_state.option=="Xbar-R":
        st.error(f"""
        부분군 크기가 8보다 큽니다. 
        - Xbar-S 관리도 사용을 권장합니다.
        - 가능하다면 부분군 크기를 4~6 사이로 조정하는 것을 고려해보세요.
        """)


def control_chart_cal(df):
    pass


def control_chart_plot(df):
    if st.session_state.option=="Xbar-R":
        xbar_r_plot(df)
    elif st.session_state.option=="Xbar-S":
        xbar_s_plot(df)
    elif st.session_state.option=="I-MR":
        imr_plot(df)
        
def control_chart_run():  
    df = convert_to_calculatable_df()
    # --------------------------------------
    
    # control_chart_cal(df)
    control_chart_plot(df)