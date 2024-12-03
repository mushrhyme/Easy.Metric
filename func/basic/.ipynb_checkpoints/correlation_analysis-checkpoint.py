from utils.tools.UTIL import *
import streamlit as st
import numpy as np
import pandas as pd

from scipy import stats

import seaborn as sns
import matplotlib 
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 한글 폰트 설정
font_name = 'NanumGothic'
plt.rc('font', family=font_name)  
plt.rcParams['axes.unicode_minus'] = False  

def correlation_analysis_set():
    df = convert_to_calculatable_df()
    tab1, tab2 = st.tabs(["분석", "그래프"])
    with tab1:
        st.session_state.predictor = st.multiselect("변수", df.columns.tolist())
        st.session_state.correlation = st.radio("상관계수", ["Pearson", "Spearman"])
        st.session_state.significance_level = 1 - st.number_input(f"신뢰수준", value=95.0, step=1.0)/100
    with tab2:
        color_set([""])

def correlation_analysis_cal(df):
    # 상관 분석 수행
    if st.session_state.correlation == 'Pearson':
        corr_matrix = df.corr(method='pearson')
        corr_method = stats.pearsonr
    elif st.session_state.correlation == 'Spearman':
        corr_matrix = df.corr(method='spearman')
        corr_method = stats.spearmanr
    st.session_state.stats_df = corr_matrix.where(~np.triu(np.ones(corr_matrix.shape)).astype(bool)).dropna(how='all').dropna(axis=1, how='all')
    

def correlation_analysis_plot(df):
    plt.rcParams['font.family'] = font_name
    conf_level = int((1-st.session_state.significance_level)*100)
    
    # Create the pairplot
    g = sns.pairplot(
        df, 
        diag_kind='hist',
        plot_kws={"color": st.session_state["color_0"]},
        diag_kws={"color": st.session_state["color_0"]},
    )
    
    for i, j in zip(*np.triu_indices_from(g.axes, k=1)):
        var1 = df.columns[i]
        var2 = df.columns[j]
        
        corr_value = st.session_state.stats_df.loc[var2, var1]

        g.axes[i, j].text(
            0.5, 0.95,
            f'r = {corr_value:.3f}',  
            horizontalalignment='center',
            verticalalignment='top',
            transform=g.axes[i, j].transAxes,
            fontsize=10,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
        )
        
        g.axes[j, i].text(
            0.5, 0.95,
            f'r = {corr_value:.3f}',
            horizontalalignment='center',
            verticalalignment='top',
            transform=g.axes[j, i].transAxes,
            fontsize=10,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
        )
    
    fig_width, fig_height = g.fig.get_size_inches()
    
    # figure 높이의 5%를 타이틀 간격으로 사용
    title_spacing = fig_height * 0.05
    
    # 메인 타이틀과 부제목의 위치를 동적으로 설정
    main_title_y = 1 + (title_spacing / fig_height)
    subtitle_y = 1 + (title_spacing / fig_height) * 0.5
    
    g.fig.suptitle(
        f"{','.join(st.session_state.predictor)}의 산점도 행렬",
        y=main_title_y,
        fontsize=16,
        fontfamily=font_name,
        fontproperties=matplotlib.font_manager.FontProperties(fname=matplotlib.font_manager.findfont(font_name))
    )
    
    g.fig.text(
        0.5, 0.99,
        f"{st.session_state.correlation} 상관 계수에 대한 {conf_level}% CI",
        ha='center',
        fontsize=14,
        fontproperties=matplotlib.font_manager.FontProperties(fname=matplotlib.font_manager.findfont(font_name))
    )
    
    st.pyplot(g.fig)

def correlation_analysis_plot(df):
    plt.rcParams['font.family'] = font_name
    conf_level = int((1-st.session_state.significance_level)*100)
    
    # Create the pairplot
    g = sns.pairplot(
        df, 
        diag_kind='hist',
        plot_kws={"color": st.session_state["color_0"]},
        diag_kws={"color": st.session_state["color_0"]},
    )
    
    for i, j in zip(*np.triu_indices_from(g.axes, k=1)):
        var1 = df.columns[i]
        var2 = df.columns[j]
        
        corr_value = st.session_state.stats_df.loc[var2, var1]
        g.axes[i, j].text(
            0.5, 0.95,
            f'r = {corr_value:.3f}',  
            horizontalalignment='center',
            verticalalignment='top',
            transform=g.axes[i, j].transAxes,
            fontsize=10,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
        )
        
        g.axes[j, i].text(
            0.5, 0.95,
            f'r = {corr_value:.3f}',
            horizontalalignment='center',
            verticalalignment='top',
            transform=g.axes[j, i].transAxes,
            fontsize=10,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
        )
    
    fig_width, fig_height = g.fig.get_size_inches()
    
    # 변수 개수에 따라 타이틀 간격 동적 조정
    var_count = len(df.columns)
    if var_count == 2:
        title_spacing = fig_height * 0.2  # 2개 변수일 때는 더 큰 간격
    else:
        title_spacing = fig_height * 0.14  # 3개 이상일 때는 기존 간격
    
    # 메인 타이틀과 부제목의 위치를 동적으로 설정
    main_title_y = 1 + (title_spacing / fig_height)
    subtitle_y = 1 + (title_spacing / fig_height) * 0.6  # 부제목 위치 조정
    
    g.fig.suptitle(
        f"{','.join(st.session_state.predictor)}의 산점도 행렬",
        y=main_title_y,
        fontsize=16,
        fontfamily=font_name,
        fontproperties=matplotlib.font_manager.FontProperties(fname=matplotlib.font_manager.findfont(font_name))
    )
    
    g.fig.text(
        0.5, subtitle_y,  # subtitle_y 사용
        f"{st.session_state.correlation} 상관 계수에 대한 {conf_level}% CI",
        ha='center',
        fontsize=14,
        fontproperties=matplotlib.font_manager.FontProperties(fname=matplotlib.font_manager.findfont(font_name))
    )
    
    st.pyplot(g.fig)

def correlation_analysis_run():  
    df = convert_to_calculatable_df()
    # --------------------------------------
    if len(st.session_state.predictor) > 1:
        df = df[st.session_state.predictor]
        correlation_analysis_cal(df)
        # --------------------------------------
        with st.container(border=True):
            correlation_analysis_plot(df)
        st.write("**상관계수**")
        st.data_editor(st.session_state.stats_df, key="stats", use_container_width=True)
    else:
        st.error("분석을 위해 최소 2개 이상의 변수를 선택해주세요.")