from utils.tools.UTIL import *
from scipy import stats
from plotly.subplots import make_subplots

import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go


def covariance_analysis_set():
    df = convert_to_calculatable_df()
    st.session_state.predictor = st.multiselect("변수", df.columns.tolist())
    for col in st.session_state.predictor:
        if df[col].dtype == "object":
            st.error(f"{col}는 계량형 변수가 아닙니다. 계량형 변수만 선택해주세요.")
            return

def covariance_analysis_cal(df):
    st.session_state.stats_df = df.cov()

def covariance_analysis_plot(df):
    pass

def covariance_analysis_run():  
    df = convert_to_calculatable_df()
    df = df[st.session_state.predictor]
    # --------------------------------------
    if len(st.session_state.predictor)>0:
        for col in st.session_state.predictor:
            if df[col].dtype == "object":
                st.error(f"{col}는 계량형 변수가 아닙니다. 계량형 변수만 선택해주세요.")
                return
        covariance_analysis_cal(df)
        st.write("**공분산**")
        st.data_editor(st.session_state.stats_df)
        # --------------------------------------
        covariance_analysis_plot(df)
    else:
        st.error("분석에 사용할 변수가 지정되지 않았습니다.")