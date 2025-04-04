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

def covariance_analysis_cal(df):
    pass

def covariance_analysis_plot(df):
    pass

def covariance_analysis_run():  
    df = convert_to_calculatable_df()
    # --------------------------------------
    covariance_analysis_cal(df)
    st.write("**공분산**")
    st.data_editor(df.cov())
    # --------------------------------------
    covariance_analysis_plot(df)