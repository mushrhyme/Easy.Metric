# 분해분석을 위한 기능 코드

from utils.tools.UTIL import *
from scipy import stats
from plotly.subplots import make_subplots

import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go


def decomposition_analysis_set():
    pass

def decomposition_analysis_cal(df):
    pass

def decomposition_analysis_plot(df):
    pass

def decomposition_analysis_run():  
    # 매우 중요 : 데이터프레임 가져오기 ------------
    df = convert_to_calculatable_df()
    # --------------------------------------
    
    decomposition_analysis_cal(df)
    decomposition_analysis_plot(df)