# 카이-제곱 적합도 검정을 위한 기능 코드

from utils.tools.UTIL import *
from scipy import stats
from plotly.subplots import make_subplots

import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go


def chi_square_goodness_of_fit_test_set():
    pass

def chi_square_goodness_of_fit_test_cal(df):
    pass

def chi_square_goodness_of_fit_test_plot(df):
    pass

def chi_square_goodness_of_fit_test_run():  
    # 매우 중요 : 데이터프레임 가져오기 ------------
    df = convert_to_calculatable_df()
    # --------------------------------------
    
    chi_square_goodness_of_fit_test_cal(df)
    chi_square_goodness_of_fit_test_plot(df)