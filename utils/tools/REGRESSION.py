# Description: 회귀분석을 위한 모듈입니다.

import streamlit as st
from utils.tools.UTIL import menu

func_name = 'func.regression'
analysis_options = {
        '적합선 그림': 'fitted_line_plot',
        '회귀분석': 'regression_analysis',
        '분산분석': 'anova',
        '등분산검증': 'homoscedasticity_test',
        '주효과도분석': 'main_effect_analysis',
        '교호작용분석': 'interaction_analysis',
    }

def import_analysis_module(module_name):
    module = __import__(f'{func_name}.{module_name}', fromlist=[''])
    return (
        getattr(module, f'{module_name}_set', None),
        getattr(module, f'{module_name}_cal', None),
        getattr(module, f'{module_name}_plot', None),
        getattr(module, f'{module_name}_run', None)
    )

def display_regression_analysis():
    with st.container(border=True):
        selected = st.selectbox('분석 방법을 선택해주세요:', options=list(analysis_options.keys()))

    with st.container(border=True):
        module_name = analysis_options[selected]
        set_func, cal_func, plot_func, run_func = import_analysis_module(module_name)

        if set_func:
            set_func()

        if cal_func and plot_func and run_func:
            menu(anlysis=cal_func, plot=plot_func, run=run_func)