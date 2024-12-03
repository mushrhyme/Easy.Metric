# Description: 품질도구를 위한 모듈입니다.

import streamlit as st
from utils.tools.UTIL import menu

func_name = 'func.qualitytool'
analysis_options = {
        '런차트': 'run_chart',
        '파레토 차트': 'pareto_chart',
        '공정능력분석': 'process_capability_analysis',
        'Gage R&R': 'gage_RnR',
        '관리도': 'control_chart',
    }

def import_analysis_module(module_name):
    module = __import__(f'{func_name}.{module_name}', fromlist=[''])
    return (
        getattr(module, f'{module_name}_set', None),
        getattr(module, f'{module_name}_cal', None),
        getattr(module, f'{module_name}_plot', None),
        getattr(module, f'{module_name}_run', None)
    )

def display_quality_tool():
    with st.container(border=True):
        selected = st.selectbox('분석 방법을 선택해주세요:', options=list(analysis_options.keys()))

    with st.container(border=True):
        module_name = analysis_options[selected]
        set_func, cal_func, plot_func, run_func = import_analysis_module(module_name)

        if set_func:
            set_func()

        if cal_func and plot_func and run_func:
            menu(anlysis=cal_func, plot=plot_func, run=run_func)