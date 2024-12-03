import streamlit as st
from utils.tools.UTIL import menu

func_name = 'func.basic'
analysis_options = {
        '기술통계량 표시': 'descriptive_statistics',
        # '그래프요약': "graph_summary",
        '1-표본 검증': 'onesample_test',
        '2-표본 검증': 'twosample_test',
        '쌍체T 검증': 'paired_ttest',
        '단일비율 검증': 'oneproportion_test',
        '두 비율 검증': 'twoproportion_test',
        '상관분석': 'correlation_analysis',
        '공분산분석': 'covariance_analysis',
        '정규성검증': 'normality_test',

    }

def import_analysis_module(module_name):
    module = __import__(f'{func_name}.{module_name}', fromlist=[''])
    return (
        getattr(module, f'{module_name}_set', None),
        getattr(module, f'{module_name}_cal', None),
        getattr(module, f'{module_name}_plot', None),
        getattr(module, f'{module_name}_run', None)
    )

def display_basic_statistics():
    with st.container(border=True):
        selected = st.selectbox('분석 방법을 선택해주세요:', options=list(analysis_options.keys()))

    with st.container(border=True):
        module_name = analysis_options[selected]
        set_func, cal_func, plot_func, run_func = import_analysis_module(module_name)

        if set_func:
            set_func()

        if cal_func and plot_func and run_func:
            menu(anlysis=cal_func, plot=plot_func, run=run_func)