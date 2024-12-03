# Description: 관리도를 위한 모듈입니다.

import streamlit as st
from utils.tools.UTIL import menu

func_name = 'func.etc'
analysis_options = {
        '부분군 계량형 관리도': 'subgroup_control_chart',
        '개별값 계량형 관리도': 'individual_control_chart',
        '계수형 관리도': 'count_control_chart',
        '주성분분석': 'principal_component_analysis',
        'k-평균 군집 분석': 'kmeans_clustering',
        '시계열 분석': 'time_series_analysis',
        '추세분석': 'trend_analysis',
        '분해분석': 'decomposition_analysis',
        '카이-제곱 연관성 검정': 'chi_square_association_test',
        '카이-제곱 적합도 검정': 'chi_square_goodness_of_fit_test',
    }

def import_analysis_module(module_name):
    module = __import__(f'{func_name}.{module_name}', fromlist=[''])
    return (
        getattr(module, f'{module_name}_set', None),
        getattr(module, f'{module_name}_cal', None),
        getattr(module, f'{module_name}_plot', None),
        getattr(module, f'{module_name}_run', None)
    )

def display_etc():
    with st.container(border=True):
        selected = st.selectbox('분석 방법을 선택해주세요:', options=list(analysis_options.keys()))

    with st.container(border=True):
        module_name = analysis_options[selected]
        set_func, cal_func, plot_func, run_func = import_analysis_module(module_name)

        if set_func:
            set_func()

        if cal_func and plot_func and run_func:
            menu(anlysis=cal_func, plot=plot_func, run=run_func)
