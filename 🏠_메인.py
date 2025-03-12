import streamlit as st
import numpy as np
import pandas as pd
from utils.tools.UTIL import *
from utils.tools.head import *
from utils.tools import (
    display_basic_statistics,
    display_regression_analysis,
    display_DOE,
    display_etc,
    display_quality_tool)
from error_handler import main_with_error_handling
from utils.tools.ip_tracker import *
def initialize_session_state():
    default_session = {
        'id': '',
        'name': '',
        'DB': '',
        'number_of_columns': 15,
        'number_of_rows': 10000,
        'number_of_newClick': 0,
        "base_df": pd.DataFrame(),
        "correlation": None,
        "color_1":None,
        "color_2": None,
        "color_3": None,
        "color_4": None,
        "color_5": None,
        "color_6": None,
        "color_7": None,
        "color_8": None,
        "color_9": None,
        # test
        "test_type": None,
        "pop_std": None,
        "sample1": None,
        "sample2": None,
        "stats_df": pd.DataFrame(),
        "diff_df": pd.DataFrame(),
        "test_df": pd.DataFrame(),
        "significance_level": 0.05,
        "confidence_level": 95,
        "null_diff": 0.0,
        "alternative": "",
        "equal_var": False,
        # regression
        "degree": None,
        "target": None,
        "predictor": [],
        "X": pd.DataFrame(),
        "X_for_opt": pd.DataFrame(),
        "interactions": None,
        "equation": "",
        "model": None,
        "residuals": None,
        "var_selection": None,
        "coding_type": None,
        "current_values": None,
        "coeff_df": pd.DataFrame(),
        "reg_df": pd.DataFrame(),
        "diag_df": pd.DataFrame(),
        "anov_df": pd.DataFrame(),
        "factor_df": pd.DataFrame(),
        "lsl": 0.0,
        "usl": 0.0,
        # gage r&r
        "part": None,
        "operator": None,
        "measurement": None,
        "tolerance": None,
        "ndc": None,
        "anova_w_inter_df": pd.DataFrame(),
        "anova_wo_inter_df": pd.DataFrame(),
        "var_comp_df": pd.DataFrame(),
        "std_comp_df": pd.DataFrame(),
        "tabs_":'기술통계',
        "k": 0.0,
        "lambd": "X",
        "uniq_col": None,
        "sigma_level": 0.0,
        "process_data_df": pd.DataFrame(),
        "overall_capt_df": pd.DataFrame(),
        "within_capt_df": pd.DataFrame(),
        "perf_df": pd.DataFrame(),
        # control chart
        "option": None,

    }
    for key, value in default_session.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if 'df' not in st.session_state:
        st.session_state['col_name'] = [f'col{i+1}' for i in range(st.session_state.number_of_columns)]
        st.session_state['df'] = pd.DataFrame(
            np.full((st.session_state.number_of_rows, st.session_state.number_of_columns), None),
            columns=st.session_state.col_name
        )


def create_menu_bar_placeholder():
    menu_items = ['login', 'history', 'save', 'rename', 'Del', 'run', 'new', 'upload']
    menu_widths = [1, 0.7, 0.7, 0.7, 0.7, 5.5, 1, 2]
    menu_cols = st.columns(menu_widths)

    for item, col in zip(menu_items, menu_cols):
        with col:
            st.session_state[item] = st.empty()

def display_content(tabs):
    content_map = {
        '기초통계': display_basic_statistics,
        '회귀분석': display_regression_analysis,
        # '실험계획': display_DOE,
        '품질도구': display_quality_tool,
        # '기타': display_etc,

    }

    display_func = content_map.get(tabs)
    display_func()

@main_with_error_handling
def main():
    # streamlit settings
    st.set_page_config(layout="wide", page_title="Easy.Metric", page_icon=":bar_chart:")

    # custom css apply
    apply_custom_css()

    #draw banner img
    # st.image('utils/image/banner.png', width=950)
    # create columns for banner and text
    col1, col2 = st.columns([0.758, 0.242])

    # 첫 번째 컬럼에 이미지
    with col1:
        st.image('utils/image/banner.png', width=950)

    # 두 번째 컬럼에 텍스트
    with col2:
        st.markdown("""
        <div style="background-color: #f8f9fa; 
                    padding: 12px; 
                    border-radius: 8px; 
                    border-left: 5px solid #cc0000;
                    margin-top: 15px;
                    margin-bottom: 30px;
                    font-size: 0.9em;"> 
            <h4 style="color: #cc0000; margin-top: 0;font-size: 1.3em;">📋 Contact Information</h4>
            <p style="margin-bottom: 3px; font-size: 1em;"><strong>[농심 DT추진팀] 조유민 주임 (7108)</strong></p>
        </div>
        """, unsafe_allow_html=True)

    # initialize session state
    initialize_session_state()

    # create sidebar
    tabs = create_sidebar()

    # create placeholder to menu bar
    create_menu_bar_placeholder()

    # display writable table content
    section_set = table()

    # display to calcurate statistics content
    with section_set:
        display_content(tabs)
    log_visitor()

if __name__ == "__main__":
    main()