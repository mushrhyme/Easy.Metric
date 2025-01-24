import streamlit as st
from st_on_hover_tabs import on_hover_tabs


def apply_custom_css():
    # 기존 버튼 스타일
    st.markdown("""
    <style>
        div.stButton > button {
            background-color: #000000;
            color: white;
        }
        div.stButton > button:hover {
            background-color: #FFFFFF;
            color: red;
        }
        div.stButton > button:active {
            background-color: #000000;
            color: white;
            transform: translateY(2px);
        }
    </style>
    """, unsafe_allow_html=True)

    # 기존 앱 배경색 및 탭 스타일
    st.markdown("""
    <style>
        .stApp {
            background-color: #d8d8d8;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 30px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 20px;
            white-space: pre-wrap;
            border-radius: 4px 4px 1px 0px;
            gap: 2px;
            padding-top: 10px;
            padding-left: 0px;
            padding-bottom: 20px;
            color: #666666;
            transition: all 0.5s ease;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: #CCFF99;
        }   

        .stTabs [aria-selected="true"] {
            color: #FF4B4B;
        }

        /* 새로운 사이드바 스타일 */
        .stSidebar {
            background-color: #bfbfbf;
            min-width: 200px;
        }

        .nav-tabs {
            width: 100%;
            padding: 0;
            margin: 0;
        }

        .nav-tabs .nav-item {
            width: 100%;
            opacity: 1 !important;
            transition: background-color 0.3s ease;
        }

        .nav-tabs .nav-item:hover {
            background-color: rgba(255, 255, 255, 0.1);
        }

        .nav-tabs .nav-item.active {
            background-color: #FF4B4B;
            color: white !important;
        }

        .nav-tabs .nav-item i {
            opacity: 1 !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # 기존 style.css 파일 로드
    st.markdown('<style>' + open('./utils/style.css').read() + '</style>', unsafe_allow_html=True)

    # 기존 컨테이너 패딩 스타일
    st.markdown("""
    <style>
        .st-emotion-cache-1jicfl2 {
            width: 100%;
            padding-top: 0rem !important;
            padding-left: 4rem !important;
            padding-right: 3rem !important;
            padding-bottom: 10rem !important;
            min-width: auto;
            max-width: initial;

        }
    </style>
    """, unsafe_allow_html=True)


def create_sidebar():
    with st.sidebar:
        tabs = on_hover_tabs(
            tabName=['기초통계', '회귀분석', '품질도구'],
            iconName=['dashboard', 'economy', 'archive'],
            default_choice=0,
            styles={
                'navtab': {
                    'background-color': '#bfbfbf',
                    'color': '#818181',
                    'font-size': '14px',
                    'transition': '.3s',
                    'white-space': 'nowrap',
                    'padding-left': '0px',
                    'text-transform': 'uppercase',
                    'width': '100%',
                    'display': 'block'
                },
                'tabStyle': {
                    'list-style-type': 'none',
                    'margin-bottom': '30px',
                    'padding-left': '30px',
                    'cursor': 'pointer',
                    'display': 'block',
                    'width': '100%',
                },
                'iconStyle': {
                    'position': 'fixed',
                    'left': '7.5px',
                    'padding-right': '0px',
                    'text-align': 'right',
                },
                'containerStyle': {
                    'display': 'block',
                    'width': '100%',
                    'padding': '10px 0',
                    'background-color': '#bfbfbf'
                }
            }
        )
    return tabs