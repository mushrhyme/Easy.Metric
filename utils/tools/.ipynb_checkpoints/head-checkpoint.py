import streamlit as st
from st_on_hover_tabs import on_hover_tabs


def apply_custom_css():
    st.markdown("""
    <style>
            div.stButton > button {
                background-color: #000000;  /* 원하는 색상으로 변경 */
                color: white;  /* 텍스트 색상 */
            }
            div.stButton > button:hover {
                background-color: #FFFFFF;  /* 마우스 오버 시 색상 */
                color: red;  /* 마우스 오버 시 텍스트 색상 */
            }
            div.stButton > button:active {
                background-color: #000000;  /* 클릭 시 배경색: 라임 그린 */
                color: white;  /* 클릭 시 텍스트 색상 */
                transform: translateY(2px);  /* 클릭 시 버튼을 약간 아래로 이동 */
            }
        </style>
        """, unsafe_allow_html=True)
    
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
            color: #666666;  /* 선택되지 않은 탭 글자의 기본 색상 (슬레이트 그레이) */
            transition: all 0.5s ease;  /* 부드러운 전환 효과 */
        }
        
        .stTabs [data-baseweb="tab"]:hover {
        color: #CCFF99;  /* 호버 시 글자 색상 */
        }   

        .stTabs [aria-selected="true"] {
            color: #FF4B4B;  /* 선택된 탭의 글자 색상 */
        }
    </style>
    """, unsafe_allow_html=True)
    
    
    st.markdown('<style>' + open('./utils/style.css').read() + '</style>', unsafe_allow_html=True)

    css_string = """
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
    """
    st.markdown(css_string, unsafe_allow_html=True)
    
def create_sidebar():
    with st.sidebar:
        tabs = on_hover_tabs(
            tabName=[
                '기초통계', '회귀분석', 
                # '실험계획', 
                '품질도구', 
                # '기타', 
                ],
            iconName=['dashboard', 'economy', 'archive', 'backpack', 'house'],
            default_choice=0,
            styles={
                'navtab': {
                    'background-color': '#bfbfbf',
                    'color': '#818181',
                    'font-size': '14px',
                    'transition': '.3s',
                    'white-space': 'nowrap',
                    'padding-left': '0px',
                    'text-transform': 'uppercase'
                },
                'tabStyle': {
                    ':hover :hover': {'color': 'red', 'cursor': 'pointer'},
                    'list-style-type': 'none',
                    'margin-bottom': '30px',
                    'padding-left': '30px'
                },
                'iconStyle': {
                    'position': 'fixed',
                    'left': '7.5px',
                    'padding-right': '0px',
                    'text-align': 'right',
                },
            }
        )
    return tabs

