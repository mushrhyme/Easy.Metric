from t_test import *
from regression import *
from anova import *
from doe import *
from gage_rnr import *
from proc_capt import *
import re

init_ = {
    # "df": pd.DataFrame(),
    "expander_open": False,
    # viz
    "background_color": None,
    "text_color": None,
    "bar_color1": None,
    "bar_color2": None,
    # ttest
    "sample1": None,
    "sample2": None,
    "stats_df": pd.DataFrame(),
    "diff_df": pd.DataFrame(),
    "ttest_df": pd.DataFrame(),
    "confidence_level": 95.0,
    "null_diff": 0.0,
    "alternative": "",
    "equal_var": False,
    # regression
    "target": None,
    "predictor": [],
    "interactions": None,
    "equation": "",
    "coeff_df": pd.DataFrame(),
    "reg_df": pd.DataFrame(),
    "diag_df": pd.DataFrame(),
    "anov_df": pd.DataFrame(),
    # doe
    "design": pd.DataFrame(),
    "num_factors": 2,
    "center_points": 0,
    "replicates": 1,
    "blocks": 1,
    "fraction": "full",
    "whole_plot_factors": 1,
    "whole_plot_replicates": 2,
    "subplot_replicates": 1,
    "use_blocks": False,
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
    # capability analysis
    "k": 0.0,
    "lambd": 0.0,
    "uniq_col": None,
    "process_data_df": pd.DataFrame(),
    "overall_capt_df": pd.DataFrame(),
    "within_capt_df": pd.DataFrame(),
}


for key, default in init_.items():
    if key not in st.session_state:
        st.session_state[key] = default
        
st.set_page_config(layout="wide")
# pio.templates.default = "plotly_white"

# data = {
#     'Part': [i for i in range(1, 11) for _ in range(2)],
#     'Operator': [i for i in range(1, 3) for _ in range(10)],
#     'Measurement':[13.41, 13.4, 17.67, 13.62, 17.87, 16.01, 13.99, 
#                    16.4, 15.18, 17.29, 16.07, 17.05, 14.6, 13.07, 15.99, 14.87, 13.6, 16.07, 13.67, 14.06]
# }

# if 'df' not in st.session_state:
#     np.random.seed(0)
#     data = {f"C{i}": list(np.random.uniform(94, 106, 10)) for i in range(1,6)}
#     st.session_state.df = pd.DataFrame(data).astype(float)

if 'design_generated' not in st.session_state:
    st.session_state.design_generated = False
    
#########################################################################################################
# Main
#########################################################################################################
with st.sidebar:
    main_option = st.sidebar.selectbox("1. 원하는 분석 작업을 선택하세요", 
                                       ["기초통계", "회귀/분산분석", "실험계획법", "품질도구", "시계열"])

    if main_option == "기초통계":
        # st.sidebar.write("**기초통계")
        sub_option = st.sidebar.selectbox("2. 구체적으로 어떤 작업을 원하세요?", ["2-표본 t검정", "두 비율 검정", "2-표본 포아송 비율"])
    elif main_option == "회귀/분산분석":
        sub_option = st.sidebar.selectbox("2. 구체적으로 어떤 작업을 원하세요?", ["회귀분석", "분산분석"])
    elif main_option == "실험계획법":
        sub_option = st.sidebar.selectbox("2. 구체적으로 어떤 작업을 원하세요?", ["요인설계생성"])
    elif main_option == "품질도구":
        sub_option = st.sidebar.selectbox("2. 구체적으로 어떤 작업을 원하세요?", ["Gage R&R", "공정능력분석"])
    elif main_option == "시계열":
        sub_option = st.sidebar.selectbox("2. 구체적으로 어떤 작업을 원하세요?", [""])


with st.expander("데이터 입력"):
    edited_df = st.data_editor(st.session_state.df)
    st.session_state.df = edited_df
    
if main_option=="기초통계":
    if sub_option=="2-표본 t검정":
        col1, col2 = st.columns([0.23, 0.77])
        with col1:
            st.session_state.sample1 = st.selectbox("표본1", st.session_state.df.columns.tolist())
            st.session_state.sample2 = st.selectbox("표본2", st.session_state.df.columns.tolist())
            st.session_state.confidence_level = st.number_input("신뢰수준", value=95.0, step=1.0)
            st.session_state.null_diff = st.number_input("귀무 가설에서의 차이", value=0.0, step=0.1)
            st.session_state.alternative = st.selectbox("대립 가설", ["차이 < 귀무가설에서의 차이", "차이 ≠ 귀무가설에서의 차이", "차이 > 귀무가설에서의 차이"])
            st.session_state.equal_var = st.checkbox("등분산 가정")
        with col2:
            if st.button('Start'):
                run_ttest()
                st.write("**기술 통계량**")
                st.data_editor(st.session_state.stats_df, hide_index=True, key="stats")
                st.write("**차이 추정치**")
                st.data_editor(st.session_state.diff_df, hide_index=True, key="diff")
                if not st.session_state.ttest_df.empty:
                    st.data_editor(st.session_state.ttest_df, hide_index=True, key="ttest")
                    draw_box_plot()


elif main_option=="회귀/분산분석":
    if sub_option=="회귀분석":
        col1, col2 = st.columns([0.23, 0.77])
        with col1:
            st.session_state.target = st.selectbox("반응변수", st.session_state.df.columns.tolist())
            predictor = st.session_state.df.columns.tolist()
            predictor.remove(st.session_state.target)
            st.session_state.predictor = st.multiselect("예측변수", predictor)
            if st.toggle("교호작용"):
                st.session_state.interactions = st.multiselect("교호작용", generate_interactions())
        with col2:
            if st.button('Start'):
                residuals, fitted_values = run_regression()
                run_anova()
                st.write("**회귀방정식**")
                st.markdown(st.session_state.equation)
                st.write("**계수**")
                st.data_editor(st.session_state.coeff_df, key="coeff")
                st.write("**모형 요약**")
                st.data_editor(st.session_state.reg_df, hide_index=True, key="reg")
                st.write("**분산 분석**")
                st.data_editor(st.session_state.anova_df, key="anova")
                st.write("**비정상적 관측치에 대한 적합치 및 진단**")
                st.data_editor(st.session_state.diag_df, hide_index=True, key="diag")
                if not st.session_state.coeff_df.empty:
                    draw_pareto()
                    draw_reg_plot(residuals, fitted_values)

elif main_option=="실험계획법":
    tab1, tab2 = st.tabs(["요인 설계 생성", "요인 설계 분석"])
    with tab1:
        col1, col2 = st.columns([0.23, 0.77])
        with col1:
            option1 = st.radio("설계 유형", ["2-수준 요인", "2-수준 분할구"])
            st.session_state.num_factors = st.selectbox("요인 수", list(range(2, 16)))
            fraction = st.selectbox("설계", [
                f"1/4 부분 설계  ({2**(st.session_state.num_factors-2)}런)", 
                f"1/2 부분 설계  ({2**(st.session_state.num_factors-1)}런)", 
                f"완전 요인 설계  ({2**st.session_state.num_factors}런)"])
            st.session_state.fraction = fraction.split(" ")[0].replace("완전", "full")
            st.session_state.center_points = st.selectbox("블럭당 중앙점 개수", list(range(25)))    
            st.session_state.replicates = st.selectbox("꼭짓점의 반복실험 횟수", list(range(1, 11)))    
            st.session_state.blocks = st.selectbox("블럭 수", [2**i for i in range(5)])   

        with col2:
            if st.button('Start', key="start1"):
                generator = ExperimentalDesignGenerator(
                        num_factors=st.session_state.num_factors,
                        center_points=st.session_state.center_points,
                        replicates=st.session_state.replicates,
                        blocks=st.session_state.blocks,
                    )
                if option1=="2-수준 요인":
                    design = generator.two_level_factorial(
                        st.session_state.fraction
                    )
                elif option1=="2-수준 분할구":
                    design = generator.two_level_split_plot(
                        whole_plot_factors=st.session_state.whole_plot_factors,
                        fraction=st.session_state.fraction,
                        whole_plot_replicates=st.session_state.whole_plot_replicates,
                        subplot_replicates=st.session_state.subplot_replicates,
                        use_blocks=st.session_state.use_blocks
                    )
                # np.random.seed(0)
                # design["반응1"] = pd.DataFrame(np.round(np.random.uniform(0, 1, 32), 2))
                st.session_state.df = design
                # st.experimental_rerun()
    with tab2:
        col1, col2 = st.columns([0.23, 0.77])
        with col1:
            st.session_state.target = st.selectbox("반응변수", st.session_state.df.columns.tolist())
            st.session_state.predictor = [col for col in st.session_state.df.columns if re.match(r'^[A-Z]+$', col)]
            st.session_state.max_degree = st.selectbox("모형에 포함되는 항의 최대 차수", list(range(1, st.session_state.num_factors)))
            interaction_terms = list(filter(lambda x: len(x.split(":"))<=st.session_state.max_degree and len(list(set(x.split(":")))) ==len(x.split(":")), generate_interactions()))
            st.session_state.interactions = st.multiselect("항", interaction_terms, interaction_terms)
        with col2:
            if st.button('Start', key="start2"):
                d = st.session_state.df[st.session_state.predictor]
                residuals, fitted_values = run_regression()
                # run_anova()
                st.write("**회귀방정식**")
                st.markdown(st.session_state.equation)
                st.write("**계수**")
                st.data_editor(st.session_state.coeff_df, key="coeff")
                # st.write("**모형 요약**")
                # st.data_editor(st.session_state.reg_df, hide_index=True, key="reg")
                
elif main_option=="품질도구":
    if sub_option=="Gage R&R":
        col1, col2 = st.columns([0.23, 0.77])
        with col1:
            st.session_state.part = st.selectbox("시료 번호", st.session_state.df.columns.tolist())
            st.session_state.operator = st.selectbox("측정 시스템", st.session_state.df.columns.tolist())
            st.session_state.measurement = st.selectbox("측정 데이터", st.session_state.df.columns.tolist())
            st.session_state.studyvar = st.number_input("연구 변동(표준 편차 수)", value=6)
            if st.toggle("공정 공차"):
                option1 = st.radio("", ["최소한 하나의 규격 한계 입력", "규격 상한-규격 하한"], label_visibility="collapsed")
                if option1 == "최소한 하나의 규격 한계 입력":
                    st.session_state.lsl = st.number_input("규격 하한", value=min(st.session_state.df[st.session_state.measurement]))
                    st.session_state.usl = st.number_input("규격 상한", value=max(st.session_state.df[st.session_state.measurement]))
                    st.session_state.tolerance = st.session_state.usl-st.session_state.lsl
                elif option1 == "규격 상한-규격 하한":
                    st.session_state.tolerance = st.number_input("", value=st.session_state.usl-st.session_state.lsl, label_visibility="collapsed")

            # st.session_state.bar_color1 = st.color_picker("첫 번째 막대 색상", "#1F77B4")
        with col2:
            if st.button('Start'):
                run_gage_rnr()
                st.write("**교호작용이 있는 이원분산분석**")
                st.data_editor(st.session_state.anova_w_inter_df, key="anova_w")
                st.write("**교호작용이 없는 이원분산분석**")
                st.data_editor(st.session_state.anova_wo_inter_df, key="anova_wo")
                st.write("**Gage R&R**")
                st.data_editor(st.session_state.var_comp_df, key="var_comp")
                if st.session_state.tolerance is not None:
                    st.write(f"공정 공차: {str(st.session_state.tolerance)}")
                st.data_editor(st.session_state.std_comp_df, key="std_comp")
                st.write(f"구별 범주의 수: {str(st.session_state.ndc)}")
                if not st.session_state.std_comp_df.empty:
                    viz_gage_rnr()
                    
    elif sub_option=="공정능력분석":
        col1, col2 = st.columns([0.23, 0.77])
        with col1:
            st.session_state.uniq_col = st.selectbox("단일 열", st.session_state.df.columns.tolist())
            st.session_state.subgroup_size = st.number_input("부분군 크기", value=1)
            st.session_state.lsl = st.number_input("규격 하한", value=None)
            st.session_state.usl = st.number_input("규격 상한", value=None)
            
            with st.popover("옵션"):
                st.session_state.target = st.number_input("목표값(표에 Cpm 추가X)", value=None)
                st.session_state.k = st.number_input("공정 능력 통계에 K x σ 공차 사용")
                st.radio("표시", ["PPM", "백분율"])
                st.radio("", ["공정 능력 통계량(Cp, PpXL)", "벤치마크 Z(σ수준 X E)"], label_visibility="collapsed")
                
                option1 = st.radio("변환", ["변환 없음", "Box-Cox 누승 변환", "Johnsom 변환(전체 산포 분석만 X J)"])
                if option1 == "Box-Cox 누승 변환":
                    with st.container(border=True):
                        option2 = st.radio("", ["최적 λ 사용", "λ = 0", "λ = 0.5", "기타(-5와 5 사이의 값 입력 X H)"], label_visibility="collapsed")
                        if option2 =="기타(-5와 5 사이의 값 입력 X H)":
                            st.session_state.lambd = st.number_input("", min_value=-5, max_value=5, label_visibility="collapsed")
                elif option1 == "Johnsom 변환(전체 산포 분석만 X J)":
                    st.session_state.p_value = st.number_input("최량 적합을 선택하기 위한 P-값", value=0.10)
                
                if st.checkbox("신뢰구간 포함"):
                    with st.container(border=True):
                        st.session_state.confidence_level = st.number_input("신뢰 수준", value=95.0, step=1.0)
                        st.session_state.confidence_interval = st.selectbox("신뢰 구간", ["단측", "양측"])
                    
                
        with col2:
            if st.button('Start'):
                run_proc_capt()
                col1, col2, col3 = st.columns([2,6,2])
                with col1:
                    st.markdown("**공정데이터**")
                    st.data_editor(st.session_state.process_data_df, key="process_data")
                with col2:
                    st.write(f'**{st.session_state.uniq_col}의 공정 능력 보고서**')
                    draw_hist()
                    
                with col3:
                    st.write("**전체 공정 능력**")
                    st.data_editor(st.session_state.overall_capt_df, key="overall_capt")
                    st.write("**잠재적(군내) 공정 능력**")
                    st.data_editor(st.session_state.within_capt_df, key="within_capt")
                # if not st.session_state.process_data_df.empty:
                #     draw_hist()
                st.write("**성능**")
                st.data_editor(st.session_state.perf_df, key="perf")
                
                
                