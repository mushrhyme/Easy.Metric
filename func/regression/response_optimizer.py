# Description: 회귀분석을 위한 함수들을 정의한 파일입니다.
from utils.tools.UTIL import *
import streamlit as st
import numpy as np
import pandas as pd
import re 

from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
from itertools import combinations_with_replacement

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_dummy_coding(df, categorical_vars, reference_categories=None):
    """
    범주형 변수를 0, 1로 더미 코딩하고 학습 데이터의 모든 카테고리를 고려
    
    Parameters:
    df: DataFrame - 인코딩할 데이터프레임
    categorical_vars: list of str - 범주형 변수명 리스트
    reference_categories: dict - 각 변수별 모든 카테고리 정보
    
    Returns:
    DataFrame: 더미 변수가 추가된 데이터프레임
    dict: 각 범주형 변수의 기준 범주(reference category) 정보
    """
    df_coded = df.copy()
    ref_cats = {}
    
    for var in categorical_vars:
        # 학습 데이터의 카테고리 정보가 있으면 사용, 없으면 현재 데이터에서 추출
        if reference_categories and var in reference_categories:
            categories = reference_categories[var]
        else:
            categories = sorted(df[var].unique())
            
        ref_cats[var] = categories[0]  # 첫 번째 범주를 기준 범주로 저장
        
        # 더미 변수 생성 (첫 번째 범주 제외)
        for cat in categories[1:]:
            col_name = f"{var}_{cat}"
            df_coded[col_name] = (df[var] == cat).astype(int)
        
        # 원본 변수 제거
        df_coded = df_coded.drop(columns=[var])
    
    return df_coded, ref_cats
    
def create_effect_coding(df, categorical_vars, reference_categories=None):
    """
    범주형 변수를 -1, 0, 1로 효과 코딩하고 마지막 범주의 계수도 계산
    학습 데이터의 모든 카테고리를 고려
    
    Parameters:
    df: DataFrame
    categorical_vars: list of str, 범주형 변수명 리스트
    reference_categories: dict, 각 변수별 모든 카테고리 정보
    
    Returns:
    DataFrame: 효과 코딩된 변수가 추가된 데이터프레임
    dict: 각 범주형 변수의 기준 범주 정보와 계수
    """
    df_coded = df.copy()
    ref_cats = {}
    
    for var in categorical_vars:
        # 학습 데이터의 카테고리 정보가 있으면 사용, 없으면 현재 데이터에서 추출
        if reference_categories and var in reference_categories:
            categories = reference_categories[var]
        else:
            categories = sorted(df[var].unique())
            
        ref_cats[var] = categories[-1]  # 마지막 범주를 기준 범주로 사용
        
        # 기존 효과 코딩 생성 (마지막 범주 제외)
        for cat in categories[:-1]:
            col_name = f"{var}_{cat}"
            df_coded[col_name] = df[var].apply(
                lambda x: 1 if x == cat else (-1 if x == categories[-1] else 0)
            )
        
        # 마지막 범주의 계수는 다른 범주들의 계수 합의 음수값
        last_cat = categories[-1]
        col_name = f"{var}_{last_cat}"
        df_coded[col_name] = df[var].apply(
            lambda x: -1 if x != last_cat else (len(categories) - 1)
        )
        
        # 원본 변수 제거
        df_coded = df_coded.drop(columns=[var])
    
    return df_coded, ref_cats


def wrap_equation(equation, max_length=100):
    """긴 방정식을 여러 줄로 나누는 함수"""
    # 방정식의 항들을 분리
    terms = equation.split(" = ")
    left_side = terms[0]
    right_terms = terms[1].split(" ")
    
    # 오른쪽 항들을 적절한 길이로 나누기
    lines = [left_side + " ="]
    current_line = ""
    
    for term in right_terms:
        if len(current_line + " " + term) > max_length:
            lines.append(current_line)
            current_line = term
        else:
            current_line += " " + term if current_line else term
    
    if current_line:
        lines.append(current_line)
        
    return "<br>".join(lines)
def create_response_optimizer(df, reg_model, X):
    # 예측값 계산 및 표시
    prediction = reg_model.predict(X)[0]
    predictors = st.session_state.predictor
    categorical_vars = [feature for feature in st.session_state.predictor
                        if df[feature].dtype == 'object' or df[feature].dtype == 'category']
    reference_categories = {}
    for col in categorical_vars:
        reference_categories[col] = df[col].unique().tolist()

    # 색상 설정
    colors = {
        'points': '#2E5BFF',
        'marker': '#FF4757',
        'bar_default': '#A8E6CF',
        'bar_selected': '#FF8B94',
        'grid': '#E6E6E6',
        'background': '#FFFFFF'
    }

    # 서브플롯 설정 - 항상 3열 유지
    n_cols = 3
    n_rows = (len(predictors) - 1) // n_cols + 1

    sub_titles = []
    for name in predictors:
        try:
            v = np.round(st.session_state[f"input_{name}"], 2)
        except:
            v = st.session_state[f"select_{name}"]
        sub_titles.append(f"{name}={v}")

    # 실제 필요한 subplot 수에 맞게 specs 생성
    total_plots = len(predictors)
    specs = []
    for i in range(n_rows):
        row = []
        for j in range(n_cols):
            # 현재 위치의 인덱스 계산
            current_index = i * n_cols + j
            # 실제 플롯이 필요한 경우에만 {} 추가, 아니면 None
            if current_index < total_plots:
                row.append({})
            else:
                row.append(None)
        specs.append(row)

    # subplot 생성 시 수정된 specs 사용
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=sub_titles,
        vertical_spacing=0.2,
        horizontal_spacing=0.15,
        specs=specs  # 수정된 specs 사용
    )

    # 현재 입력값 딕셔너리 생성
    current_values = {}
    for feature in predictors:
        if feature in categorical_vars:
            current_values[feature] = st.session_state[f"select_{feature}"]
        else:
            current_values[feature] = st.session_state[f"input_{feature}"]

    for idx, feature in enumerate(predictors):
        if idx < n_cols:  # 첫 번째 행
            row = 1
        else:  # 두 번째 행
            row = 2
        col = idx % n_cols + 1
        # 마지막 플롯이 첫 번째 위치에 그려지는 것을 방지
        if idx == len(predictors) and len(predictors) % n_cols == 1:
                col = 1
        if feature in categorical_vars:
            # 범주형 변수 처리
            current_cat = current_values[feature]
            categories = df[feature].unique()

            # 다른 범주형 변수들의 현재 값을 고정
            fixed_cats = {f: current_values[f] for f in categorical_vars if f != feature}

            y_pred = []
            for cat in categories:
                # 기본 값은 현재 수치형 변수들의 값을 유지
                temp_values = {f: current_values[f] for f in predictors if f not in categorical_vars}

                # 현재 처리 중인 범주형 변수의 값 설정
                temp_values[feature] = cat

                # 다른 범주형 변수들의 고정된 값 추가
                temp_values.update(fixed_cats)

                # 새로운 데이터프레임 생성
                temp_df = pd.DataFrame([temp_values])

                # 인코딩 적용
                if st.session_state.coding_type == "dummy(1, 0)":
                    temp_encoded, _ = create_dummy_coding(temp_df, categorical_vars, reference_categories)
                else:
                    temp_encoded, _ = create_effect_coding(temp_df, categorical_vars, reference_categories)

                # 교호작용 추가
                if st.session_state.interactions:
                    X_inter = pd.DataFrame()
                    for interaction in st.session_state.interactions:
                        former_var, latter_var = interaction.split(":")
                        for former in list(filter(lambda x: former_var in x, temp_encoded.columns)):
                            for latter in list(filter(lambda x: latter_var in x, temp_encoded.columns)):
                                interaction_product = temp_encoded[former] * temp_encoded[latter]
                                interaction = former + ":" + latter
                                X_inter[interaction] = interaction_product
                    temp_encoded = pd.concat([temp_encoded, X_inter], axis=1)

                # X와 동일한 컬럼 구조 만들기
                pred_X = X.copy()
                for col_name in pred_X.columns:
                    if col_name in temp_encoded.columns:
                        pred_X[col_name] = temp_encoded[col_name].values[0]
                # 예측
                y_pred.append(reg_model.predict(pred_X)[0])
            # 바 색상 설정
            colors_bar = [colors['bar_selected'] if cat == current_cat
                          else colors['bar_default'] for cat in categories]

            # 바 차트 추가
            fig.add_trace(
                go.Bar(
                    x=categories,
                    y=y_pred,
                    marker_color=colors_bar,
                    showlegend=False
                ),
                row=row, col=col
            )

        else:
            # 연속형 변수 처리 - Jittered points
            jitter_amount = (df[feature].max() - df[feature].min()) * 0.02
            x_jittered = df[feature] + np.random.normal(0, jitter_amount, len(df))

            fig.add_trace(
                go.Scatter(
                    x=x_jittered,
                    y=df[st.session_state.target],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=colors['points'],
                        opacity=0.5
                    ),
                    showlegend=False
                ),
                row=row, col=col
            )

            fig.add_trace(
                go.Scatter(
                    x=[current_values[feature]],
                    y=[prediction],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=colors['marker'],
                        line=dict(color='white', width=2)
                    ),
                    showlegend=False
                ),
                row=row, col=col
            )

        # 축 설정
        fig.update_xaxes(
            title_text=feature,
            showgrid=True,
            gridwidth=1,
            gridcolor=colors['grid'],
            row=row,
            col=col
        )

        if col == 1:
            fig.update_yaxes(
                title_text=st.session_state.target,
                showgrid=True,
                gridwidth=1,
                gridcolor=colors['grid'],
                row=row,
                col=col
            )
        else:
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor=colors['grid'],
                row=row,
                col=col
            )

    wrapped_equation = wrap_equation(st.session_state.equation)
    # 레이아웃 업데이트
    subplot_height = 250  # 각 subplot의 기본 높이
    equation_height = 150  # 방정식을 위한 추가 높이
    total_height = subplot_height * n_rows + equation_height  # 여백과 방정식 공간 추가
    relative_spacing = 1 / (total_height + equation_height)
    equation_y_position = -(relative_spacing * equation_height)

    fig.update_layout(
        title={
            'text': f'현재 예측값: {prediction:.2f}',
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        showlegend=False,
        height=total_height,
        width=450 * n_cols,
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        font=dict(
            family="Arial, sans-serif",
            size=12
        ),
        margin=dict(
            t=60,
            l=50,
            r=50,
            b=equation_height
        ),
        annotations=[
            dict(
                text=wrapped_equation,
                xref='paper',
                yref='paper',
                x=0.5,
                y=-0.6 if n_rows == 1 else -0.3,
                showarrow=False,
                font=dict(
                    size=12,
                    color='lightgray'
                ),
                align='center'
            )
        ]
    )
    # 그래프 표시
    st.plotly_chart(fig, use_container_width=True)
def create_inverse_optimizer(df, reg_model, target_value, constraints=None):
    from scipy.optimize import minimize
    import numpy as np
    
    predictors = st.session_state.predictor
    categorical_vars = [feature for feature in predictors 
                       if df[feature].dtype == 'object' or df[feature].dtype == 'category']
    numerical_vars = [feature for feature in predictors 
                     if feature not in categorical_vars]
    
    # 수치형 변수의 초기값 설정
    x0 = []
    bounds = []
    for var in numerical_vars:
        x0.append(df[var].mean())
        bounds.append((df[var].min(), df[var].max()))
    
    # 목적함수: (예측값 - 목표값)^2 최소화
    def objective(x):
        # 현재 수치형 변수값으로 데이터프레임 생성
        current_values = {}
        for i, var in enumerate(numerical_vars):
            current_values[var] = x[i]
        
        # 범주형 변수는 현재 선택된 값 사용
        for var in categorical_vars:
            current_values[var] = st.session_state[f"select_{var}"]
            
        # 예측용 데이터프레임 생성
        pred_df = pd.DataFrame([current_values])
        
        # 인코딩 적용
        reference_categories = {col: df[col].unique().tolist() for col in categorical_vars}
        if categorical_vars:
            if st.session_state.coding_type == "dummy(1, 0)":
                X, _ = create_dummy_coding(pred_df, categorical_vars, reference_categories)
            else:
                X, _ = create_effect_coding(pred_df, categorical_vars, reference_categories)
        else:
            X = pred_df.copy()
            
        # 교호작용 추가
        if st.session_state.interactions:
            X_inter = pd.DataFrame()
            for interaction in st.session_state.interactions:
                former_var, latter_var = interaction.split(":")
                for former in list(filter(lambda x: former_var in x, X.columns)):
                    for latter in list(filter(lambda x: latter_var in x, X.columns)):
                        interaction_product = X[former] * X[latter]
                        interaction = former+":"+latter
                        X_inter[interaction] = interaction_product
            X = pd.concat([X, X_inter], axis=1)
            
        X.insert(0, "const", 1)
        
        # 예측값 계산
        pred = reg_model.predict(X)[0]
        return (pred - target_value) ** 2
    
    # 최적화 실행
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
    
    # 최적해를 딕셔너리로 변환
    optimal_values = {}
    for i, var in enumerate(numerical_vars):
        optimal_values[var] = result.x[i]
    
    # 범주형 변수 추가
    for var in categorical_vars:
        optimal_values[var] = st.session_state[f"select_{var}"]
    
    # 최종 예측값 계산
    pred_df = pd.DataFrame([optimal_values])
    reference_categories = {col: df[col].unique().tolist() for col in categorical_vars}
    
    if categorical_vars:
        if st.session_state.coding_type == "dummy(1, 0)":
            X, _ = create_dummy_coding(pred_df, categorical_vars, reference_categories)
        else:
            X, _ = create_effect_coding(pred_df, categorical_vars, reference_categories)
    else:
        X = pred_df.copy()
        
    # 교호작용 추가
    if st.session_state.interactions:
        X_inter = pd.DataFrame()
        for interaction in st.session_state.interactions:
            former_var, latter_var = interaction.split(":")
            for former in list(filter(lambda x: former_var in x, X.columns)):
                for latter in list(filter(lambda x: latter_var in x, X.columns)):
                    interaction_product = X[former] * X[latter]
                    interaction = former+":"+latter
                    X_inter[interaction] = interaction_product
        X = pd.concat([X, X_inter], axis=1)
        
    X.insert(0, "const", 1)
    final_prediction = reg_model.predict(X)[0]
    
    return optimal_values, final_prediction


def response_optimizer_run(df):   
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("**최적 조합 찾기**")
        target_value = st.number_input(
            "목표값",
            min_value=float(0),
            max_value=float(10**8),
            value=float(df[st.session_state.target].mean()),
            step=0.01,
            key=f"input_{st.session_state.target}"
        )
        
        if st.button("찾기"):
            optimal_values, predicted_value = create_inverse_optimizer(
                df, st.session_state.model, target_value
            )
            st.session_state.optimal_values = optimal_values
            st.session_state.predicted_value = predicted_value
            
            # 최적값으로 입력값 업데이트
            for feature, value in optimal_values.items():
                if feature in df.select_dtypes("number").columns:
                    st.session_state[f"input_{feature}"] = value
                else:
                    st.session_state[f"select_{feature}"] = value
            st.write("**최적 조합 결과**")
            st.write(f"예측값: {st.session_state.predicted_value:.4f}")
            st.write(f"목표값 대비 오차: {abs(target_value - st.session_state.predicted_value):.4f}")
 
    with col2:
        numerical_vars = df[st.session_state.predictor].select_dtypes("number").columns
        categorical_vars = [feature for feature in st.session_state.predictor 
                           if feature not in numerical_vars]
        
        input_values = {}
    
        if len(numerical_vars)>0:
            st.write("**수치형 변수**")
            for feature in numerical_vars:
                input_values[feature] = st.number_input(
                    feature,
                    min_value=float(0),
                    max_value=float(10**8),
                    value=float(df[feature].mean()),
                    step=0.01,
                    key=f"input_{feature}"
                )
    
        if len(categorical_vars)>0:
            st.write("**범주형 변수**")
            for feature in categorical_vars:
                unique_values = df[feature].unique().tolist()
                input_values[feature] = st.selectbox(
                    feature,
                    options=unique_values,
                    key=f"select_{feature}"
                )
    
    # 입력값을 데이터프레임으로 변환
    reference_categories = {}
    for col in categorical_vars:
        reference_categories[col] = df[col].unique().tolist()

    input_df = pd.DataFrame([input_values])

    # 범주형 변수 인코딩
    if len(categorical_vars) > 0:
        if st.session_state.coding_type == "dummy(1, 0)":
            X, _ = create_dummy_coding(input_df, categorical_vars, reference_categories)
        else:
            X, _ = create_effect_coding(input_df, categorical_vars, reference_categories)
    else:
        X = input_df.copy()

    # 교호작용 추가
    if st.session_state.interactions:
        X_inter = pd.DataFrame()
        for interaction in st.session_state.interactions:
            former_var, latter_var = interaction.split(":")
            for former in list(filter(lambda x: former_var in x, X.columns)):
                for latter in list(filter(lambda x: latter_var in x, X.columns)):
                    interaction_product = X[former] * X[latter]
                    interaction = former+":"+latter
                    X_inter[interaction] = interaction_product
        X = pd.concat([X, X_inter], axis=1)

    # 상수항 추가
    X.insert(0, "const", [1] * len(X))
    st.session_state.X_for_opt = X

    create_response_optimizer(df, st.session_state.model, st.session_state.X_for_opt)