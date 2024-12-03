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


def log_transform(data, base=None):
    data = np.array(data)
    
    if np.any(data <= 0):
        raise ValueError("로그 변환을 하려면 모든 값이 양수이어야 합니다.")
    
    if base is None:
        base = np.e
        transform_func = np.log
    elif base == 2:
        transform_func = np.log2
    elif base == 10:
        transform_func = np.log10
    else:
        transform_func = lambda x: np.log(x) / np.log(base)

    return transform_func(data)


def box_cox_transform(data, lambda_param):
    gm = None
    if lambda_param is None:
        data_transformed, lambda_param = stats.boxcox(data)
        if lambda_param <= -5:
            lambda_param = -5
        elif lambda_param >= 5:
            lambda_param = 5
    if lambda_param == 0:
        data_transformed = log_transform(data)
    elif lambda_param == 0.5:
        data_transformed = np.sqrt(data)
    else:
        gm = np.round(np.exp(np.mean(log_transform(data))), 3)
        data_transformed = (data**lambda_param-1)/(lambda_param*gm**(lambda_param-1))
    return data_transformed, lambda_param, gm


def generate_interactions(df):
    categorical_vars = df.select_dtypes("object").columns.tolist()
    interaction_terms = []
    # 2차 교호작용만 생성 (degree = 2로 고정)
    for combo in combinations_with_replacement(st.session_state.predictor, 2):
        # 범주형 변수가 자기 자신과 교호작용하는 경우는 제외
        if combo[0] == combo[1] and combo[0] in categorical_vars:
            continue
        term = ":".join(combo)
        interaction_terms.append(term)
    return interaction_terms

    
def forward_selection(X, y):
    always_included = set(st.session_state.always_included or [])
    included = list(always_included.copy())
    candidates = [col for col in X.columns if col not in always_included]
    step = 1
    
    with st.expander("전진 선택 과정 상세 보기", expanded=False):
        while True:
            changed, included, candidates = forward_selection_step(X, y, included, candidates, step)
            step += 1
            
            if not changed:
                st.divider()
                st.markdown("**변수 선택 완료**")
                st.markdown(f"최종 사용 변수: {included}")
                break
            st.divider()
    
    return included

    
def backward_elimination(X, y):
    always_included = set(st.session_state.always_included or [])
    included = list(X.columns)
    step = 1
    
    with st.expander("후진 제거 과정 상세 보기", expanded=False):
        while True:
            changed, removed_var, included = backward_elimination_step(X, y, included, always_included, step)
            step += 1
            
            if not changed:
                st.divider()
                st.markdown("**변수 선택 완료**")
                st.markdown(f"최종 사용 변수: {included}")
                break
            st.divider()
    return included


def stepwise_selection(X, y):
    always_included = set(st.session_state.always_included or [])
    initial_included = set(st.session_state.initial_included or [])
    included = list(set(always_included) | set(initial_included))
    candidates = list(set(X.columns) - set(included) - set(always_included))
    
    removed = []
    step = 1
    with st.expander("단계별 선택 과정 상세 보기", expanded=False):
        while True:
            changed = False
            # 후진 제거
            if len(included) > 0:
                changed, removed_var, included = backward_elimination_step(X, y, included, always_included, step)
                step += 1
                if changed:
                    removed.append(removed_var)
            
            # 제거된 변수가 없을 경우 전진 선택
            if not changed:
                changed, included, candidates = forward_selection_step(X, y, included, candidates, step)
                step += 1
            
            # 변동이 없을 경우 종료
            if not changed:
                st.divider()
                st.markdown("**변수 선택 완료**")
                st.markdown(f"최종 사용 변수: {included}")
                break
            st.divider()
    
    return included

    
def backward_elimination_step(X, y, included, always_included, step):
    if len(included) <= len(always_included):
        return False, None, included
    
    X_current = sm.add_constant(X[included])
    model = sm.OLS(y, X_current).fit()
    
    display_step_info(step, model)
    
    pvalues = model.pvalues.drop('const')
    pvalues = pvalues.drop(always_included)
    
    if pvalues.empty:
        return False, None, included
    
    worst_pval = pvalues.max()
    worst_feature = pvalues.idxmax()
    
    if worst_pval > st.session_state.alpha_out:
        included.remove(worst_feature)
        display_removal_info(worst_feature, worst_pval)
        return True, worst_feature, included
    else:
        display_not_removal_info(worst_feature, worst_pval)    
    return False, None, included


def forward_selection_step(X, y, included, candidates, step):
    if not candidates:
        return False, included, candidates
    
    new_pval = pd.Series(index=candidates)
    models = {}  # 각 후보 변수에 대한 모델을 저장할 딕셔너리
    for new_column in candidates:
        X_new = sm.add_constant(X[included + [new_column]])
        model = sm.OLS(y, X_new).fit()
        new_pval[new_column] = model.pvalues[new_column]
        models[new_column] = model  # 각 모델을 저장
    
    if new_pval.empty:
        return False, included, candidates

    best_pval = new_pval.min()
    best_feature = new_pval.idxmin()
    best_model = models[best_feature]  # 가장 좋은 모델 선택
    
    if best_pval < st.session_state.alpha_in:
        included.append(best_feature)
        candidates.remove(best_feature)

        st.write(f"현재 포함된 변수: {included}")
        display_step_info(step, best_model)
        display_addition_info(best_feature, best_pval)
        return True, included, candidates
    else:
        st.write(f"현재 포함된 변수: {included}")
        display_step_info(step, best_model)
        display_not_addition_info(best_feature, best_pval)
    return False, included, candidates


def forward_selection_step(X, y, included, candidates, step):
    if not candidates:
        return False, included, candidates

    # 범주형 변수들의 더미/효과 코딩 변수들을 그룹화
    var_groups = {}
    for var in candidates:
        base_var = var.split('_')[0]  # 기본 변수명 추출
        if base_var not in var_groups:
            var_groups[base_var] = []
        var_groups[base_var].append(var)

    # 각 변수(그룹)에 대한 p-value 계산
    new_pval = pd.Series(index=var_groups.keys())
    models = {}

    for base_var, group_vars in var_groups.items():
        # 이미 포함된 변수들과 현재 검토 중인 변수(그룹) 결합
        current_vars = included + group_vars
        X_new = sm.add_constant(X[current_vars])
        model = sm.OLS(y, X_new).fit()

        # 그룹의 전체 유의성 검정 (Type II ANOVA)
        if len(group_vars) > 1:  # 범주형 변수
            reduced_vars = [var for var in current_vars if not var.startswith(base_var)]
            X_reduced = sm.add_constant(X[reduced_vars])
            reduced_model = sm.OLS(y, X_reduced).fit()

            # F-test로 그룹 전체의 유의성 검정
            f_stat = ((reduced_model.ssr - model.ssr) / (len(group_vars))) / model.mse_resid
            new_pval[base_var] = 1 - stats.f.cdf(f_stat, len(group_vars), model.df_resid)
        else:  # 연속형 변수
            new_pval[base_var] = model.pvalues[group_vars[0]]

        models[base_var] = model

    if new_pval.empty:
        return False, included, candidates

    best_pval = new_pval.min()
    best_feature = new_pval.idxmin()
    best_model = models[best_feature]

    if best_pval < st.session_state.alpha_in:
        # 선택된 변수(그룹)의 모든 더미/효과 코딩 변수 추가
        included.extend(var_groups[best_feature])
        # 해당 변수(그룹)의 모든 더미/효과 코딩 변수 제거
        for var in var_groups[best_feature]:
            candidates.remove(var)

        st.write(f"현재 포함된 변수: {included}")
        display_step_info(step, best_model)
        display_addition_info(best_feature, best_pval)
        return True, included, candidates
    else:
        st.write(f"현재 포함된 변수: {included}")
        display_step_info(step, best_model)
        display_not_addition_info(best_feature, best_pval)
    return False, included, candidates


def backward_elimination_step(X, y, included, always_included, step):
    if len(included) <= len(always_included):
        return False, None, included

    X_current = sm.add_constant(X[included])
    model = sm.OLS(y, X_current).fit()

    display_step_info(step, model)

    # 범주형 변수들의 더미/효과 코딩 변수들을 그룹화
    var_groups = {}
    for var in included:
        if var not in always_included:
            base_var = var.split('_')[0]
            if base_var not in var_groups:
                var_groups[base_var] = []
            var_groups[base_var].append(var)

    # 각 변수(그룹)에 대한 p-value 계산
    pvalues = pd.Series(index=var_groups.keys())

    for base_var, group_vars in var_groups.items():
        if len(group_vars) > 1:  # 범주형 변수
            # 현재 변수(그룹)을 제외한 모델 적합
            reduced_vars = [var for var in included if not var.startswith(base_var)]
            X_reduced = sm.add_constant(X[reduced_vars])
            reduced_model = sm.OLS(y, X_reduced).fit()

            # F-test로 그룹 전체의 유의성 검정
            f_stat = ((reduced_model.ssr - model.ssr) / (len(group_vars))) / model.mse_resid
            pvalues[base_var] = 1 - stats.f.cdf(f_stat, len(group_vars), model.df_resid)
        else:  # 연속형 변수
            pvalues[base_var] = model.pvalues[group_vars[0]]

    if pvalues.empty:
        return False, None, included

    worst_pval = pvalues.max()
    worst_feature = pvalues.idxmax()

    if worst_pval > st.session_state.alpha_out:
        # 해당 변수(그룹)의 모든 더미/효과 코딩 변수 제거
        for var in var_groups[worst_feature]:
            included.remove(var)
        display_removal_info(worst_feature, worst_pval)
        return True, worst_feature, included
    else:
        display_not_removal_info(worst_feature, worst_pval)
    return False, None, included


def stepwise_selection(X, y):
    always_included = set(st.session_state.always_included or [])
    initial_included = set(st.session_state.initial_included or [])

    # always_included와 initial_included의 모든 더미/효과 코딩 변수 포함
    all_included_vars = set()
    for var in always_included | initial_included:
        all_included_vars.update([col for col in X.columns if col.startswith(var)])

    included = list(all_included_vars)
    candidates = [col for col in X.columns if col not in all_included_vars]

    removed = []
    step = 1
    with st.expander("단계별 선택 과정 상세 보기", expanded=False):
        while True:
            changed = False
            # 후진 제거
            if len(included) > 0:
                changed, removed_var, included = backward_elimination_step(X, y, included, always_included, step)
                step += 1
                if changed:
                    removed.append(removed_var)

            # 제거된 변수가 없을 경우 전진 선택
            if not changed:
                changed, included, candidates = forward_selection_step(X, y, included, candidates, step)
                step += 1

            # 변동이 없을 경우 종료
            if not changed:
                st.divider()
                st.markdown("**변수 선택 완료**")
                st.markdown(f"최종 사용 변수: {included}")
                break
            st.divider()

    return included
    
def display_step_info(step, model):
    st.write(f"**단계 {step}**")
    st.write("현재 모델의 p-값:")
    st.write(pd.DataFrame({"계수": model.params, "p-값": model.pvalues}))


def display_removal_info(feature, p_value):
    st.markdown(f":green-background[{feature}]의 p-값 = :blue-background[{p_value:.4f}] > :red-background[{st.session_state.alpha_out}] = α")
    st.markdown(f"{feature}을(를) 모형에서 제거합니다.")


def display_not_removal_info(feature, p_value):
    st.markdown(f":green-background[{feature}]의 p-값 = :blue-background[{p_value:.4f}] < :red-background[{st.session_state.alpha_out}] = α")
    st.markdown(f"{feature}을(를) 모형에서 제거하지 않습니다.")


def display_addition_info(feature, p_value):
    st.markdown(f":green-background[{feature}]의 p-값 = :blue-background[{p_value:.4f}] < :red-background[{st.session_state.alpha_in}] = α")
    st.markdown(f"{feature}을(를) 모형에 추가합니다.")


def display_not_addition_info(feature, p_value):
    st.markdown(f":green-background[{feature}]의 p-값 = :blue-background[{p_value:.4f}] > :red-background[{st.session_state.alpha_in}] = α")
    st.markdown(f"{feature}을(를) 모형에 추가하지 않습니다.")


def check_multicollinearity(X, threshold=10):
    # X = X.drop('const', axis=1)  # VIF 계산 시 상수항 제외
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif[vif["VIF"] > threshold]["Variable"].tolist()


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


def run_regression(df):
    X = df[st.session_state.predictor]
    # 범주형 변수 처리
    categorical_vars = X.select_dtypes("object")

    if categorical_vars.shape[1] > 0:
        st.write("**범주형 예측변수 코드화**")
        if st.session_state.coding_type == "dummy(1, 0)":
            X, reference_cats = create_dummy_coding(X, categorical_vars)
            st.write("(1, 0) - 기준 범주:")
            for var, ref in reference_cats.items():
                st.write(f"- {var}: {ref}")
        else:  # effect coding
            X, reference_cats = create_effect_coding(X, categorical_vars)
            st.write("(1, 0, -1) - 기준 범주:")
            for var, ref in reference_cats.items():
                st.write(f"- {var}: {ref}")
    else:
        X = df.copy()[st.session_state.predictor]
    predictors = X.columns
    y = df[st.session_state.target]

    # Box-Cox 변환
    if st.session_state.lambd != "X":
        y, st.session_state.lambd, gm = box_cox_transform(y, st.session_state.lambd)

    # 변수 선택
    if st.session_state.var_selection == "단계적 회귀":
        predictors = stepwise_selection(X, y)
    if st.session_state.var_selection == "전진 선택":
        predictors = forward_selection(X, y)
    if st.session_state.var_selection == "후진 제거":
        predictors = backward_elimination(X, y)

    # 항이 모두 제거된 경우
    if len(predictors) == 0:
        st.write("단계적 절차가 모형에서 모든 항을 제거했습니다. 어느 항도 모형에 들어갈 수 없습니다.")
        return

    # 회귀 분석
    st.session_state.X = X
    st.session_state.X.insert(0, "const", [1] * len(X))

    # 교호작용 추가
    predictors = predictors.tolist()

    X_inter = pd.DataFrame()
    if st.session_state.interactions:
        for interaction in st.session_state.interactions:
            former_var, latter_var = interaction.split(":")
            for former in list(filter(lambda x: former_var in x, X.columns)):
                for latter in list(filter(lambda x: latter_var in x, X.columns)):
                    interaction_product = X[former] * X[latter]
                    interaction = former + ":" + latter
                    X_inter[interaction] = interaction_product
                    predictors.append(interaction)
    st.session_state.X = pd.concat([st.session_state.X, X_inter], axis=1)
    n, p = st.session_state.X.shape
    reg_model = sm.OLS(y, st.session_state.X).fit()
    st.session_state.model = reg_model

    # 회귀방정식
    if st.session_state.lambd == "X":
        expression_y = st.session_state.target
    elif st.session_state.lambd == 0.0:
        expression_y = f"ln({st.session_state.target})"
    elif st.session_state.lambd == 0.5:
        expression_y = f"{st.session_state.target}^0.5"
    else:
        expression_y = f"({st.session_state.target}^λ-1)/(λx{gm}^(λ-1))"

    equation = f"{expression_y} = {reg_model.params[0]:.4f}"
    for i in range(1, len(reg_model.params)):
        coef = reg_model.params[i]
        sign = "+" if coef >= 0 else "-"
        equation += f" {sign} {abs(coef):.3f} {predictors[i - 1]}"
    st.session_state.equation = equation

    # 계수
    st.session_state.coeff_df = pd.DataFrame({
        '계수': np.round(reg_model.params, 3),
        'SE 계수': np.around(reg_model.bse, 3),
        'T-값': np.round(reg_model.tvalues, 2),
        'P-값': np.around(reg_model.pvalues, 3),
        'VIF': [""] + [np.around(variance_inflation_factor(st.session_state.X.values, i), 2) for i in range(1, p)]
    })
    st.session_state.coeff_df.index = ["상수"] + predictors

    # 모형 요약
    st.session_state.reg_df = pd.DataFrame({
        'S': [np.sqrt(reg_model.mse_resid)],
        'R-제곱': f"{np.round(reg_model.rsquared * 100, 2)}%",
        'R-제곱(수정)': f"{np.round(max(0, reg_model.rsquared_adj) * 100, 2)}%",
        'R-제곱(예측)': f"{max(0, 1 - (1 - reg_model.rsquared) * (n - 1) / n - p) * 100}%"
    })

    st.session_state.model = reg_model
    st.session_state.diag_df = pd.DataFrame({
        "관측": np.arange(1, n + 1),
        st.session_state.target: y,
        "적합치": reg_model.fittedvalues,
        "잔차": reg_model.resid,
        "표준화 잔차": reg_model.resid_pearson,
        "레버리지 (hi)": reg_model.get_influence().hat_matrix_diag,
        "Cook's D": reg_model.get_influence().cooks_distance[0]
    }).reset_index(drop=True)


def safe_column_name(name):
    """
    특수문자와 공백이 포함된 변수명을 안전한 변수명으로 변환
    """
    # 공백과 특수문자를 언더스코어로 변환
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)

    # 숫자로 시작하는 경우 앞에 'x' 추가
    if safe_name[0].isdigit():
        safe_name = 'x' + safe_name

    return safe_name


def create_name_mapping(names):
    """
    원본 변수명과 안전한 변수명 간의 매핑 생성
    """
    return {name: safe_column_name(name) for name in names}


def calculate_sequential_ss(df, target, terms, name_mapping):
    """
    순차적으로 SS를 계산하는 함수
    """
    ss_values = []
    df_values = []  # 각 항의 자유도를 저장할 리스트
    previous_model = None

    # 안전한 변수명으로 데이터프레임 복사
    safe_df = df.copy()
    safe_df.rename(columns=name_mapping, inplace=True)
    safe_target = name_mapping[target]
    safe_terms = [name_mapping[term] for term in terms]

    for i in range(len(safe_terms)):
        current_formula = f"{safe_target} ~ " + " + ".join(safe_terms[:i + 1])
        current_model = ols(current_formula, data=safe_df).fit()

        # 자유도 계산
        if previous_model:
            df_term = current_model.df_model - previous_model.df_model
        else:
            df_term = current_model.df_model
        df_values.append(df_term)

        # SS 계산
        if previous_model is None:
            ss = np.sum((current_model.fittedvalues - safe_df[safe_target].mean()) ** 2)
        else:
            ss = np.sum((current_model.fittedvalues - previous_model.fittedvalues) ** 2)

        ss_values.append(ss)
        previous_model = current_model

    return ss_values, df_values, current_model


def calculate_adjusted_ss(df, target, terms, name_mapping):
    """
    Type III SS (Adjusted SS) 계산 - statsmodels 활용
    """
    # 안전한 변수명으로 데이터프레임 복사
    safe_df = df.copy()
    safe_df.rename(columns=name_mapping, inplace=True)
    safe_target = name_mapping[target]
    safe_terms = [name_mapping[term] for term in terms]

    # 전체 모델식 생성
    formula = f"{safe_target} ~ " + " + ".join(safe_terms)

    # Type III SS 계산
    model = ols(formula, data=safe_df).fit()
    type3 = sm.stats.anova_lm(model, typ=3)

    # 결과를 원래 변수명으로 변환
    reverse_mapping = {v: k for k, v in name_mapping.items()}
    return {reverse_mapping.get(k, k): v for k, v in type3['sum_sq'].to_dict().items()}


def run_anova(df):
    # 범주형 변수 처리
    numerical_vars = df[st.session_state.predictor].select_dtypes("number").columns
    categorical_vars = df[st.session_state.predictor].select_dtypes("object").columns

    # 변수명 매핑 생성
    all_vars = list(st.session_state.predictor) + [st.session_state.target]
    name_mapping = create_name_mapping(all_vars)

    # 데이터프레임 복사 및 안전한 변수명으로 변환
    safe_df = df.copy()

    for var in numerical_vars:
        safe_var = name_mapping[var]
        safe_df[safe_var + "_sq"] = df[var] ** 2

    if len(categorical_vars) > 0:
        for var in categorical_vars:
            safe_var = name_mapping[var]
            safe_df[safe_var] = df[var].astype('category')

    # 수식 생성
    safe_interactions = [name_mapping[i.split(":")[0]] + "_sq" if i.split(":")[0] == i.split(":")[1]
                         else i.replace(":", "_") for i in st.session_state.interactions]
    safe_terms = [name_mapping[term] for term in st.session_state.predictor] + safe_interactions

    ss_values, df_values, final_model = calculate_sequential_ss(safe_df, st.session_state.target,
                                                                st.session_state.predictor + st.session_state.interactions,
                                                                name_mapping)
    adj_ss = calculate_adjusted_ss(safe_df, st.session_state.target,
                                   st.session_state.predictor + st.session_state.interactions,
                                   name_mapping)

    # ANOVA 테이블 생성 (원래 변수명 사용)
    original_terms = st.session_state.predictor + st.session_state.interactions
    results_df = pd.DataFrame(index=['회귀'] + original_terms + ['오차', '총계'])

    # 총 제곱합과 자유도 계산
    SST = np.sum((df[st.session_state.target] - df[st.session_state.target].mean()) ** 2)
    SSE = np.sum((df[st.session_state.target] - final_model.fittedvalues) ** 2)
    SSR = SST - SSE

    DF_total = len(df) - 1
    DF_reg = final_model.df_model
    DF_error = final_model.df_resid

    # Sequential SS 할당
    results_df.loc['회귀', ['Seq SS', 'DF']] = [SSR, DF_reg]
    for term, ss, df_term in zip(original_terms, ss_values, df_values):
        results_df.loc[term, 'Seq SS'] = ss
        results_df.loc[term, 'DF'] = df_term

    # Adjusted SS 할당
    results_df.loc['회귀', 'Adj SS'] = SSR
    for term in original_terms:
        if term in adj_ss:
            results_df.loc[term, 'Adj SS'] = adj_ss[term]

    # 오차, 총계 설정
    results_df.loc['오차', ['Seq SS', 'DF', 'Adj SS']] = [SSE, DF_error, SSE]
    results_df.loc['총계', ['Seq SS', 'DF', 'Adj SS']] = [SST, DF_total, SST]

    # MS 계산
    results_df['MS'] = results_df['Adj SS'] / results_df['DF']

    # F값 계산
    ms_error = results_df.loc['오차', 'MS']
    results_df['F-값'] = results_df['MS'] / ms_error
    results_df.loc[['오차', '총계'], 'F-값'] = np.nan

    # P값 계산
    for idx in results_df.index:
        if idx not in ['오차', '총계']:
            df1 = results_df.loc[idx, 'DF']
            df2 = results_df.loc['오차', 'DF']
            f_value = results_df.loc[idx, 'F-값']
            results_df.loc[idx, 'P-값'] = 1 - stats.f.cdf(f_value, df1, df2)

    results_df.loc[['오차', '총계'], 'P-값'] = np.nan

    # 제곱항 표시 변경
    results_df.index = [x.split("_sq")[0] + ":" + x.split("_sq")[0] if "_sq" in x else x
                        for x in results_df.index]

    st.session_state.anova_df = results_df[['DF', 'Seq SS', 'Adj SS', 'MS', 'F-값', 'P-값']].round(3).fillna("")


def draw_pareto():
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=st.session_state.coeff_df.index,
        x=st.session_state.coeff_df,
        orientation='h',
        marker=dict(color='skyblue')
    ))

    fig.add_trace(go.Scatter(
        y=st.session_state.coeff_df.index,
        x=[2] * len(st.session_state.coeff_df),
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='유의수준 (T값 = 2)'
    ))

    fig.update_layout(
        title='표준화된 효과의 Pareto 차트',
        xaxis_title='표준화된 효과',
        yaxis_title='항',
        yaxis=dict(autorange='reversed'),
        showlegend=False,
    )
    st.plotly_chart(fig)

    
def draw_reg_plot(residuals, fitted_values):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('정규 확률도', '잔차 대 적합치', '잔차의 히스토그램', '잔차 대 순서')
    )
    
    # 정규 확률도 (QQ plot)
    qq = sm.ProbPlot(residuals)
    qq_theoretical, qq_sample = stats.probplot(residuals, dist="norm")[0]
    percentiles = np.sort(stats.norm.cdf(qq_sample) * 100)
    percentiles = stats.norm.cdf(qq_theoretical) * 100

    # 백분율 1%에서 99% 사이의 값을 생성
    percentiles_1_to_99 = np.arange(1, 99)

    # 백분율 1%에서 99%에 해당하는 이론적 Quantiles 계산
    theoretical_quantiles = stats.norm.ppf(percentiles_1_to_99 / 100)

    # 직선을 그리기 위한 시작점과 끝점 설정
    line_start = min(theoretical_quantiles)
    line_end = max(theoretical_quantiles)
    line_y_start = min(percentiles_1_to_99)
    line_y_end = max(percentiles_1_to_99)

    # 정규 확률도 (QQ plot)
    fig.add_trace(
        go.Scatter(x=qq_sample, y=percentiles, mode='markers', name='잔차'),
        row=1, col=1
    )

    # 정규성을 가정한 직선 추가
    fig.add_trace(
        go.Scatter(x=[line_start, line_end], y=[line_y_start, line_y_end],
                   mode='lines', name='정규선', line=dict(color='red')),
        row=1, col=1
    )

    # 잔차 대 적합치
    fig.add_trace(
        go.Scatter(x=fitted_values, y=residuals, mode='markers', name='잔차'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=fitted_values, y=[0]*len(fitted_values), mode='lines', name='0선', line=dict(color='red', dash='dash')),
        row=1, col=2
    )

    # 잔차의 히스토그램
    fig.add_trace(
        go.Histogram(x=residuals, nbinsx=10, name='잔차 히스토그램', marker=dict(color='skyblue')),
        row=2, col=1
    )

    # 잔차 대 순서
    fig.add_trace(
        go.Scatter(x=np.arange(1, len(residuals) + 1), y=residuals, mode='lines+markers', name='잔차'),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=np.arange(1, len(residuals) + 1), y=[0]*len(residuals), mode='lines', name='0선', line=dict(color='red', dash='dash')),
        row=2, col=2
    )

    # 레이아웃 설정
    fig.update_layout(
        height=800, width=800,
        title_text='잔차 그림',
        showlegend=False
    )

    fig.update_xaxes(title_text='이론적 Quantiles', row=1, col=1)
    fig.update_yaxes(title_text='샘플 Quantiles', row=1, col=1)

    fig.update_xaxes(title_text='적합치', row=1, col=2)
    fig.update_yaxes(title_text='잔차', row=1, col=2)

    fig.update_xaxes(title_text='잔차', row=2, col=1)
    fig.update_yaxes(title_text='빈도', row=2, col=1)

    fig.update_xaxes(title_text='순서', row=2, col=2)
    fig.update_yaxes(title_text='잔차', row=2, col=2)

    st.plotly_chart(fig)


###########################################################################################################################
def regression_analysis_set():
    df = convert_to_calculatable_df()
    st.session_state.target = st.selectbox("반응변수", df.columns.tolist())
    predictor = df.columns.tolist()
    predictor.remove(st.session_state.target)
    st.session_state.predictor = st.multiselect("예측변수", predictor)
    if df[st.session_state.target].dtype=="object":
        st.error(f"{st.session_state.target}는 계량형 변수가 아닙니다. 계량형 변수만 선택해주세요.")
        return

    st.session_state.coding_type = st.radio("Coding Type", ["dummy(1, 0)", "effect(1, 0, -1)"])
    if st.toggle("교호작용"):
        interactions = generate_interactions(df)
        st.session_state.interactions = st.multiselect("교호작용", interactions, interactions)

    # if st.toggle("변수선택"):
    #     with st.container(border=True):
    #         st.session_state.var_selection = st.selectbox("방법", ["단계적 회귀", "전진 선택", "후진 제거"])
    #         is_disabled_or = st.session_state.var_selection == "전진 선택" or st.session_state.var_selection == "후진 제거"
    #         is_disabled_for = st.session_state.var_selection == "전진 선택"
    #         is_disabled_back = st.session_state.var_selection == "후진 제거"
    #         st.session_state.always_included = st.multiselect("E = 모든 모형에 항 포함", st.session_state.predictor)
    #         st.session_state.initial_included = st.multiselect(
    #             "I = 초기 모형에 항 포함",
    #             list(set(st.session_state.predictor)-set(st.session_state.always_included)),
    #             disabled=is_disabled_or)
    #         st.session_state.alpha_in = st.number_input("입력할 변수에 대한 알파", value=0.15, disabled=is_disabled_back)
    #         st.session_state.alpha_out = st.number_input("제거할 변수에 대한 알파", value=0.15, disabled=is_disabled_for)
    # else:
    st.session_state.var_selection = None

    with st.popover("옵션"):
        st.session_state.residuals = st.selectbox("사용할 잔차", ["정규 잔차", "표준화 잔차", "외적 스튜던트화 잔차"])
        lambd = st.radio("Box-Cox 변환", ["변환 없음", "최적 λ", "λ = 0(자연로그)", "λ = 0.5(제곱근)", "λ"])
        if lambd == "변환 없음":
            st.session_state.lambd = "X"
        elif lambd == "최적 λ":
            st.session_state.lambd = None
        elif lambd =="λ":
            st.session_state.lambd = st.number_input("", label_visibility="collapsed", step=1)
        else:
            st.session_state.lambd = float(re.findall(r'-?\d+(?:\.\d+)?', lambd)[0])
        
            
def regression_analysis_cal(df):
    run_regression(df)
    if len(st.session_state.predictor)>0:
        run_anova(df)


def regression_analysis_plot(df):
    draw_pareto()
    if st.session_state.residuals == "정규 잔차":
        draw_reg_plot(st.session_state.model.resid, st.session_state.model.fittedvalues)
    elif st.session_state.residuals == "표준화 잔차":
        draw_reg_plot(st.session_state.model.resid_pearson, st.session_state.model.fittedvalues)
    elif st.session_state.residuals == "외적 스튜던트화 잔차":
        draw_reg_plot(st.session_state.model.outlier_test()["student_resid"], st.session_state.model.fittedvalues)


def regression_analysis_run():  
    df = convert_to_calculatable_df()
    if len(st.session_state.predictor)==0:
        st.error("예측변수를 선택하지 않았습니다. 모형에 추가할 변수가 없습니다.")
    else:
        regression_analysis_cal(df)
        if len(st.session_state.predictor)>0:
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
            # --------------------------------------
            with st.expander("결과 그래프", expanded=False):
                # with st.container(border=True):
                regression_analysis_plot(df)