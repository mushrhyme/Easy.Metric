import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import time
import plotly.express as px
from pathlib import Path

# 페이지 설정
st.set_page_config(
    layout="wide",
    page_title="시스템 관리",
    page_icon="🔧",
    initial_sidebar_state="expanded"
)


def load_css(css_file):
    with open(css_file, 'r') as f:
        return f.read()


# CSS 적용
st.markdown(f'<style>{load_css("/Users/macpro2/Desktop/Minitab/utils/style.css")}</style>', unsafe_allow_html=True)


def save_question(question_data, file_path='questions.json'):
    """Q&A 데이터를 JSON 파일에 저장"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                questions = json.load(f)
        else:
            questions = []

        questions.append(question_data)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(questions, f, ensure_ascii=False, indent=2)

        return True
    except Exception as e:
        st.error(f"질문 저장 중 오류 발생: {str(e)}")
        return False


def load_recent_error():
    """현재 로그인한 사용자의 가장 최근 에러 로그 파일을 읽어옴"""
    try:
        # 현재 로그인한 사용자 ID 확인
        current_user_id = st.session_state.get('id')
        if not current_user_id:
            return None

        # error_logs 디렉토리의 모든 에러 로그 파일 검색
        error_files = list(Path('error_logs').glob('error_*.json'))
        if not error_files:
            return None

        # 사용자의 최근 에러 찾기
        user_error_files = []
        for file in error_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    error_data = json.load(f)
                    # 현재 사용자의 에러인 경우에만 추가
                    if error_data['context']['user_id'] == current_user_id:
                        user_error_files.append((file, error_data))
            except Exception as e:
                st.warning(f"파일 {file.name} 읽기 실패: {str(e)}")
                continue

        # 사용자의 에러가 없는 경우
        if not user_error_files:
            return None

        # 타임스탬프를 기준으로 가장 최근 에러 찾기
        latest_error = max(user_error_files,
                         key=lambda x: datetime.fromisoformat(x[1]['timestamp']))
        latest_file, error_data = latest_error

        return {
            'timestamp': datetime.fromisoformat(error_data['timestamp']),
            'error_type': error_data['error_type'],
            'error_message': error_data['error_message'],
            'traceback': error_data['traceback'],
            'context': error_data['context'],
            'file_name': latest_file.name
        }
    except Exception as e:
        st.warning(f"에러 로그 읽기 실패: {str(e)}")
        return None

def display_qa_form():
    """일반 사용자를 위한 Q&A 폼"""
    st.title("❓ 문의사항 등록")

    # 로그인 상태 확인
    if 'id' not in st.session_state or 'name' not in st.session_state:
        st.warning("로그인을 먼저 해 주세요")
        return  # 로그인하지 않은 경우 여기서 함수 종료

    # 로그인된 경우 사용자 정보 표시
    user_id = st.session_state.id
    st.info(f"사용자: {st.session_state.name} ({user_id})")

    # 최근 에러 로그 확인
    recent_error = load_recent_error()
    if recent_error and recent_error['context']['user_id'] == user_id:
        with st.expander("최근 발생한 에러 정보", expanded=True):
            st.info(f"""
📅 발생 시간: {recent_error['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
🔍 에러 유형: {recent_error['error_type']}
❌ 에러 메시지: {recent_error['error_message']}
📍 발생 위치: {recent_error['context']['current_tab']}
""")
            if st.toggle("상세 트레이스백 보기"):
                st.code(recent_error['traceback'])

            # 에러 정보를 폼에 자동으로 채우기 위한 버튼
            if st.button("이 에러에 대해 문의하기"):
                st.session_state.fill_error_info = recent_error

    # 문의사항 입력 폼
    with st.form("question_form"):
        # 자동 채우기 기본값 설정
        default_title = ""
        default_category = "일반 문의"
        default_content = ""

        # 에러 정보 자동 채우기가 요청된 경우
        if hasattr(st.session_state, 'fill_error_info') and st.session_state.fill_error_info:
            error = st.session_state.fill_error_info
            default_title = f"에러 발생 문의: {error['error_type']}"
            default_category = "버그 신고"
            default_content = f"""
발생 시간: {error['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
에러 유형: {error['error_type']}
에러 메시지: {error['error_message']}
발생 위치: {error['context']['current_tab']}

[자동 생성된 에러 정보]
로그 파일: {error['file_name']}

상세 트레이스백:
{error['traceback']}

추가 설명:
"""

        title = st.text_input("제목", value=default_title)
        category = st.selectbox(
            "카테고리",
            ["일반 문의", "버그 신고", "기능 제안", "기타"],
            index=["일반 문의", "버그 신고", "기능 제안", "기타"].index(default_category)
        )
        content = st.text_area("내용", value=default_content, height=300)

        submitted = st.form_submit_button("제출하기")

        if submitted:
            if not title.strip() or not content.strip():
                st.error("제목과 내용을 모두 입력해주세요.")
                return

            question_data = {
                "title": title,
                "category": category,
                "content": content,
                "user_id": st.session_state.id,
                "user_name": st.session_state.name,
                "timestamp": datetime.now().isoformat(),
                "status": "접수됨"
            }

            if save_question(question_data):
                st.success("문의사항이 성공적으로 등록되었습니다!")
                # 성공적으로 저장된 후에 session_state 초기화
                if hasattr(st.session_state, 'fill_error_info'):
                    del st.session_state.fill_error_info
@st.cache_data(ttl=600)  # 1분 캐시
def load_error_logs(log_dir='error_logs'):
    """모든 에러 로그 파일을 읽어서 DataFrame으로 변환"""

    error_logs = []
    for file in Path(log_dir).glob('error_*.json'):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                log = json.load(f)
                error_info = {
                    'timestamp': datetime.fromisoformat(log['timestamp']),
                    'error_type': log['error_type'],
                    'error_message': log['error_message'],
                    'user_id': log['context']['user_id'],
                    'name': log['context']['user_name'],
                    'current_tab': log['context']['current_tab'],
                    'file_name': file.name,
                    'traceback': log['traceback'],
                }
                error_logs.append(error_info)
        except Exception as e:
            st.warning(f"로그 파일 {file.name} 읽기 실패: {str(e)}")
    return pd.DataFrame(error_logs)

def display_statistics(filtered_df):
    """통계 정보를 표시하는 함수"""
    # 중복 제거를 위한 처리
    deduped_df = filtered_df.copy()

    # 날짜만 추출 (시간 정보 제거)
    deduped_df['date'] = deduped_df['timestamp'].dt.date
    # 같은 날짜, 같은 사용자, 같은 traceback을 가진 에러를 하나로 처리
    deduped_df = deduped_df.drop_duplicates(subset=['date', 'user_id', 'traceback'])

    # 처리 상태 확인 및 통계 계산
    total_errors = len(deduped_df)
    unhandled_errors = len(deduped_df[deduped_df['status'] == 'unhandled'])
    handled_ratio = ((total_errors - unhandled_errors) / total_errors * 100) if total_errors > 0 else 0

    # 기본 통계 표시
    col1, col2, col3 = st.columns(3)
    with col1:
        # 중복 제거된 데이터로 총 에러 수 표시
        st.metric("총 에러 수", total_errors)
    with col2:
        st.metric("미처리 에러 수", unhandled_errors)
    with col3:
        st.metric("처리 완료율", f"{handled_ratio:.1f}%")

    # 일별 에러 발생 추이 (처리/미처리 구분)
    st.subheader("📈 일별 에러 발생 현황")

    # 날짜별, 상태별로 집계
    daily_status = deduped_df.groupby(['date', 'status']).size().unstack(fill_value=0)

    if 'unhandled' not in daily_status.columns:
        daily_status['unhandled'] = 0

    if 'handled' not in daily_status.columns:
        daily_status['handled'] = 0
    daily_status.columns = ['처리완료', '미처리']
    daily_status.index = pd.to_datetime(daily_status.index)
    # date_range = pd.date_range(start=daily_status.index.min(), end=daily_status.index.max(), freq='1W')
    # formatted_dates = [d.strftime('%Y년 %m월 %d일') for d in date_range]
    daily_status['formatted_date'] = daily_status.index.strftime('%Y/%m/%d')

    fig = px.bar(
        daily_status,  # daily_status를 직접 사용
        title="일별 에러 발생 현황 (처리/미처리)",
        color_discrete_map={'처리완료': '#00CC96', '미처리': '#EF553B'},
        barmode='stack',
        custom_data=['formatted_date']
    )
    fig.update_layout(
        xaxis_title="날짜",
        yaxis_title="에러 발생 수",
        xaxis=dict(
            tickformat='%Y/%m/%d',
        #     tickmode='array',
        #     ticktext=formatted_dates,
        #     tickvals=date_range,
        #     tickangle=45  # 날짜가 겹치지 않도록 45도 기울임
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    fig.update_traces(
        hovertemplate="%{customdata[0]} : %{value}건<extra></extra>"
    )

    st.plotly_chart(fig)
def get_error_status(error_id, file_path='error_status.json'):
    """특정 에러의 처리 상태 조회"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                status_data = json.load(f)
                return status_data.get(error_id, {}).get('status', 'unhandled')
        return 'unhandled'
    except Exception:
        return 'unhandled'


def initialize_error_status(error_logs_dir='error_logs', status_file='error_status.json'):
    """에러 상태 파일을 초기화하는 함수"""
    try:
        # 모든 에러 로그 파일 검색
        error_files = list(Path(error_logs_dir).glob('error_*.json'))

        # 현재 상태 데이터 로드 또는 새로 생성
        if os.path.exists(status_file):
            with open(status_file, 'r', encoding='utf-8') as f:
                status_data = json.load(f)
        else:
            status_data = {}

        # 모든 에러 파일에 대해 상태 초기화
        for file in error_files:
            if file.name not in status_data:
                status_data[file.name] = {
                    'status': 'unhandled',
                    'updated_at': datetime.now().isoformat(),
                    'updated_by': None
                }

        # 상태 파일 저장
        with open(status_file, 'w', encoding='utf-8') as f:
            json.dump(status_data, f, ensure_ascii=False, indent=2)

        return True
    except Exception as e:
        st.error(f"상태 파일 초기화 중 오류 발생: {str(e)}")
        return False


def update_error_status(error_id, new_status, file_path='error_status.json'):
    """에러 처리 상태를 업데이트"""
    try:
        # 상태 파일 초기화
        initialize_error_status()

        # 현재 상태 로드
        with open(file_path, 'r', encoding='utf-8') as f:
            status_data = json.load(f)

        # 상태 업데이트
        status_data[error_id] = {
            'status': new_status,
            'updated_at': datetime.now().isoformat(),
            'updated_by': st.session_state.id
        }

        # 파일 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(status_data, f, ensure_ascii=False, indent=2)

        return True
    except Exception as e:
        st.error(f"상태 업데이트 중 오류 발생: {str(e)}")
        return False

def display_error_tab_content(filtered_df, status_type):
    """각 탭의 내용을 표시하는 함수"""
    # 필터링된 데이터에서 해당 상태의 에러만 표시
    tab_df = filtered_df[filtered_df['status'] == ('unhandled' if status_type == 'unhandled' else 'handled')]

    if tab_df.empty:
        st.info(f"필터링된 결과 중 {'미처리' if status_type == 'unhandled' else '처리된'} 에러가 없습니다.")
        return

    # 페이지네이션 설정
    items_per_page = 5
    total_pages = len(tab_df) // items_per_page + (1 if len(tab_df) % items_per_page > 0 else 0)

    if total_pages > 0:
        subcol1, subcol2 = st.columns([3, 7])
        with subcol1:
            page_number = st.number_input("페이지", min_value=1, max_value=total_pages, value=1, key=f"error_page_{status_type}")

        start_idx = (page_number - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(tab_df))
        page_df = tab_df.iloc[start_idx:end_idx]

        # 전체 선택 체크박스
        all_selected = st.checkbox(
            "전체 선택",
            key=f"all_select_{status_type}",
            value=status_type == 'handled'
        )

        # 선택된 에러들을 저장할 세션 상태
        if f'selected_errors_{status_type}' not in st.session_state:
            st.session_state[f'selected_errors_{status_type}'] = set()

        # 전체 선택 시 모든 에러 선택
        if all_selected:
            st.session_state[f'selected_errors_{status_type}'] = set(page_df['file_name'])

        # 에러 목록 표시
        for _, error in page_df.iterrows():
            col1, col2 = st.columns([0.1, 0.9])

            with col1:
                # 체크박스 상태 설정
                is_checked = st.checkbox(
                    "##",
                    key=f"check_{error['file_name']}",
                    value=error['file_name'] in st.session_state[
                        f'selected_errors_{status_type}'] or status_type == 'handled'
                )

                # 체크박스 상태 저장
                if is_checked:
                    st.session_state[f'selected_errors_{status_type}'].add(error['file_name'])
                else:
                    st.session_state[f'selected_errors_{status_type}'].discard(error['file_name'])

            with col2:
                # 최근 업데이트된 항목 하이라이트
                highlight_style = "background-color: #90EE90;" if error['recently_updated'] else ""

                st.markdown(
                    f"""
                    <div style='{highlight_style}'>
                        <strong>{error['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</strong> ({error['name']})<br>
                        {error['error_message'][:50]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # 선택된 에러 확인
        selected_errors = st.session_state[f'selected_errors_{status_type}']

        # 상태 업데이트를 위한 컬럼 생성
        col1, col2, col3 = st.columns([2, 1, 2])

        with col2:
            # 상태 업데이트 버튼
            if st.button(
                    "처리 완료로 변경" if status_type == 'unhandled' else "미처리로 변경",
                    key=f"update_button_{status_type}",
                    disabled=len(selected_errors) == 0  # 선택된 에러가 없으면 비활성화
            ):
                if not selected_errors:
                    st.warning("선택된 에러가 없습니다.")
                    return

                # 상태 업데이트
                new_status = 'handled' if status_type == 'unhandled' else 'unhandled'
                success = True
                updated_count = 0

                for error_id in selected_errors:
                    if update_error_status(error_id, new_status):
                        updated_count += 1
                    else:
                        success = False

                if success:
                    st.success(f"{updated_count}개의 에러 상태가 변경되었습니다.")
                    # 세션 상태 초기화
                    st.session_state[f'selected_errors_{status_type}'] = set()
                    # 상태 변경 완료 플래그 설정
                    st.session_state.status_updated = True
                    st.rerun()
                else:
                    st.error("일부 에러 상태 변경에 실패했습니다.")



def display_error_list(df):
    """상세 에러 목록을 표시하는 함수"""
    initialize_error_status()
    st.subheader("🔍 필터 옵션")

    # 날짜 필터
    col1, col2, col3 = st.columns([2, 4, 4])
    with col1:
        start_date = st.date_input(
            "시작 날짜",
            value=df['timestamp'].min().date(),
            min_value=df['timestamp'].min().date(),
            max_value=df['timestamp'].max().date()
        )

        end_date = st.date_input(
            "종료 날짜",
            value=df['timestamp'].max().date(),
            min_value=df['timestamp'].min().date(),
            max_value=df['timestamp'].max().date()
        )

        # 날짜로 필터링
        mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
        filtered_df = df[mask]

        # 사용자 이름 검색
        user_search = st.text_input("사용자 이름 검색", "")
        if user_search:
            filtered_df = filtered_df[filtered_df['name'].str.contains(user_search, case=False, na=False)]

        # 정렬 옵션
        sort_options = {
            '최신순': ('timestamp', False),
            '오래된순': ('timestamp', True),
            '사용자명순': ('name', True),
            '에러타입순': ('error_type', True)
        }
        selected_sort = st.selectbox("정렬 기준", list(sort_options.keys()))
        sort_column, ascending = sort_options[selected_sort]
        filtered_df = filtered_df.sort_values(by=sort_column, ascending=ascending)
        filtered_df['recently_updated'] = filtered_df['file_name'].apply(
            lambda x: (datetime.now() - datetime.fromisoformat(
                json.load(open('error_status.json', 'r'))
                .get(x, {}).get('updated_at', '2000-01-01T00:00:00')
            )).total_seconds() < 300  # 5분 이내 업데이트된 항목
        )

        # 처리 상태 필터
        selected_type = st.selectbox("처리 상태", ["전체", "미처리", "처리"])
        if selected_type=="처리":
            filtered_df = filtered_df[filtered_df['status'] == "handled"]
        elif selected_type == "미처리":
            filtered_df = filtered_df[filtered_df['status'] == "unhandled"]

    with col2:
        # 페이지네이션 설정
        items_per_page = 10
        total_pages = len(filtered_df) // items_per_page + (1 if len(filtered_df) % items_per_page > 0 else 0)

        if total_pages > 0:
            st.write(f"총 {len(filtered_df)}개 항목")
            subcol1, subcol2 = st.columns([3, 7])
            with subcol1:
                page_number = st.number_input("페이지", min_value=1, max_value=total_pages, value=1, key="error_page")

            start_idx = (page_number - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, len(filtered_df))
            filtered_df['date'] = filtered_df['timestamp'].dt.date
            page_df = filtered_df.iloc[start_idx:end_idx].drop_duplicates(subset=['date', 'user_id', 'traceback'])

        else:
            st.write("검색 결과가 없습니다.")
            return

        # 에러 목록을 테이블로 먼저 보여주기
        st.dataframe(
            page_df[['timestamp', 'name', 'error_type', 'error_message', 'status']]
            .rename(columns={
                'timestamp': '발생시간',
                'name': '사용자명',
                'error_type': '에러유형',
                'error_message': '에러메시지',
                'status': '처리상태'
            }),
            width=1000,
            hide_index=True
        )

    with col3:
        # 에러 목록 탭 생성
        unhandled_tab, handled_tab = st.tabs(["🔴 미처리 에러", "✅ 처리된 에러"])

        # 미처리 에러 탭
        with unhandled_tab:
            unhandled_errors = filtered_df[filtered_df['status'] == 'unhandled']
            display_error_tab_content(unhandled_errors, 'unhandled')

        # 처리된 에러 탭
        with handled_tab:
            handled_errors = filtered_df[filtered_df['status'] == 'handled']
            display_error_tab_content(handled_errors, 'handled')
    # 구분선 추가
    st.markdown("---")

    # 선택된 에러의 상세 정보 표시
    selected_error = st.selectbox(
        "상세 정보를 볼 에러를 선택하세요",
        page_df.apply(
            lambda x: f"{x['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - {x['name']} - {x['error_type']}",
            axis=1
        ).tolist()
    )

    if selected_error:
        selected_row = page_df[page_df.apply(
            lambda x: f"{x['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - {x['name']} - {x['error_type']}" == selected_error,
            axis=1
        )].iloc[0]

        with st.expander("🔍 상세 정보", expanded=True):
            tab1, tab2 = st.tabs(["기본 정보", "전체 트레이스백"])

            with tab1:
                st.write(f"**사용자:** {selected_row['user_id']} ({selected_row['name']})")
                st.write(f"**현재 탭:** {selected_row['current_tab']}")
                st.write(f"**에러 메시지:** {selected_row['error_message']}")

            with tab2:
                st.code(selected_row['traceback'])

    # 상태 변경 완료 플래그 초기화
    if 'status_updated' not in st.session_state:
        st.session_state.status_updated = False



def load_visitor_logs(file_path):
    """방문자 로그를 DataFrame으로 변환"""
    try:
        with open(file_path, 'r') as f:
            logs = json.load(f)
        df = pd.DataFrame(logs)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.warning(f"방문자 로그 읽기 실패: {str(e)}")
        return pd.DataFrame(columns=['timestamp', 'ip_address'])


def display_visitor_stats(df):
    """방문자 통계를 표시하는 함수"""
    if df.empty:
        st.info("방문자 로그가 없습니다.")
        return

    # 기본 통계
    col1, col2, col3 = st.columns(3)

    total_visits = len(df)
    unique_ips = df['ip_address'].nunique()

    # 일평균 방문자 수 계산 로직 수정
    date_diff = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400  # 일 단위로 변환
    if date_diff < 1:  # 하루 미만인 경우
        avg_daily_visits = total_visits  # 당일 방문자 수를 그대로 사용
    else:
        avg_daily_visits = round(total_visits / date_diff, 1)

    with col1:
        st.metric("총 방문 수", total_visits)
    with col2:
        st.metric("고유 방문자 수", unique_ips)
    with col3:
        if date_diff < 1:
            st.metric("오늘의 방문", total_visits)
        else:
            st.metric("일평균 방문", avg_daily_visits)

    # 일별 방문자 수 추이
    st.subheader("📈 일별 방문자 추이")
    daily_visits = df.groupby(df['timestamp'].dt.date).agg({
        'ip_address': ['count', 'nunique']
    }).reset_index()
    daily_visits.columns = ['date', 'total_visits', 'unique_visitors']

    fig_daily = px.line(
        daily_visits,
        x='date',
        y=['total_visits', 'unique_visitors'],
        labels={
            'value': '방문자 수',
            'date': '날짜',
            'variable': '구분'
        },
        title='일별 방문자 추이'
    )
    # 범례 이름 변경
    fig_daily.for_each_trace(lambda t: t.update(
        name={'total_visits': '총 방문수', 'unique_visitors': '고유 방문자'}[t.name],
        showlegend=True
    ))
    fig_daily.update_layout(
        legend=dict(
            title=None,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    st.plotly_chart(fig_daily)

    # 시간대별 방문자 수 차트
    st.subheader("📊 시간대별 방문자 수")
    hourly_visits = df.groupby(df['timestamp'].dt.hour)['ip_address'].count()

    # 없는 시간대 0으로 채우기
    all_hours = pd.Series(0, index=range(24))
    hourly_visits = hourly_visits.add(all_hours, fill_value=0)

    fig_hourly = px.bar(
        x=hourly_visits.index,
        y=hourly_visits.values,
        labels={'x': '시간', 'y': '방문자 수'},
        title='시간대별 방문자 분포'
    )
    fig_hourly.update_xaxes(ticktext=[f"{i}시" for i in range(24)], tickvals=list(range(24)))
    st.plotly_chart(fig_hourly)

    # IP 주소별 방문 횟수
    st.subheader("🔍 자주 방문한 IP")
    top_ips = df['ip_address'].value_counts().reset_index()
    top_ips.columns = ['IP 주소', '방문 횟수']

    fig_ips = px.bar(
        top_ips,
        x='IP 주소',
        y='방문 횟수',
        title='방문자 IP별 방문 횟수'
    )
    st.plotly_chart(fig_ips)

    # 방문 기록 테이블
    st.subheader("📋 상세 방문 기록")
    st.dataframe(
        df.sort_values('timestamp', ascending=False)
        .rename(columns={
            'timestamp': '방문 시간',
            'ip_address': 'IP 주소'
        }),
        hide_index=True
    )


@st.cache_data(ttl=3600)  # 1시간 캐시
def load_questions(file_path='questions.json'):
    """문의사항 데이터를 DataFrame으로 변환"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            df = pd.DataFrame(questions)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        return pd.DataFrame(columns=['timestamp', 'title', 'category', 'content', 'user_id', 'user_name', 'status'])
    except Exception as e:
        st.warning(f"문의사항 데이터 읽기 실패: {str(e)}")
        return pd.DataFrame(columns=['timestamp', 'title', 'category', 'content', 'user_id', 'user_name', 'status'])


def update_question_status(questions, index, new_status, file_path='questions.json'):
    """문의사항 상태 업데이트"""
    try:
        questions_copy = questions.copy()
        questions_copy['timestamp'] = questions_copy['timestamp'].astype(str)
        questions_copy.at[index, 'status'] = new_status
        questions_dict = questions_copy.to_dict('records')

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(questions_dict, f, ensure_ascii=False, indent=2)

        # 캐시 즉시 초기화
        load_questions.clear()
        return True
    except Exception as e:
        st.error(f"상태 업데이트 중 오류 발생: {str(e)}")
        return False


def display_questions_dashboard():
    """문의사항 관리 대시보드"""
    st.subheader("📬 문의사항 관리")

    # 데이터 로드
    df = load_questions()
    if df.empty:
        st.info("등록된 문의사항이 없습니다.")
        return

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 필터 옵션 섹션
    st.subheader("🔍 필터 옵션")

    # 날짜 필터
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "시작 날짜",
            value=df['timestamp'].min().date(),
            min_value=df['timestamp'].min().date(),
            max_value=df['timestamp'].max().date()
        )
    with col2:
        end_date = st.date_input(
            "종료 날짜",
            value=df['timestamp'].max().date(),
            min_value=df['timestamp'].min().date(),
            max_value=df['timestamp'].max().date()
        )

    # 날짜로 필터링
    mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
    filtered_df = df[mask]

    # 추가 필터 옵션
    col1, col2, col3= st.columns(3)

    with col1:
        categories = ['전체'] + list(filtered_df['category'].unique())
        selected_category = st.selectbox("카테고리", categories)

    with col2:
        statuses = ['전체'] + list(filtered_df['status'].unique())
        selected_status = st.selectbox("처리상태", statuses)

    with col3:
        user_search = st.text_input("사용자명 검색")

    # 필터링 적용
    if selected_category != '전체':
        filtered_df = filtered_df[filtered_df['category'] == selected_category]
    if selected_status != '전체':
        filtered_df = filtered_df[filtered_df['status'] == selected_status]
    if user_search:
        filtered_df = filtered_df[filtered_df['user_name'].str.contains(user_search, case=False, na=False)]

    # 정렬 옵션
    sort_options = {
        '최신순': ('timestamp', False),
        '오래된순': ('timestamp', True),
        '사용자명순': ('user_name', True),
        '카테고리순': ('category', True)
    }
    selected_sort = st.selectbox("정렬 기준", list(sort_options.keys()))
    sort_column, ascending = sort_options[selected_sort]
    filtered_df = filtered_df.sort_values(by=sort_column, ascending=ascending)

    # 페이지네이션 설정
    items_per_page = 10  # 고정값으로 설정
    total_pages = len(filtered_df) // items_per_page + (1 if len(filtered_df) % items_per_page > 0 else 0)

    if total_pages > 0:
        col1, col2 = st.columns([7, 3])
        with col1:
            st.write(f"총 {len(filtered_df)}개의 항목")
        with col2:
            page_number = st.number_input("페이지", min_value=1, max_value=total_pages, value=1, key="question_page")

        start_idx = (page_number - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(filtered_df))
        page_df = filtered_df.iloc[start_idx:end_idx]
    else:
        st.write("검색 결과가 없습니다.")
        return

    # 통계 요약
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("총 문의", len(df))
    with col2:
        st.metric("처리 대기", len(df[df['status'] == '접수됨']))
    with col3:
        st.metric("처리 중", len(df[df['status'] == '처리중']))
    with col4:
        st.metric("처리 완료", len(df[df['status'] == '완료']))

    # 문의사항 목록을 테이블로 먼저 보여주기
    st.markdown("---")
    st.dataframe(
        page_df[['timestamp', 'user_name', 'category', 'title', 'status']]
        .rename(columns={
            'timestamp': '등록시간',
            'user_name': '작성자',
            'category': '카테고리',
            'title': '제목',
            'status': '상태'
        }),
        hide_index=True
    )

    # 선택된 문의사항의 상세 정보 표시
    st.markdown("---")
    selected_question = st.selectbox(
        "상세 정보를 볼 문의사항을 선택하세요",
        page_df.apply(
            lambda x: f"{x['timestamp'].strftime('%Y-%m-%d %H:%M')} - {x['user_name']} - {x['title']}",
            axis=1
        ).tolist()
    )

    if selected_question:
        selected_row = page_df[page_df.apply(
            lambda
                x: f"{x['timestamp'].strftime('%Y-%m-%d %H:%M')} - {x['user_name']} - {x['title']}" == selected_question,
            axis=1
        )].iloc[0]

        with st.expander("📝 문의사항 상세 정보", expanded=True):
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                st.write(f"**작성자:** {selected_row['user_name']} ({selected_row['user_id']})")
                st.write(f"**카테고리:** {selected_row['category']}")

            with col2:
                st.write(f"**등록시간:** {selected_row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"**현재상태:** {selected_row['status']}")

            with col3:
                new_status = st.selectbox(
                    "상태 변경",
                    ['접수됨', '처리중', '완료'],
                    index=['접수됨', '처리중', '완료'].index(selected_row['status']),
                    key=f"status_{selected_row.name}"
                )
                if new_status != selected_row['status']:
                    if st.button("상태 업데이트", key=f"update_{selected_row.name}"):
                        if update_question_status(df, selected_row.name, new_status):
                            st.success("상태가 업데이트되었습니다.")
                            st.rerun()

            st.markdown("---")
            st.markdown("**제목:**")
            st.markdown(selected_row['title'])
            st.markdown("**문의내용:**")
            st.markdown(selected_row['content'])

def load_login_logs(file_path='login_logs.json'):
    """로그인 로그를 DataFrame으로 변환"""
    try:
        with open(file_path, 'r') as f:
            logs = json.load(f)
        df = pd.DataFrame(logs)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.warning(f"로그인 로그 읽기 실패: {str(e)}")
        return pd.DataFrame(columns=['timestamp', 'user_id', 'user_name', 'ip_address'])

def display_login_stats(df):
    """로그인 통계를 표시하는 함수"""
    if df.empty:
        st.info("로그인 로그가 없습니다.")
        return

    # 기본 통계
    col1, col2, col3 = st.columns(3)

    total_logins = len(df)
    unique_users = df['user_id'].nunique()

    # 일평균 로그인 수 계산
    date_diff = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400  # 일 단위로 변환
    if date_diff < 1:  # 하루 미만인 경우
        avg_daily_logins = total_logins
    else:
        avg_daily_logins = round(total_logins / date_diff, 1)

    with col1:
        st.metric("총 로그인 수", total_logins)
    with col2:
        st.metric("고유 사용자 수", unique_users)
    with col3:
        if date_diff < 1:
            st.metric("오늘의 로그인", total_logins)
        else:
            st.metric("일평균 로그인", avg_daily_logins)

    # 일별 로그인 추이
    st.subheader("📈 일별 로그인 추이")
    daily_logins = df.groupby(df['timestamp'].dt.date).agg({
        'user_id': ['count', 'nunique']
    }).reset_index()
    daily_logins.columns = ['date', 'total_logins', 'unique_users']

    fig_daily = px.line(
        daily_logins,
        x='date',
        y=['total_logins', 'unique_users'],
        labels={
            'value': '로그인 수',
            'date': '날짜',
            'variable': '구분'
        },
        title='일별 로그인 추이'
    )
    fig_daily.for_each_trace(lambda t: t.update(
        name={'total_logins': '총 로그인수', 'unique_users': '고유 사용자'}[t.name],
        showlegend=True
    ))
    fig_daily.update_layout(
        legend=dict(
            title=None,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    st.plotly_chart(fig_daily)

    # 시간대별 로그인 수 차트
    st.subheader("📊 시간대별 로그인 수")
    hourly_logins = df.groupby(df['timestamp'].dt.hour)['user_id'].count()

    # 없는 시간대 0으로 채우기
    all_hours = pd.Series(0, index=range(24))
    hourly_logins = hourly_logins.add(all_hours, fill_value=0)

    fig_hourly = px.bar(
        x=hourly_logins.index,
        y=hourly_logins.values,
        labels={'x': '시간', 'y': '로그인 수'},
        title='시간대별 로그인 분포'
    )
    fig_hourly.update_xaxes(ticktext=[f"{i}시" for i in range(24)], tickvals=list(range(24)))
    st.plotly_chart(fig_hourly)

    # 사용자별 로그인 횟수
    st.subheader("🔍 자주 로그인한 사용자")
    top_users = df.groupby(['user_id', 'user_name'])['timestamp'].count().reset_index()
    top_users.columns = ['사용자 ID', '사용자명', '로그인 횟수']
    top_users = top_users.sort_values('로그인 횟수', ascending=False)

    fig_users = px.bar(
        top_users,
        x='사용자명',
        y='로그인 횟수',
        title='사용자별 로그인 횟수',
        hover_data=['사용자 ID']
    )
    st.plotly_chart(fig_users)

    # 로그인 기록 테이블
    st.subheader("📋 상세 로그인 기록")
    st.dataframe(
        df.sort_values('timestamp', ascending=False)
        .rename(columns={
            'timestamp': '로그인 시간',
            'user_id': '사용자 ID',
            'user_name': '사용자명',
            'ip_address': 'IP 주소'
        }),
        hide_index=True
    )

def display_error_dashboard():
    st.title("🔍 시스템 모니터링 대시보드")

    # 관리자 확인
    if st.session_state.id != "9999999" or st.session_state.name != "admin":  # 관리자 ID 목록
        st.error("접근 권한이 없습니다.")
        return
    # 캐시 초기화 버튼
    col1, col2, _ = st.columns([1, 3, 1])
    with col1:
        if st.button("캐시 초기화", type="primary"):
            load_error_logs.clear()
            load_questions.clear()
            st.success("캐시가 초기화되었습니다!")
            st.rerun()
    with col2:
        st.info("데이터가 제대로 반영되지 않으면 버튼을 눌러주세요.")

    # 데이터 로드
    error_df = load_error_logs()
    error_df['status'] = error_df['file_name'].apply(lambda x: get_error_status(x))
    visitor_df = load_visitor_logs("./visitor_logs.json")
    login_df = load_login_logs()

    # 탭 생성
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 에러 통계", "📝 상세 에러 목록", "👥 방문자 분석", "🔐 로그인 분석", "💬 문의사항 관리"])

    with tab1:
        display_statistics(error_df)

    with tab2:
        display_error_list(error_df)

    with tab3:
        display_visitor_stats(visitor_df)

    with tab4:
        display_login_stats(login_df)

    with tab5:
        display_questions_dashboard()

def main():
    # 관리자 확인
    is_admin = st.session_state.get('id') == '9999999' and st.session_state.get('name') == "admin"

    if is_admin:
        display_error_dashboard()
    else:
        display_qa_form()


if __name__ == "__main__":
    main()